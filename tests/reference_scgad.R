# Reference scGAD computation via pure GenomicRanges (no GenomicInteractions
# dependency). Faithful translation of BandNorm/R/scGAD.R's `getCount`,
# restricted to the in-memory hic_df path.
#
# We use the real mm9Annotations (10180 genes, with strand) shipped with
# BandNorm — not the toy 202-row geneANNO from scGADExample.rda, which has
# no strand column and would silently produce an all-NaN result through
# ifelse(NA, ...).
#
# Usage:
#   Rscript reference_scgad.R <example_rda> <annotations_rda> <output_prefix>

suppressMessages({
  library(data.table)
  library(GenomicRanges)
})

args <- commandArgs(trailingOnly = TRUE)
example_rda <- args[1]
ann_rda <- args[2]
out_prefix <- args[3]

load(example_rda)  # scgad_df, geneANNO (unused)
load(ann_rda)      # mm9Annotations etc.

# Use full mm9 annotation as the gene set (strand-bearing).
geneANNO <- mm9Annotations

# Replicate BandNorm's gene-extension logic verbatim from scGAD.R lines 19-23.
genes <- as.data.frame(geneANNO)
discardCounts <- max(genes$s2 - genes$s1)
genes$s1 <- ifelse(genes$strand == "+", genes$s1 - 1000, genes$s1)
genes$s2 <- ifelse(genes$strand == "-", genes$s2, genes$s2 + 1000)
colnames(genes) <- c("chr", "start", "end", "strand", "names")
genes_gr <- makeGRangesFromDataFrame(genes, keep.extra.columns = TRUE)
g_names <- genes$names

scgad_df <- setDT(scgad_df)
res <- 10000
cell_names <- unique(scgad_df$cell)

get_count <- function(k) {
  cell <- scgad_df[cell == cell_names[k]]
  cell <- cell[abs(binB - binA) <= discardCounts]
  if (nrow(cell) == 0) return(rep(0, length(g_names)))
  anchor1 <- GRanges(cell$chrom, IRanges(cell$binA, width = res))
  anchor2 <- GRanges(cell$chrom, IRanges(cell$binB, width = res))
  hits1 <- findOverlaps(anchor1, genes_gr, select = "all")
  hits2 <- findOverlaps(anchor2, genes_gr, select = "all")
  h1 <- data.frame(queryHits = queryHits(hits1),
                   subjectHits = subjectHits(hits1))
  h2 <- data.frame(queryHits = queryHits(hits2),
                   subjectHits = subjectHits(hits2))
  # `generics::intersect` over two data.frames returns rows in BOTH (deduped).
  hits <- merge(h1, h2)  # inner join on identical (queryHits, subjectHits) -> same as intersect
  # The original code used generics::intersect which returns common DISTINCT rows.
  # merge(..., by=c("queryHits","subjectHits")) returns matching rows;
  # since each (q, s) appears at most once in each side, the result equals intersect.
  if (nrow(hits) == 0) return(rep(0, length(g_names)))
  hits <- as.data.table(hits)
  hits[, reads := cell$count[queryHits]]
  agg <- hits[, .(reads = sum(reads)), by = "subjectHits"]
  out <- rep(0, length(g_names))
  out[agg$subjectHits] <- agg$reads
  out
}

cat("computing scGAD reference for", length(cell_names), "cells...\n")
t0 <- Sys.time()
output <- sapply(seq_along(cell_names), get_count)
output[is.na(output)] <- 0
cat("  raw aggregation: ", round(as.numeric(difftime(Sys.time(), t0, units="secs")),2), "s\n")

rownames(output) <- g_names
colnames(output) <- cell_names

# Save raw counts (before drop / depth-norm / Z-score) so the Python parity
# test can compare each post-processing step independently.
fwrite(as.data.table(output, keep.rownames = "gene"),
       paste0(out_prefix, "_raw.tsv"), sep = "\t")

# Then apply BandNorm's filtering + normalisation steps, exactly as in scGAD.R.
output_f <- output[rowSums(output) > 0, ]
output_f <- output_f[!is.na(rowSums(output_f)), ]
output_n <- t(t(output_f) / colSums(output_f)) * 1e4
mat_var <- apply(output_n, 1, var)
mat_mean <- rowMeans(output_n)
GAD <- (output_n - mat_mean) / sqrt(mat_var)

fwrite(as.data.table(output_n, keep.rownames = "gene"),
       paste0(out_prefix, "_depth_norm.tsv"), sep = "\t")
fwrite(as.data.table(GAD, keep.rownames = "gene"),
       paste0(out_prefix, "_gad.tsv"), sep = "\t")

cat("output:", nrow(GAD), "genes ×", ncol(GAD), "cells\n")
cat("done.\n")
