"""bandnorm-rs: Rust port of BandNorm scGAD + bandnorm normalization.

Drop-in replacement for the inner aggregation loops of
https://github.com/sshen82/BandNorm — bit-equivalent (within f64
epsilon) and ~10–30× faster on real scHi-C datasets.

Public API:
    scgad(hic_df, genes, resolution=10_000, depth_norm=True)
    bandnorm(hic_df)
    set_num_threads(n)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from bandnorm_rs._rust import (
        py_scgad_compute as _scgad_compute,
        py_bandnorm_compute as _bandnorm_compute,
        set_num_threads as _set_num_threads,
    )
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    _scgad_compute = None
    _bandnorm_compute = None
    _set_num_threads = None


def set_num_threads(n: int) -> bool:
    """Set the number of rayon worker threads used by the Rust backend."""
    if _set_num_threads is None:
        return False
    return _set_num_threads(n)


def _encode_chrom(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Encode chromosome strings as u16 ids. Returns (codes, levels)."""
    series = pd.Categorical(values)
    codes = series.codes.astype(np.uint16, copy=False)
    if (series.codes < 0).any():
        raise ValueError("hic_df has missing chrom values (NaN).")
    return codes, list(series.categories)


def _extend_genes(genes: pd.DataFrame) -> pd.DataFrame:
    """Apply BandNorm's promoter extension (literal R behaviour, including
    the asymmetric `+`/`-` strand handling — see scGAD.R lines 19-23)."""
    g = genes.copy()
    s1 = g["s1"].to_numpy(copy=True)
    s2 = g["s2"].to_numpy(copy=True)
    strand = g["strand"].to_numpy()
    plus = strand == "+"
    minus = strand == "-"
    not_minus = ~minus
    # Replicate R: ifelse(strand == "+", s1 - 1000, s1)
    s1[plus] = s1[plus] - 1000
    # Replicate R: ifelse(strand == "-", s2, s2 + 1000)
    s2[not_minus] = s2[not_minus] + 1000
    g["s1"] = s1
    g["s2"] = s2
    return g


def scgad(
    hic_df: pd.DataFrame,
    genes: pd.DataFrame,
    resolution: int = 10_000,
    depth_norm: bool = True,
) -> pd.DataFrame:
    """Compute scGAD scores from in-memory contact pairs (option C in
    BandNorm's tutorial).

    Parameters
    ----------
    hic_df : pandas.DataFrame
        5 columns: ``chrom``, ``binA``, ``binB``, ``count``, ``cell``.
        Order doesn't matter; column names are required.
    genes : pandas.DataFrame
        5 columns: ``chr``, ``s1``, ``s2``, ``strand``, ``gene_name``.
        Same shape as BandNorm's bundled ``mm9Annotations`` /
        ``hg38Annotations`` etc.
    resolution : int
        Bin resolution (default 10_000).
    depth_norm : bool
        Apply column-wise depth normalisation (sum→1e4) and row-wise
        Z-score, matching BandNorm's default behaviour.

    Returns
    -------
    pandas.DataFrame
        Rows = genes (those with non-zero raw counts kept), columns =
        cells. If ``depth_norm`` is True, values are Z-scored GAD scores;
        otherwise raw counts.
    """
    if not _RUST_AVAILABLE:
        raise RuntimeError("Rust backend not available — `maturin develop --release` first.")

    required_hic = {"chrom", "binA", "binB", "count", "cell"}
    missing = required_hic - set(hic_df.columns)
    if missing:
        raise ValueError(f"hic_df missing columns: {missing}")
    required_genes = {"chr", "s1", "s2", "strand", "gene_name"}
    missing = required_genes - set(genes.columns)
    if missing:
        raise ValueError(f"genes missing columns: {missing}")

    # Apply gene extension (BandNorm logic).
    genes_ext = _extend_genes(genes)

    # Build a unified chromosome dictionary from both tables, so both share ids.
    all_chroms = pd.unique(np.concatenate([
        hic_df["chrom"].to_numpy(),
        genes_ext["chr"].to_numpy(),
    ]))
    chrom_to_id = {c: i for i, c in enumerate(all_chroms)}
    pair_chrom = np.array([chrom_to_id[c] for c in hic_df["chrom"].to_numpy()],
                          dtype=np.uint16)
    gene_chrom = np.array([chrom_to_id[c] for c in genes_ext["chr"].to_numpy()],
                          dtype=np.uint16)

    pair_pos_a = hic_df["binA"].to_numpy().astype(np.uint32, copy=False)
    pair_pos_b = hic_df["binB"].to_numpy().astype(np.uint32, copy=False)
    pair_count = hic_df["count"].to_numpy().astype(np.float64, copy=False)
    cells = pd.Categorical(hic_df["cell"].to_numpy())
    pair_cell = cells.codes.astype(np.int64)
    n_cells = len(cells.categories)

    # Sort all per-pair arrays by cell so the Rust core can slice contiguously.
    order = np.argsort(pair_cell, kind="stable")
    pair_chrom = pair_chrom[order]
    pair_pos_a = pair_pos_a[order]
    pair_pos_b = pair_pos_b[order]
    pair_count = pair_count[order]
    pair_cell = pair_cell[order]

    # Build cell offsets: cell_offsets[c] = first row index of cell c.
    cell_offsets = np.zeros(n_cells + 1, dtype=np.uint32)
    counts_per_cell = np.bincount(pair_cell, minlength=n_cells)
    cell_offsets[1:] = np.cumsum(counts_per_cell)

    gene_start = genes_ext["s1"].to_numpy().astype(np.uint32, copy=False)
    gene_end = genes_ext["s2"].to_numpy().astype(np.uint32, copy=False)

    discard_counts = int((genes["s2"] - genes["s1"]).max())  # original (non-extended) gene span max

    raw = _scgad_compute(
        pair_chrom, pair_pos_a, pair_pos_b, pair_count,
        cell_offsets, n_cells,
        gene_chrom, gene_start, gene_end,
        int(resolution), int(discard_counts),
    )
    # raw has shape (n_genes, n_cells). gene order matches `genes` row order.
    raw_df = pd.DataFrame(raw,
                          index=genes["gene_name"].to_numpy(),
                          columns=list(cells.categories))

    if not depth_norm:
        return raw_df

    # Filter out rows whose total count is zero or NaN (matches R `output[rowSums(output) > 0, ]`).
    row_sums = raw_df.sum(axis=1).to_numpy()
    keep = (row_sums > 0) & np.isfinite(row_sums)
    out = raw_df.iloc[keep]

    # Column-wise depth normalisation: each column sums to 1e4.
    col_sums = out.sum(axis=0).to_numpy()
    out = out.div(col_sums, axis=1) * 1e4
    # Row-wise Z-score: (x - rowMean) / sqrt(rowVar). R uses (n-1) denominator.
    row_mean = out.mean(axis=1).to_numpy()
    row_var = out.var(axis=1, ddof=1).to_numpy()
    gad = (out.sub(row_mean, axis=0)).div(np.sqrt(row_var), axis=0)
    return gad


def bandnorm(hic_df: pd.DataFrame) -> pd.DataFrame:
    """BandNorm normalisation. Mirrors ``BandNorm/R/bandnorm.R``.

    Input  : DataFrame with columns ``chrom, binA, binB, count, cell``.
    Output : same DataFrame with ``count`` replaced by ``BandNorm`` values
             (``count / band_depth(chrom,diag,cell) * mean_band_depth(chrom,diag)``)
             and an explicit ``diag = abs(binB - binA)`` column added.
    """
    if not _RUST_AVAILABLE:
        raise RuntimeError("Rust backend not available — `maturin develop --release` first.")

    df = hic_df.copy()
    df["diag"] = np.abs(df["binB"].to_numpy() - df["binA"].to_numpy()).astype(np.uint32)

    chrom_codes, _ = _encode_chrom(df["chrom"].to_numpy())
    cells = pd.Categorical(df["cell"].to_numpy())
    cell_codes = cells.codes.astype(np.uint32)

    out = _bandnorm_compute(
        chrom_codes,
        df["diag"].to_numpy().astype(np.uint32, copy=False),
        cell_codes,
        df["count"].to_numpy().astype(np.float64, copy=False),
    )
    df["BandNorm"] = np.asarray(out)
    return df.drop(columns=["count"])


__all__ = ["scgad", "bandnorm", "set_num_threads"]
