// bandnorm-rust: Rust port of BandNorm scGAD + bandnorm.
//
// Goal: bit-equivalent (within f64 epsilon) to the upstream R
// implementation at https://github.com/sshen82/BandNorm.
//
// scGAD pipeline mirrored from R/scGAD.R:
//   1. Extend gene intervals (R-side promoter logic, replicated literally
//      including its quirky `+` / `-` strand asymmetry).
//   2. For every contact pair (chrA, binA, chrB, binB, count) with
//      |binB - binA| <= max_gene_length and chrA == chrB:
//        - Anchor1 = [binA, binA + res - 1], Anchor2 = [binB, binB + res - 1]
//        - Find genes overlapping each anchor.
//        - For genes overlapping BOTH anchors: counts[gene_idx] += count.
//   3. Stack into matrix (n_genes × n_cells).
//
// bandnorm pipeline mirrored from R/bandnorm.R:
//   - For each (chrom, diag, cell): band_depth = sum(count)
//   - For each (chrom, diag): alpha = mean(band_depth across cells)
//   - normalized_count = count / band_depth * alpha
//
// All cell-level loops parallelised via rayon.

use ahash::AHashMap;
use ndarray::{Array2, Axis, ShapeBuilder};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

// ============================================================
// Gene index — built once, shared across cells.
// ============================================================
//
// Per-chromosome: genes sorted by `start`. Plus a per-bin lookup table
// (bin_idx → vec of gene_idx that overlap that bin) for O(1) query when
// anchors are bin-aligned (the common scHi-C case).
struct ChromIndex {
    // All arrays sorted by start (paired sort).
    starts: Vec<u32>,
    ends: Vec<u32>,
    gene_idx: Vec<u32>,               // original gene-table row index
    // bins[b] = positions (into starts/ends/gene_idx) of genes overlapping
    // bin [b*res, b*res + res - 1].
    bins: Vec<Vec<u32>>,
    bin_size: u32,
}

impl ChromIndex {
    fn new(starts: Vec<u32>, ends: Vec<u32>, gene_idx: Vec<u32>, bin_size: u32) -> Self {
        let mut perm: Vec<usize> = (0..starts.len()).collect();
        perm.sort_unstable_by_key(|&i| starts[i]);
        let starts: Vec<u32> = perm.iter().map(|&i| starts[i]).collect();
        let ends: Vec<u32> = perm.iter().map(|&i| ends[i]).collect();
        let gene_idx: Vec<u32> = perm.iter().map(|&i| gene_idx[i]).collect();

        let max_bin = ends.iter().copied().max().unwrap_or(0) / bin_size + 1;
        let mut bins: Vec<Vec<u32>> = vec![Vec::new(); max_bin as usize + 1];
        for pos in 0..starts.len() {
            let first = (starts[pos] / bin_size) as usize;
            let last = (ends[pos] / bin_size) as usize;
            for b in first..=last.min(bins.len() - 1) {
                bins[b].push(pos as u32);
            }
        }

        Self { starts, ends, gene_idx, bins, bin_size }
    }

    /// Push gene_idx values for genes overlapping anchor [p, p + bin_size - 1]
    /// into `scratch`. `scratch` is reused/resized per call.
    fn anchor_genes<'a>(&self, p: u32, scratch: &'a mut Vec<u32>) -> &'a [u32] {
        scratch.clear();
        let lo_bin = (p / self.bin_size) as usize;
        let hi_bin = ((p + self.bin_size - 1) / self.bin_size) as usize;
        if lo_bin >= self.bins.len() {
            return &scratch[..];
        }
        let q_end = p + self.bin_size - 1;
        if lo_bin == hi_bin {
            for &pos in &self.bins[lo_bin] {
                let pos = pos as usize;
                if self.starts[pos] <= q_end && self.ends[pos] >= p {
                    scratch.push(self.gene_idx[pos]);
                }
            }
        } else {
            let hi = hi_bin.min(self.bins.len() - 1);
            for b in lo_bin..=hi {
                for &pos in &self.bins[b] {
                    let pos = pos as usize;
                    let gi = self.gene_idx[pos];
                    if scratch.contains(&gi) { continue; }
                    if self.starts[pos] <= q_end && self.ends[pos] >= p {
                        scratch.push(gi);
                    }
                }
            }
        }
        &scratch[..]
    }
}

struct GeneIndex {
    chrom: AHashMap<u16, ChromIndex>,
    n_genes: usize,
}

impl GeneIndex {
    fn build(
        gene_chrom: &[u16],
        gene_start: &[u32],
        gene_end: &[u32],
        bin_size: u32,
    ) -> Self {
        let n_genes = gene_chrom.len();
        let mut by_chrom: AHashMap<u16, (Vec<u32>, Vec<u32>, Vec<u32>)> = AHashMap::new();
        for k in 0..n_genes {
            let entry = by_chrom.entry(gene_chrom[k]).or_default();
            entry.0.push(gene_start[k]);
            entry.1.push(gene_end[k]);
            entry.2.push(k as u32);
        }
        let chrom = by_chrom
            .into_iter()
            .map(|(c, (s, e, idx))| (c, ChromIndex::new(s, e, idx, bin_size)))
            .collect();
        Self { chrom, n_genes }
    }
}

// ============================================================
// scGAD core: per-pair aggregation.
// ============================================================

fn scgad_one_cell(
    pair_chrom: &[u16],
    pair_pos_a: &[u32],
    pair_pos_b: &[u32],
    pair_count: &[f64],
    gene_idx: &GeneIndex,
    discard_counts: u32,
    out: &mut [f64],
) {
    let mut buf_a: Vec<u32> = Vec::with_capacity(8);
    let mut buf_b: Vec<u32> = Vec::with_capacity(8);
    for k in 0..pair_chrom.len() {
        let a = pair_pos_a[k];
        let b = pair_pos_b[k];
        let span = if b > a { b - a } else { a - b };
        if span > discard_counts {
            continue;
        }
        let chr = pair_chrom[k];
        let chrom_idx = match gene_idx.chrom.get(&chr) {
            Some(c) => c,
            None => continue,
        };
        // Stash genes_a into a fresh Vec so we still own it after a second
        // anchor_genes call reuses buf_a's storage. (anchor_genes returns
        // &[u32] tied to its scratch buffer.)
        let genes_a: Vec<u32> = chrom_idx.anchor_genes(a, &mut buf_a).to_vec();
        let genes_b = chrom_idx.anchor_genes(b, &mut buf_b);
        for ga in &genes_a {
            if genes_b.contains(ga) {
                out[*ga as usize] += pair_count[k];
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    pair_chrom, pair_pos_a, pair_pos_b, pair_count,
    cell_offsets, n_cells,
    gene_chrom, gene_start, gene_end,
    resolution, discard_counts,
))]
#[allow(clippy::too_many_arguments)]
fn py_scgad_compute<'py>(
    py: Python<'py>,
    pair_chrom: PyReadonlyArray1<'py, u16>,
    pair_pos_a: PyReadonlyArray1<'py, u32>,
    pair_pos_b: PyReadonlyArray1<'py, u32>,
    pair_count: PyReadonlyArray1<'py, f64>,
    // cell_offsets[i..i+1] gives the slice in the pair arrays for cell i.
    // Length n_cells + 1; cell_offsets[0] = 0; cell_offsets[n_cells] = N_pairs.
    // Caller is responsible for sorting pair arrays by cell_id.
    cell_offsets: PyReadonlyArray1<'py, u32>,
    n_cells: usize,
    gene_chrom: PyReadonlyArray1<'py, u16>,
    gene_start: PyReadonlyArray1<'py, u32>,
    gene_end: PyReadonlyArray1<'py, u32>,
    resolution: u32,
    discard_counts: u32,
) -> Bound<'py, PyArray2<f64>> {
    let pair_chrom = pair_chrom.as_slice().expect("contig pair_chrom");
    let pair_pos_a = pair_pos_a.as_slice().expect("contig pair_pos_a");
    let pair_pos_b = pair_pos_b.as_slice().expect("contig pair_pos_b");
    let pair_count = pair_count.as_slice().expect("contig pair_count");
    let cell_offsets = cell_offsets.as_slice().expect("contig cell_offsets");
    let gene_chrom_s = gene_chrom.as_slice().expect("contig gene_chrom");
    let gene_start_s = gene_start.as_slice().expect("contig gene_start");
    let gene_end_s = gene_end.as_slice().expect("contig gene_end");

    let gene_idx = GeneIndex::build(gene_chrom_s, gene_start_s, gene_end_s, resolution);
    let n_genes = gene_idx.n_genes;

    // Use column-major (Fortran) layout so each column is a contiguous
    // slice in memory — required for the per-cell parallel writers below.
    let mut out = Array2::<f64>::zeros((n_genes, n_cells).f());
    out.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(c, mut col)| {
            let lo = cell_offsets[c] as usize;
            let hi = cell_offsets[c + 1] as usize;
            scgad_one_cell(
                &pair_chrom[lo..hi],
                &pair_pos_a[lo..hi],
                &pair_pos_b[lo..hi],
                &pair_count[lo..hi],
                &gene_idx,
                discard_counts,
                col.as_slice_mut().expect("contig col"),
            );
        });

    let _ = resolution;  // flows into GeneIndex via bin_size
    out.to_pyarray_bound(py)
}

// ============================================================
// bandnorm core.
// ============================================================
//
// Output: same length as input, with `count` replaced by
//   count / band_depth(chrom, diag, cell) * mean_band_depth(chrom, diag)

#[pyfunction]
#[pyo3(signature = (chrom, diag, cell, count))]
fn py_bandnorm_compute<'py>(
    py: Python<'py>,
    chrom: PyReadonlyArray1<'py, u16>,
    diag: PyReadonlyArray1<'py, u32>,
    cell: PyReadonlyArray1<'py, u32>,
    count: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let chrom = chrom.as_slice().expect("contig chrom");
    let diag = diag.as_slice().expect("contig diag");
    let cell = cell.as_slice().expect("contig cell");
    let count = count.as_slice().expect("contig count");
    let n = chrom.len();

    // Pass 1: per (chrom, diag, cell) → band_depth.
    let mut band_depth: AHashMap<(u16, u32, u32), f64> = AHashMap::new();
    for i in 0..n {
        *band_depth.entry((chrom[i], diag[i], cell[i])).or_insert(0.0) += count[i];
    }
    // Pass 2: per (chrom, diag) → mean band_depth across cells.
    // alpha[c,d] = mean over cells with non-empty band of band_depth[c,d,cell].
    let mut by_band: AHashMap<(u16, u32), (f64, u32)> = AHashMap::new(); // (sum, count_of_cells)
    for ((c, d, _cell), &bd) in band_depth.iter() {
        let entry = by_band.entry((*c, *d)).or_insert((0.0, 0));
        entry.0 += bd;
        entry.1 += 1;
    }
    // Pass 3: write normalised counts.
    let mut out = vec![0.0_f64; n];
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let key = (chrom[i], diag[i], cell[i]);
        let bd = band_depth[&key];
        let (sum, cnt) = by_band[&(chrom[i], diag[i])];
        let alpha = sum / (cnt as f64);
        *o = count[i] / bd * alpha;
    });

    PyArray1::from_vec_bound(py, out)
}

// ============================================================
// Threading control.
// ============================================================

#[pyfunction]
fn set_num_threads(n: usize) -> bool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .is_ok()
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_scgad_compute, m)?)?;
    m.add_function(wrap_pyfunction!(py_bandnorm_compute, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    Ok(())
}
