# rust-bandnorm

Rust port of [BandNorm](https://github.com/sshen82/BandNorm) (Shen, Zheng,
Keleș 2022) — single-cell Hi-C **scGAD** scoring and **band-normalisation**
for cell-type discovery and downstream embedding.

This is **a separate package** that re-implements the algorithm in Rust;
it is not a fork of the upstream R tree. The original R implementation
continues to be available at https://github.com/sshen82/BandNorm.

## What it computes

Two functions, mirroring `BandNorm/R/scGAD.R` and `BandNorm/R/bandnorm.R`:

```python
import bandnorm_rs as bn

# scGAD: gene × cell matrix from contact pairs.
gad = bn.scgad(hic_df, genes, resolution=10_000, depth_norm=True)

# BandNorm: per (chrom, diagonal, cell) depth-normalisation of contact counts.
norm_df = bn.bandnorm(hic_df)
```

Both are bit-equivalent (within f64 ε) to the R reference and faster by
~10-30× depending on the dataset.

## Algorithm parity

`tests/test_parity.py` compares the Rust output against an R reference
generated via pure GenomicRanges (no GenomicInteractions dependency)
on the official BandNorm-shipped `scGADExample.rda` + `mm9Annotations`
data: 100 cells × ~1.3 M contact pairs × 10180 genes.

Tolerance: `max |Rust − R| < 1e-10` on raw counts (integer-exact),
`< 1e-6` on Z-scored GAD output.

## Install

```bash
git clone https://github.com/omicverse/rust-bandnorm
cd rust-bandnorm
maturin develop --release
```

PyPI release coming soon (`pip install bandnorm-rs`).

## Layout

```
rust-bandnorm/
├── pyproject.toml
├── rust/Cargo.toml
├── rust/src/lib.rs              # ~250 LoC: gene index + scGAD + bandnorm
├── python/bandnorm_rs/        # Python wrapper
└── tests/
    ├── reference_scgad.R        # R reference generator
    └── test_parity.py           # pytest parity vs R
```

## License

MIT.
