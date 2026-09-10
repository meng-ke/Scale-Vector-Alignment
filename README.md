# CDD-SVA

**CDD Scale-Vector Alignment (SVA)** is a scale-aware similarity method for
comparing two matched multiscale intensity fields.

SVA uses
[Constrained Diffusion Decomposition (CDD)](https://github.com/gxli/Constrained-Diffusion-Decomposition)
to represent the local distribution of image intensity across spatial scale as
a vector. Morphological similarity is then measured through alignment of the
two scale vectors.

The package returns four quantities:

- `S_pix`: pixel-wise similarity, locating where two images share multiscale structure.
- `S_crit`: empirical reference threshold from spatially shifted null realizations.
- `scales`: CDD spatial scales in pixels.
- `S_scale`: scale-wise similarity, showing at which spatial scales the component maps agree.

## Scope

CDD-SVA is designed primarily for **extended intensity fields with meaningful
multiscale spatial structure**, such as molecular-line, continuum, column-density,
infrared, or optical-emission maps.

Sparse unresolved point-source fields are not the primary use case because most
of their information is carried by isolated sources rather than extended
multiscale structure. Such data can still be studied after conversion to a
continuous intensity or source-density representation when that representation
is physically appropriate.

Before comparison, the two input images should be placed on the **same pixel
grid and sky projection and matched to the same effective angular resolution**.

## Installation

After cloning the repository:

```bash
cd CDD-SVA
pip install .
```

After a PyPI release, installation will be:

```bash
pip install cdd-sva
```

For development and the example notebook:

```bash
pip install -e ".[examples,dev]"
```

CDD-SVA depends on the `constrained-diffusion` package, which is installed
automatically.

### GPU acceleration

CDD supports CUDA and Apple Silicon MPS through PyTorch. Install the appropriate
PyTorch build for your system, then use:

```python
result = scale_similarity(I1, I2, use_gpu=True)
```

A device can be selected explicitly:

```python
result = scale_similarity(I1, I2, use_gpu=True, device="cuda:0")
```

or

```python
result = scale_similarity(I1, I2, use_gpu=True, device="mps")
```

## Quick start

```python
import numpy as np
from cdd_sva import scale_similarity

# I1 and I2 are matched 2-D NumPy arrays.
# NaNs may be used to mark invalid pixels.
result = scale_similarity(I1, I2)

S_pix = result.S_pix
S_crit = result.S_crit
scales = result.scales
S_scale = result.S_scale

print("S_crit =", S_crit)
```

For compatibility with the original analysis-style interface, the first two
outputs can also be unpacked directly:

```python
S_pix, S_crit = scale_similarity(I1, I2)
```

The scale-resolved outputs remain available through the result object:

```python
result = scale_similarity(I1, I2)
print(result.scales)
print(result.S_scale)
```

## Main parameters

The defaults reproduce the analysis choices used in the SVA methods paper.

```python
result = scale_similarity(
    I1,
    I2,

    # empirical null calibration
    n_null=200,
    min_shift=30,
    percentile=95,
    seed=123,

    # CDD scale sampling
    mode="log",
    min_scale=1,
    max_scale=None,
    num_channels=None,
    log_scale_base=2.0,
    linear_scale_step=None,

    # CDD algorithm
    up_sample=True,
    constrained=True,
    inverted=False,

    # hardware
    use_gpu=False,
    device=None,

    # scale-wise quality filter
    scale_rms_fraction=1e-4,
)
```

### Scale sampling

Logarithmic scales are the default:

```python
result = scale_similarity(
    I1, I2,
    mode="log",
    log_scale_base=2.0,
)
```

Finer logarithmic sampling can be requested with a smaller base:

```python
result = scale_similarity(
    I1, I2,
    mode="log",
    log_scale_base=1.5,
)
```

Linear scale sampling is also available:

```python
result = scale_similarity(
    I1, I2,
    mode="lin",
    min_scale=2,
    max_scale=128,
    linear_scale_step=4,
)
```

## Mathematical definition

CDD decomposes the two images into matched scale components,

\[
\mathbf{A}(x,y) = [A_1(x,y), \ldots, A_N(x,y)],
\qquad
\mathbf{B}(x,y) = [B_1(x,y), \ldots, B_N(x,y)].
\]

The pixel-wise similarity is the cosine alignment of the local scale vectors,

\[
S_{\rm pix}(x,y)
=
\frac{
\sum_n A_n(x,y)B_n(x,y)
}{
\sqrt{\sum_n A_n^2(x,y)}
\sqrt{\sum_n B_n^2(x,y)}
}.
\]

At a fixed scale \(l_n\), the component maps are treated as spatial vectors,

\[
S_{\rm scale}(l_n)
=
\frac{
\sum_{x,y}A_n(x,y)B_n(x,y)
}{
\sqrt{\sum_{x,y}A_n^2(x,y)}
\sqrt{\sum_{x,y}B_n^2(x,y)}
}.
\]

For the default constrained decomposition of positive emission fields, both
similarities naturally lie in `[0, 1]`. For signed component fields, CDD-SVA
retains the natural cosine range `[-1, 1]`.

## Null calibration

`S_crit` is obtained from random non-wrapping spatial shifts of the second CDD
cube. The shifted validity mask is moved by the same offset. Valid pixel
similarities from all realizations are pooled, and the requested percentile
(default: 95th) defines the empirical reference threshold.

The pooled null pixels are spatially correlated; `S_crit` is therefore intended
as an empirical reference threshold rather than a formal per-pixel p-value.

## Example notebook

`examples/example.ipynb` provides a worked OMC-1 example using two small,
pre-matched FITS cutouts included in the repository:

```text
examples/data/omc1_image1.fits   # H2 column density
examples/data/omc1_image2.fits   # C18O(1-0) integrated intensity
```

Both images are 144 x 144 pixels, sampled on a 5 arcsec grid and matched to a
36 arcsec effective resolution. The column-density cutout is derived from the
Herschel/Planck Orion A product of Lombardi et al. (2014), and the C18O(1-0)
cutout is derived from the CARMA-NRO Orion Survey product of Kong et al. (2018).

The notebook can be launched either from the repository root or from the
`examples/` directory; it resolves the example-data path automatically.

See `examples/data/README.md` for data provenance and preparation details.

## Testing

```bash
pytest
```

To verify that the package builds:

```bash
python -m build
twine check dist/*
```

## Citation

If you use CDD-SVA, please cite both the Scale-Vector Alignment methods paper
and the original CDD paper:

- Zhao, M., Li, G.-X., Qiu, K., & Li, S., *Scale-Vector Alignment: A CDD-based
  Framework for Spatially Resolved Morphological Similarity in Astronomical
  Images*.
- Li, G.-X. 2022, ApJS, 259, 59, doi:10.3847/1538-4365/ac4bc4.

See `CITATION.cff` for software citation metadata.

## License

GNU General Public License v3.0. See `LICENSE`.
