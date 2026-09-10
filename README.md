# Scale-Vector Alignment (SVA)

**Scale-Vector Alignment (SVA)** is a CDD-based, scale-aware method for measuring morphological similarity between multiscale astronomical intensity fields.

SVA uses **Constrained Diffusion Decomposition (CDD)** to represent, at each image position, how intensity is distributed across spatial scale. The local CDD amplitudes form a **scale vector**, and morphological similarity is measured through the alignment of scale vectors.

The package provides:

- `S_pix`: pixel-wise similarity, showing **where** two images share multiscale structure.
- `S_crit`: an empirical reference threshold from spatially shifted null realizations.
- `S_scale`: scale-wise similarity, showing **at which spatial scales** the two images are most similar.
- `scales`: the CDD spatial scales corresponding to `S_scale`.

The method is designed primarily for **extended intensity fields with meaningful multiscale spatial structure**, such as molecular-line, dust-continuum, column-density, infrared, and optical-emission maps. Sparse unresolved point-source fields are not the primary use case unless they are first represented as a physically meaningful continuous intensity or source-density field.

---

## Installation

### Install directly from GitHub

```bash
pip install git+https://github.com/meng-ke/Scale-Vector-Alignment.git
```

### Clone and install locally

```bash
git clone https://github.com/meng-ke/Scale-Vector-Alignment.git
cd Scale-Vector-Alignment
pip install .
```

### Development installation

For development, testing, and the example notebook:

```bash
git clone https://github.com/meng-ke/Scale-Vector-Alignment.git
cd Scale-Vector-Alignment
pip install -e ".[examples,dev]"
```

Once the package is released on PyPI, installation will simply be:

```bash
pip install cdd-sva
```

CDD-SVA depends on the `constrained-diffusion` package, which is installed automatically.

---

## Quick start

```python
from astropy.io import fits
from cdd_sva import scale_similarity

I1 = fits.getdata("image1.fits").squeeze()
I2 = fits.getdata("image2.fits").squeeze()

result = scale_similarity(
    I1,
    I2,
    use_gpu=False,
    mode="log",
)

print("S_crit =", result.S_crit)
print("CDD scales =", result.scales)

S_pix = result.S_pix
S_scale = result.S_scale
```

For compatibility with the original analysis-style interface, the first two outputs can also be unpacked directly:

```python
S_pix, S_crit = scale_similarity(I1, I2)
```

The scale-resolved outputs remain available through the result object:

```python
result.scales
result.S_scale
```

---

## Input requirements

The two input images should be prepared before running SVA so that they have the same:

1. 2-D array shape,
2. celestial projection / WCS,
3. pixel grid,
4. effective angular resolution.

Invalid pixels may be represented by `NaN`. Their original validity masks are preserved through the similarity calculation.

CDD-SVA does **not** perform beam matching or reprojection internally. These operations are intentionally left outside the core package so that the inputs to the similarity calculation are explicit and reproducible.

---

## Main parameters

```python
result = scale_similarity(
    I1,
    I2,

    # empirical null calibration
    n_null=200,
    min_shift=30,
    percentile=95,
    seed=123,

    # scale-wise quality filter
    scale_rms_fraction=1e-4,

    # CDD scale sampling
    mode="log",
    num_channels=None,
    max_scale=None,
    min_scale=1,
    log_scale_base=2.0,
    linear_scale_step=None,

    # CDD algorithm
    up_sample=True,
    constrained=True,
    inverted=False,

    # hardware
    use_gpu=False,
    device=None,
)
```

The defaults reproduce the analysis choices used in the SVA methods paper.

### Scale sampling

Logarithmic scale sampling is the default:

```python
result = scale_similarity(
    I1,
    I2,
    mode="log",
    log_scale_base=2.0,
)
```

A smaller logarithmic base gives finer scale sampling:

```python
result = scale_similarity(
    I1,
    I2,
    mode="log",
    log_scale_base=1.5,
)
```

Linear scale sampling is also available:

```python
result = scale_similarity(
    I1,
    I2,
    mode="lin",
    min_scale=2,
    max_scale=128,
    linear_scale_step=4,
)
```

---

## GPU acceleration

CDD supports GPU acceleration through PyTorch.

For CUDA:

```python
result = scale_similarity(
    I1,
    I2,
    use_gpu=True,
    device="cuda",
)
```

A specific CUDA device can be selected with, for example:

```python
result = scale_similarity(
    I1,
    I2,
    use_gpu=True,
    device="cuda:0",
)
```

On supported Apple Silicon systems:

```python
result = scale_similarity(
    I1,
    I2,
    use_gpu=True,
    device="mps",
)
```

Install a PyTorch build appropriate for your hardware before enabling GPU acceleration.

---

## Method

CDD decomposes each image into matched spatial-scale components,

\[
\mathbf{A}(x,y) =
[A_1(x,y), A_2(x,y), \ldots, A_N(x,y)],
\]

\[
\mathbf{B}(x,y) =
[B_1(x,y), B_2(x,y), \ldots, B_N(x,y)].
\]

At each pixel, the component amplitudes define a local scale vector. The pixel-wise similarity is

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

Thus, `S_pix` compares how the two images distribute their local intensity across scale rather than comparing their absolute normalization.

At a fixed CDD scale \(l_n\), the two component maps are treated as spatial vectors,

\[
S_{\rm scale}(l_n)
=
\frac{
\sum_{x,y} A_n(x,y)B_n(x,y)
}{
\sqrt{\sum_{x,y} A_n^2(x,y)}
\sqrt{\sum_{x,y} B_n^2(x,y)}
}.
\]

For positive emission fields with constrained CDD, the similarities naturally lie in `[0, 1]`. For signed component fields, CDD-SVA retains the natural cosine range `[-1, 1]`.

---

## Null calibration

High local similarity can occur by chance in structured images, so SVA estimates an empirical reference distribution using random non-wrapping spatial shifts of the second CDD cube.

The validity mask of the second image is shifted by the same offset. Valid pixel similarities from all shifted realizations are pooled, and the requested percentile of that distribution defines `S_crit`, with the default being the 95th percentile.

The pooled null pixels are spatially correlated. `S_crit` is therefore used as an **empirical reference threshold**, not as a formal per-pixel p-value.

---

## OMC-1 example

A worked example is provided in:

```text
examples/example.ipynb
```

The repository includes two small matched OMC-1 FITS cutouts:

```text
examples/data/omc1_image1.fits   H2 column density
examples/data/omc1_image2.fits   C18O(1-0) integrated intensity
```

Both images are sampled on a **5 arcsec grid** and matched to a **36 arcsec effective resolution**.

The H2 column-density cutout is derived from the Herschel/Planck Orion A product described by **Lombardi et al. (2014, A&A, 566, A45)**. The C18O(1-0) cutout is derived from the **CARMA-NRO Orion Survey** product described by **Kong et al. (2018, ApJS, 236, 25)**.

To run the example:

```bash
jupyter notebook examples/example.ipynb
```

The notebook resolves the example-data path whether it is launched from the repository root or from the `examples/` directory.

---

## Testing

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

Then run:

```bash
pytest
```

The repository also includes a GitHub Actions workflow that runs the test suite automatically on supported Python versions.

To verify that the package builds locally:

```bash
python -m build
twine check dist/*
```

---

## Citation

If you use Scale-Vector Alignment in scientific work, please cite both the SVA methods paper and the original CDD paper.

**Scale-Vector Alignment methods paper**

Mengke Zhao, Guang-Xing Li, Keping Qiu, and Shanghuo Li,  
*Scale-Vector Alignment: A CDD-based Framework for Spatially Resolved Morphological Similarity in Astronomical Images.*

**CDD**

Li, G.-X. 2022, *The Astrophysical Journal Supplement Series*, 259, 59.  
DOI: `10.3847/1538-4365/ac4bc4`

Software citation metadata are also provided in `CITATION.cff`.

---

## License

This project is released under the **GNU General Public License v3.0 (GPL-3.0)**. See `LICENSE` for details.
