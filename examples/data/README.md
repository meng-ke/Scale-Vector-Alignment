# OMC-1 example data

This directory contains the two small matched FITS cutouts used by
`examples/example.ipynb`:

```text
omc1_image1.fits   H2 column density
omc1_image2.fits   C18O(1-0) integrated intensity
```

The files have the same:

1. 2-D array shape: 144 x 144 pixels,
2. celestial WCS and sky footprint,
3. pixel grid: 5 arcsec per pixel,
4. effective angular resolution: 36 arcsec.

Invalid pixels, when present, should be represented by NaNs; CDD-SVA preserves
the original validity masks when calculating the reported similarities.

## Provenance

`omc1_image1.fits` is a processed OMC-1 cutout derived from the
Herschel/Planck Orion A H2 column-density product described by Lombardi et al.
(2014, A&A, 566, A45).

`omc1_image2.fits` is a processed OMC-1 cutout derived from the C18O(1-0)
integrated-intensity product of the CARMA-NRO Orion Survey described by Kong
et al. (2018, ApJS, 236, 25).

The two products were placed on a common 5 arcsec grid and matched to a common
36 arcsec effective resolution before the cutout was made.

CDD-SVA itself does not perform beam convolution or reprojection. Those steps
remain outside the core package so that the similarity calculation operates on
explicitly matched input fields.

When redistributing derived example data, users should also follow the data-use
and acknowledgement requirements of the original surveys.
