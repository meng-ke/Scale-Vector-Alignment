"""Core implementation of CDD Scale-Vector Alignment (SVA)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class SVAResult:
    """Result returned by :func:`scale_similarity`.

    Parameters
    ----------
    S_pix
        Pixel-wise scale-vector similarity map.
    S_crit
        Empirical threshold from the spatial-shift null distribution.
    scales
        CDD spatial scales in pixels.
    S_scale
        Scale-wise similarity evaluated at ``scales``.

    Notes
    -----
    For convenience, the result can also be unpacked as ``S_pix, S_crit``::

        result = scale_similarity(I1, I2)
        S_pix, S_crit = result

    The full scale-resolved outputs remain available as
    ``result.scales`` and ``result.S_scale``.
    """

    S_pix: np.ndarray
    S_crit: float
    scales: np.ndarray
    S_scale: np.ndarray

    def __iter__(self) -> Iterator[object]:
        """Allow backward-compatible ``S_pix, S_crit = result`` unpacking."""
        yield self.S_pix
        yield self.S_crit


def _get_cdd():
    try:
        import constrained_diffusion as cdd
    except ImportError as exc:
        raise ImportError(
            "CDD-SVA requires the 'constrained-diffusion' package. "
            "Install CDD-SVA with `pip install .` or install the dependency "
            "directly with `pip install constrained-diffusion`."
        ) from exc
    return cdd


def _validate_images(I1, I2):
    I1 = np.asarray(I1)
    I2 = np.asarray(I2)

    if I1.ndim != 2 or I2.ndim != 2:
        raise ValueError("scale_similarity expects two 2-D images.")

    if I1.shape != I2.shape:
        raise ValueError(
            f"Input images must have the same shape; got {I1.shape} and {I2.shape}."
        )

    return I1, I2


def _decompose_pair(
    I1,
    I2,
    *,
    use_gpu=False,
    mode="log",
    num_channels=None,
    max_scale=None,
    min_scale=1,
    log_scale_base=2.0,
    linear_scale_step=None,
    up_sample=True,
    constrained=True,
    inverted=False,
    device=None,
):
    """CDD decomposition of two images with identical scale settings."""

    if mode not in {"log", "lin"}:
        raise ValueError("mode must be 'log' or 'lin'.")

    cdd = _get_cdd()

    dtype = np.float32 if use_gpu else np.float64

    # CDD requires finite numerical arrays. The original validity masks are
    # preserved separately by scale_similarity and applied to all statistics.
    X1 = np.nan_to_num(
        I1, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(dtype, copy=False)
    X2 = np.nan_to_num(
        I2, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(dtype, copy=False)

    kwargs = dict(
    num_channels=num_channels,
    max_scale=max_scale,
    min_scale=min_scale,
    mode=mode,
    log_scale_base=log_scale_base,
    linear_scale_step=linear_scale_step,
    up_sample=up_sample,
    constrained=constrained,
    inverted=inverted,
    use_gpu=use_gpu,
    device=device,
    return_scales=True,
    )

    A, _, scales_A = cdd.constrained_diffusion_decomposition(X1, **kwargs)
    B, _, scales_B = cdd.constrained_diffusion_decomposition(X2, **kwargs)

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    scales_A = np.asarray(scales_A, dtype=np.float64)
    scales_B = np.asarray(scales_B, dtype=np.float64)

    if scales_A.shape != scales_B.shape or not np.allclose(scales_A, scales_B):
        raise ValueError("CDD scales do not match between the two images.")

    if A.shape != B.shape:
        raise ValueError("CDD component cubes do not have matching shapes.")

    if A.ndim != 3 or A.shape[1:] != I1.shape:
        raise ValueError(
            "Unexpected CDD output shape; expected (nscale, ny, nx)."
        )

    return A, B, scales_A


def _shift_cube(B, dy, dx):
    """Shift a scale cube without wrap-around."""

    _, ny, nx = B.shape
    out = np.full(B.shape, np.nan, dtype=np.float64)

    ys = slice(max(0, -dy), min(ny, ny - dy))
    yd = slice(max(0,  dy), min(ny, ny + dy))
    xs = slice(max(0, -dx), min(nx, nx - dx))
    xd = slice(max(0,  dx), min(nx, nx + dx))

    out[:, yd, xd] = B[:, ys, xs]
    return out


def _shift_mask(mask, dy, dx):
    """Shift a 2-D validity mask without wrap-around."""

    ny, nx = mask.shape
    out = np.zeros(mask.shape, dtype=bool)

    ys = slice(max(0, -dy), min(ny, ny - dy))
    yd = slice(max(0,  dy), min(ny, ny + dy))
    xs = slice(max(0, -dx), min(nx, nx - dx))
    xd = slice(max(0,  dx), min(nx, nx + dx))

    out[yd, xd] = mask[ys, xs]
    return out


def _pixel_cosine(A, B, mask):
    """Stable cosine similarity between local scale vectors."""

    sa = np.max(np.abs(A), axis=0)
    sb = np.max(np.where(np.isfinite(B), np.abs(B), 0.0), axis=0)

    good = (
        mask
        & np.isfinite(sa)
        & np.isfinite(sb)
        & (sa > 0)
        & (sb > 0)
    )

    As = np.zeros_like(A, dtype=np.float64)
    Bs = np.zeros_like(B, dtype=np.float64)

    np.divide(A, sa[None, :, :], out=As, where=sa[None, :, :] > 0)
    np.divide(B, sb[None, :, :], out=Bs, where=sb[None, :, :] > 0)

    num = np.nansum(As * Bs, axis=0)
    den = np.sqrt(
        np.nansum(As * As, axis=0)
        * np.nansum(Bs * Bs, axis=0)
    )

    good &= np.isfinite(den) & (den > 0)

    S = np.full(mask.shape, np.nan, dtype=np.float64)
    S[good] = num[good] / den[good]

    # Retain the natural cosine range. For positive emission fields with
    # constrained CDD, the result naturally lies in [0, 1].
    S[good] = np.clip(S[good], -1.0, 1.0)

    return S


def _scale_cosine(A, B, mask, *, rms_fraction=1e-4):
    """Cosine similarity between the two CDD component maps at each scale."""

    if rms_fraction < 0:
        raise ValueError("scale_rms_fraction must be >= 0.")

    nscale = A.shape[0]
    S_scale = np.full(nscale, np.nan, dtype=np.float64)
    rms_A = np.full(nscale, np.nan, dtype=np.float64)
    rms_B = np.full(nscale, np.nan, dtype=np.float64)

    if not np.any(mask):
        return S_scale

    for n in range(nscale):
        a = A[n][mask].astype(np.float64, copy=False)
        b = B[n][mask].astype(np.float64, copy=False)

        good = np.isfinite(a) & np.isfinite(b)
        a = a[good]
        b = b[good]

        if a.size == 0:
            continue

        rms_A[n] = np.sqrt(np.mean(a * a))
        rms_B[n] = np.sqrt(np.mean(b * b))

    if not np.any(np.isfinite(rms_A)) or not np.any(np.isfinite(rms_B)):
        return S_scale

    max_rms_A = np.nanmax(rms_A)
    max_rms_B = np.nanmax(rms_B)

    keep = (
        np.isfinite(rms_A)
        & np.isfinite(rms_B)
        & (rms_A > rms_fraction * max_rms_A)
        & (rms_B > rms_fraction * max_rms_B)
    )

    for n in np.flatnonzero(keep):
        a = A[n][mask].astype(np.float64, copy=False)
        b = B[n][mask].astype(np.float64, copy=False)

        good = np.isfinite(a) & np.isfinite(b)
        a = a[good]
        b = b[good]

        if a.size == 0:
            continue

        sa = np.max(np.abs(a))
        sb = np.max(np.abs(b))

        if not np.isfinite(sa) or not np.isfinite(sb) or sa <= 0 or sb <= 0:
            continue

        a = a / sa
        b = b / sb

        den = np.sqrt(np.sum(a * a) * np.sum(b * b))
        if not np.isfinite(den) or den <= 0:
            continue

        value = np.sum(a * b) / den

        S_scale[n] = np.clip(value, -1.0, 1.0)

    return S_scale


def scale_similarity(
    I1,
    I2,
    *,
    # Null calibration
    n_null=200,
    min_shift=30,
    percentile=95,
    seed=123,
    # Scale-wise quality filter
    scale_rms_fraction=1e-4,
    # CDD scale controls
    mode="log",
    num_channels=None,
    max_scale=None,
    min_scale=1,
    log_scale_base=2.0,
    linear_scale_step=None,
    # CDD algorithm controls
    up_sample=True,
    constrained=True,
    inverted=False,
    # Hardware / memory controls
    use_gpu=False,
    device=None,
):
    """Compute CDD Scale-Vector Alignment (SVA) between two images.

    Parameters
    ----------
    I1, I2 : array_like, shape (ny, nx)
        Two 2-D intensity fields on the same pixel grid and at the same
        effective angular resolution. NaN/inf values are treated as invalid
        pixels. SVA is intended primarily for extended images with meaningful
        multiscale spatial structure rather than sparse unresolved point-source
        fields.

    n_null : int, default=200
        Number of random non-wrapping spatial shifts used for the empirical
        null distribution.
    min_shift : float, default=30
        Minimum null-shift displacement in pixels.
    percentile : float, default=95
        Percentile of the pooled null distribution used as ``S_crit``.
    seed : int or None, default=123
        Random seed for null shifts.

    scale_rms_fraction : float, default=1e-4
        A scale contributes to ``S_scale`` only when the RMS component
        amplitude exceeds this fraction of the maximum RMS in both images.

    mode : {"log", "lin"}, default="log"
        CDD scale spacing.
    num_channels : int or None, default=None
        Number of CDD scale channels.
    max_scale : float or None, default=None
        Largest CDD scale in pixels.
    min_scale : float, default=1
        Smallest CDD scale in pixels.
    log_scale_base : float, default=2.0
        Base of logarithmic scale spacing. Smaller values sample scale more
        finely.
    linear_scale_step : float or None, default=None
        Fixed scale step when ``mode="lin"``.

    up_sample : bool, default=True
        Use CDD's hybrid upsampling treatment of the smallest scales.
    constrained : bool, default=True
        Use constrained diffusion. This is the SVA paper default.
    inverted : bool, default=False
        Decompose depressions/holes rather than positive peaks.

    use_gpu : bool, default=False
        Use CDD's GPU backend.
    device : str or None, default=None
        Accelerator device, e.g. ``"cuda"``, ``"cuda:0"``, or ``"mps"``.
    Returns
    -------
    SVAResult
        ``result.S_pix`` : pixel-wise similarity map.

        ``result.S_crit`` : empirical null threshold.

        ``result.scales`` : CDD scales in pixels.

        ``result.S_scale`` : scale-wise similarity.

        For convenience, ``S_pix, S_crit = result`` is also supported.

    Notes
    -----
    The defaults reproduce the analysis choices in the SVA methods paper:
    constrained logarithmic CDD, hybrid upsampling, 200 spatial shifts,
    30-pixel minimum displacement, 95th-percentile threshold, seed 123, and
    a scale RMS cut of 1e-4.

    The null distribution is used as an empirical reference distribution,
    not as a set of independent samples for formal per-pixel p-values.
    """

    I1, I2 = _validate_images(I1, I2)

    if not isinstance(n_null, (int, np.integer)) or n_null <= 0:
        raise ValueError("n_null must be a positive integer.")
    if min_shift < 0:
        raise ValueError("min_shift must be >= 0.")
    if not 0 < percentile < 100:
        raise ValueError("percentile must lie strictly between 0 and 100.")
    if scale_rms_fraction < 0:
        raise ValueError("scale_rms_fraction must be >= 0.")

    # Preserve original validity before finite filling for CDD.
    valid1 = np.isfinite(I1)
    valid2 = np.isfinite(I2)
    valid = valid1 & valid2

    if not np.any(valid):
        raise ValueError("The two images have no overlapping valid pixels.")

    A, B, scales = _decompose_pair(
        I1,
        I2,
        use_gpu=use_gpu,
        mode=mode,
        num_channels=num_channels,
        max_scale=max_scale,
        min_scale=min_scale,
        log_scale_base=log_scale_base,
        linear_scale_step=linear_scale_step,
        up_sample=up_sample,
        constrained=constrained,
        inverted=inverted,
        device=device,
    )

    S_pix = _pixel_cosine(A, B, valid)

    S_scale = _scale_cosine(
        A,
        B,
        valid,
        rms_fraction=scale_rms_fraction,
    )

    rng = np.random.default_rng(seed)
    _, ny, nx = B.shape

    max_shift = np.hypot(nx // 2, ny // 2)
    if min_shift > max_shift:
        raise ValueError(
            f"min_shift={min_shift} is too large for image shape {I1.shape}; "
            f"the largest allowed half-image displacement is "
            f"{max_shift:.2f} pixels."
        )

    null_values = []

    for _ in range(n_null):
        while True:
            dx = int(rng.integers(-nx // 2, nx // 2 + 1))
            dy = int(rng.integers(-ny // 2, ny // 2 + 1))

            if np.hypot(dx, dy) >= min_shift:
                break

        B0 = _shift_cube(B, dy, dx)
        valid2_0 = _shift_mask(valid2, dy, dx)

        # Keep the original common analysis footprint and additionally require
        # the shifted B pixel to originate from a valid position in I2.
        null_mask = valid & valid2_0

        S0 = _pixel_cosine(
            A,
            B0,
            null_mask,
        )

        values = S0[np.isfinite(S0)]
        if values.size:
            null_values.append(values)

    if not null_values:
        raise RuntimeError("No valid samples were produced by the null shifts.")

    S_null = np.concatenate(null_values)
    S_crit = float(np.nanpercentile(S_null, percentile))

    return SVAResult(
        S_pix=S_pix,
        S_crit=S_crit,
        scales=scales,
        S_scale=S_scale,
    )
