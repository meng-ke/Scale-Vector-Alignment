import numpy as np

from cdd_sva import scale_similarity


def test_public_api_with_identical_images():
    n = 32
    y, x = np.indices((n, n))

    image = (
        np.exp(-((x - 10.0) ** 2 + (y - 12.0) ** 2) / (2 * 2.5 ** 2))
        + 0.7 * np.exp(-((x - 22.0) ** 2 + (y - 20.0) ** 2) / (2 * 4.0 ** 2))
    )

    result = scale_similarity(
        image,
        image.copy(),
        n_null=3,
        min_shift=5,
        max_scale=8,
        use_gpu=False,
    )

    finite = np.isfinite(result.S_pix)

    assert finite.any()
    np.testing.assert_allclose(result.S_pix[finite], 1.0, atol=1e-10)
    assert np.isfinite(result.S_crit)
    assert result.scales.ndim == 1
    assert result.S_scale.shape == result.scales.shape
