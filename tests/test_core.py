import numpy as np

from cdd_sva.core import _pixel_cosine, _scale_cosine, _shift_cube, _shift_mask


def test_shift_mask_no_wrap():
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 1] = True

    shifted = _shift_mask(mask, dy=1, dx=2)

    assert shifted[2, 3]
    assert shifted.sum() == 1


def test_shift_cube_no_wrap():
    cube = np.zeros((2, 4, 5), dtype=float)
    cube[:, 1, 1] = 3.0

    shifted = _shift_cube(cube, dy=1, dx=2)

    assert np.all(shifted[:, 2, 3] == 3.0)
    assert np.isnan(shifted[:, 0, 0]).all()


def test_identical_local_vectors_have_unit_similarity():
    A = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ]
    )
    mask = np.ones((2, 2), dtype=bool)

    S = _pixel_cosine(A, A.copy(), mask)

    np.testing.assert_allclose(S, 1.0)


def test_scale_cosine_identical_maps():
    A = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[4.0, 3.0], [2.0, 1.0]],
        ]
    )
    mask = np.ones((2, 2), dtype=bool)

    S = _scale_cosine(
        A,
        A.copy(),
        mask,
        rms_fraction=0.0,
    )

    np.testing.assert_allclose(S, 1.0)



def test_signed_local_vectors_preserve_negative_cosine():
    A = np.zeros((2, 1, 1), dtype=float)
    B = np.zeros((2, 1, 1), dtype=float)

    A[:, 0, 0] = [1.0, 0.0]
    B[:, 0, 0] = [-1.0, 0.0]

    mask = np.ones((1, 1), dtype=bool)
    S = _pixel_cosine(A, B, mask)

    np.testing.assert_allclose(S[0, 0], -1.0)
