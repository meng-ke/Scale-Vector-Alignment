import numpy as np

from cdd_sva import SVAResult


def test_result_unpacking():
    S = np.ones((3, 3))
    result = SVAResult(
        S_pix=S,
        S_crit=0.9,
        scales=np.array([1.0, 2.0]),
        S_scale=np.array([0.8, 0.9]),
    )

    S_pix, S_crit = result

    assert S_pix is S
    assert S_crit == 0.9
    np.testing.assert_array_equal(result.scales, [1.0, 2.0])
