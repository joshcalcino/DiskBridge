import numpy as np

from diskbridge.chemistry.shielding.healpix_utils import integrate_rays, integrate_rays_with_pathlength


class _DummyCartesianTracer:
    def __init__(self, x_edges: np.ndarray, y_edges: np.ndarray, z_edges: np.ndarray):
        self.x_edges = np.asarray(x_edges, dtype=np.float64)
        self.y_edges = np.asarray(y_edges, dtype=np.float64)
        self.z_edges = np.asarray(z_edges, dtype=np.float64)


def test_self_weight_first_segment_only_cartesian_single_ray() -> None:
    tracer = _DummyCartesianTracer(
        x_edges=np.array([0.0, 1.0, 2.0]),
        y_edges=np.array([0.0, 1.0]),
        z_edges=np.array([0.0, 1.0]),
    )

    density = 2.0
    n_field = np.full((2, 1, 1), density, dtype=np.float64)

    cell_centers = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
    directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    N1 = integrate_rays(tracer, cell_centers, directions, n_field, self_weight=1.0)
    N05 = integrate_rays(tracer, cell_centers, directions, n_field, self_weight=0.5)

    dx = tracer.x_edges[1] - tracer.x_edges[0]
    ds_first = 0.5 * dx

    expected_diff = (1.0 - 0.5) * density * ds_first
    assert np.isclose(float(N1[0, 0] - N05[0, 0]), expected_diff, rtol=0.0, atol=1e-12)


def test_self_weight_does_not_change_pathlength_cartesian_single_ray() -> None:
    tracer = _DummyCartesianTracer(
        x_edges=np.array([0.0, 1.0, 2.0]),
        y_edges=np.array([0.0, 1.0]),
        z_edges=np.array([0.0, 1.0]),
    )

    density = 2.0
    n_field = np.full((2, 1, 1), density, dtype=np.float64)

    cell_centers = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
    directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    N1, S1 = integrate_rays_with_pathlength(tracer, cell_centers, directions, n_field, self_weight=1.0)
    N05, S05 = integrate_rays_with_pathlength(tracer, cell_centers, directions, n_field, self_weight=0.5)

    assert np.isclose(float(S1[0, 0]), float(S05[0, 0]), rtol=0.0, atol=1e-12)

    dx = tracer.x_edges[1] - tracer.x_edges[0]
    ds_first = 0.5 * dx

    expected_diff = (1.0 - 0.5) * density * ds_first
    assert np.isclose(float(N1[0, 0] - N05[0, 0]), expected_diff, rtol=0.0, atol=1e-12)
