"""Deterministic microbenchmark for eliminating hand-eye retriangulation."""

import argparse
import time

import numpy as np

from paradex.transforms.triangulate import _triangulate
from src.calibration.allinone.calculate import scale_triangulated_points


def project(points, projections):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = np.einsum("cij,pj->cpi", projections, homogeneous)
    return projected[..., :2] / projected[..., 2:3]


def timed(function, repeats=5):
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - started)
    return min(samples), result


def benchmark(point_count=2000):
    rng = np.random.default_rng(42)
    points = rng.uniform([-0.3, -0.2, 1.5], [0.3, 0.2, 2.5], (point_count, 3))
    camera_centers = np.array(
        [[-0.5, 0.0, 0.0], [-0.2, 0.1, 0.0], [0.2, -0.1, 0.0], [0.5, 0.0, 0.0]]
    )
    projections = np.array(
        [np.column_stack((np.eye(3), -center)) for center in camera_centers]
    )
    observations = project(points, projections)
    scale = 0.37
    scaled_projections = projections.copy()
    scaled_projections[:, :, 3] *= scale

    def legacy_retriangulation():
        return np.vstack(
            [
                _triangulate(observations[:, index], scaled_projections)
                for index in range(point_count)
            ]
        )

    baseline_seconds, baseline_points = timed(legacy_retriangulation, repeats=3)
    optimized_seconds, optimized_points = timed(
        lambda: scale_triangulated_points(points, scale)
    )
    np.testing.assert_allclose(optimized_points, baseline_points, atol=1e-10)
    return baseline_seconds, optimized_seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assert-improvement", action="store_true")
    args = parser.parse_args()
    baseline, optimized = benchmark()
    speedup = baseline / optimized
    print(f"retriangulation baseline: {baseline:.6f}s")
    print(f"scaled-point reuse:       {optimized:.6f}s")
    print(f"speedup:                  {speedup:.1f}x")
    if args.assert_improvement and speedup < 2.0:
        raise AssertionError(f"Expected at least 2x speedup, got {speedup:.2f}x")


if __name__ == "__main__":
    main()
