"""Focused checks for Stage-B image-motion gating and patch change scoring."""

import argparse
import pathlib
import sys
import unittest

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import segment_parts as sp  # noqa: E402


class StageBImageMotionTests(unittest.TestCase):
    def test_group_object_gate_rejects_detached_change(self):
        raw_motion = np.zeros((20, 20), dtype=bool)
        raw_motion[8:12, 8:12] = True
        raw_motion[0:3, 0:3] = True
        group_mask = np.zeros_like(raw_motion)
        group_mask[7:13, 7:13] = True

        accepted, rejected, support = sp._gate_stage_b_image_motion(raw_motion, [group_mask], 1)

        self.assertTrue(np.all(accepted[8:12, 8:12]))
        self.assertFalse(np.any(accepted[0:3, 0:3]))
        self.assertTrue(np.all(rejected[0:3, 0:3]))
        self.assertTrue(np.any(support))

    def test_raster_change_precision_rewards_matching_footprint(self):
        vertices = np.array(
            [
                [-0.20, -0.10, 1.0],
                [0.20, -0.10, 1.0],
                [0.20, 0.10, 1.0],
                [-0.20, 0.10, 1.0],
            ],
            dtype=np.float64,
        )
        # Winding produces a -Z normal, facing a camera at the world origin.
        faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
        projection = np.array(
            [[100.0, 0.0, 50.0, 0.0], [0.0, 100.0, 50.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        camera = {"projection": projection, "cam_from_world": np.eye(4, dtype=np.float64)}
        moved_vertices = vertices.copy()
        moved_vertices[:, 0] += 0.18
        patch_faces = np.arange(faces.shape[0], dtype=np.int64)
        rest = sp._rasterize_patch_mask(vertices, faces, patch_faces, camera, (100, 100), 0.0)
        moved = sp._rasterize_patch_mask(moved_vertices, faces, patch_faces, camera, (100, 100), 0.0)
        evidence = np.logical_xor(rest, moved)

        precision, cameras, predicted_pixels, overlap_pixels = sp._state_patch_change_precision(
            vertices,
            moved_vertices,
            faces,
            patch_faces,
            {"cam": camera},
            {"cam": evidence},
            0.0,
            0,
            1,
        )

        self.assertEqual(cameras, 1)
        self.assertGreater(predicted_pixels, 0)
        self.assertEqual(predicted_pixels, overlap_pixels)
        self.assertGreater(precision, 0.99)

    def test_pooled_precision_rejects_diffuse_edge_overlap(self):
        self.assertLess(sp._pooled_change_precision(10_000, 590), 0.12)
        self.assertGreater(sp._pooled_change_precision(1_000, 620), 0.12)

    def test_depth_composite_does_not_assign_hidden_rear_patch_motion(self):
        vertices = np.array(
            [
                [-0.50, -0.50, 1.0], [0.50, -0.50, 1.0], [0.50, 0.50, 1.0], [-0.50, 0.50, 1.0],
                [-0.16, -0.12, 2.0], [0.16, -0.12, 2.0], [0.16, 0.12, 2.0], [-0.16, 0.12, 2.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 2, 1], [0, 3, 2], [4, 6, 5], [4, 7, 6]], dtype=np.int64)
        camera = {
            "projection": np.array(
                [[80.0, 0.0, 50.0, 0.0], [0.0, 80.0, 50.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            "cam_from_world": np.eye(4, dtype=np.float64),
        }
        patch_faces = np.array([2, 3], dtype=np.int64)
        patch_lookup = np.array([False, False, True, True])
        full_layers = sp._rasterize_mesh_depth_layers(vertices, faces, np.arange(4), camera, (100, 100))
        rest_layers = sp._rasterize_mesh_depth_layers(vertices, faces, patch_faces, camera, (100, 100))
        moved_vertices = vertices.copy()
        moved_vertices[4:, 0] += 0.15
        moved_layers = sp._rasterize_mesh_depth_layers(moved_vertices, faces, patch_faces, camera, (100, 100))

        predicted, rest_visible, moved_visible = sp._composite_patch_owner_change(
            full_layers, rest_layers, moved_layers, patch_lookup
        )

        self.assertFalse(np.any(rest_visible))
        self.assertFalse(np.any(moved_visible))
        self.assertFalse(np.any(predicted))

    def test_depth_composite_keeps_visible_front_patch_motion(self):
        vertices = np.array(
            [
                [-0.50, -0.50, 2.0], [0.50, -0.50, 2.0], [0.50, 0.50, 2.0], [-0.50, 0.50, 2.0],
                [-0.16, -0.12, 1.0], [0.16, -0.12, 1.0], [0.16, 0.12, 1.0], [-0.16, 0.12, 1.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 2, 1], [0, 3, 2], [4, 6, 5], [4, 7, 6]], dtype=np.int64)
        camera = {
            "projection": np.array(
                [[80.0, 0.0, 50.0, 0.0], [0.0, 80.0, 50.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            "cam_from_world": np.eye(4, dtype=np.float64),
        }
        patch_faces = np.array([2, 3], dtype=np.int64)
        patch_lookup = np.array([False, False, True, True])
        full_layers = sp._rasterize_mesh_depth_layers(vertices, faces, np.arange(4), camera, (100, 100))
        rest_layers = sp._rasterize_mesh_depth_layers(vertices, faces, patch_faces, camera, (100, 100))
        moved_vertices = vertices.copy()
        moved_vertices[4:, 0] += 0.22
        moved_layers = sp._rasterize_mesh_depth_layers(moved_vertices, faces, patch_faces, camera, (100, 100))

        predicted, rest_visible, moved_visible = sp._composite_patch_owner_change(
            full_layers, rest_layers, moved_layers, patch_lookup
        )

        self.assertTrue(np.any(rest_visible))
        self.assertTrue(np.any(moved_visible))
        self.assertTrue(np.any(predicted))

    def test_joint_predicted_matches_composite_owner_change(self):
        # The sparse joint composition must reproduce _composite_patch_owner_change
        # for a single patch (same scene as the visible-front-patch test).
        vertices = np.array(
            [
                [-0.50, -0.50, 2.0], [0.50, -0.50, 2.0], [0.50, 0.50, 2.0], [-0.50, 0.50, 2.0],
                [-0.16, -0.12, 1.0], [0.16, -0.12, 1.0], [0.16, 0.12, 1.0], [-0.16, 0.12, 1.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 2, 1], [0, 3, 2], [4, 6, 5], [4, 7, 6]], dtype=np.int64)
        camera = {
            "projection": np.array(
                [[80.0, 0.0, 50.0, 0.0], [0.0, 80.0, 50.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            "cam_from_world": np.eye(4, dtype=np.float64),
        }
        patch_faces = np.array([2, 3], dtype=np.int64)
        patch_lookup = np.array([False, False, True, True])
        moved_vertices = vertices.copy()
        moved_vertices[4:, 0] += 0.22

        full_layers = sp._rasterize_mesh_depth_layers(vertices, faces, np.arange(4), camera, (100, 100))
        rest_layers = sp._rasterize_mesh_depth_layers(vertices, faces, patch_faces, camera, (100, 100))
        moved_layers = sp._rasterize_mesh_depth_layers(moved_vertices, faces, patch_faces, camera, (100, 100))
        predicted_compose, _rest_visible, _moved_visible = sp._composite_patch_owner_change(
            full_layers, rest_layers, moved_layers, patch_lookup
        )

        view = {
            "static_base": sp._joint_static_base(full_layers, patch_lookup),
            "evidence": np.zeros(100 * 100, dtype=bool),
            "rest": {0: sp._sparse_top_depth(vertices, faces, patch_faces, camera, (100, 100))},
            "moved": {0: sp._sparse_top_depth(moved_vertices, faces, patch_faces, camera, (100, 100))},
        }
        predicted_joint = sp._joint_predicted_change(view, [0]).reshape(100, 100)

        self.assertTrue(np.any(predicted_joint))
        self.assertTrue(np.array_equal(predicted_joint, predicted_compose))

    @staticmethod
    def _disocclusion_scene():
        # Frame body at z=2 (never a candidate), handle at z=1 (candidate 0, truly
        # moves), rear surface at z=1.5 fully hidden behind the handle at rest
        # (candidate 1, truly static but revealed by the handle's motion).
        vertices = np.array(
            [
                [-0.50, -0.50, 2.0], [0.50, -0.50, 2.0], [0.50, 0.50, 2.0], [-0.50, 0.50, 2.0],
                [-0.16, -0.12, 1.0], [0.16, -0.12, 1.0], [0.16, 0.12, 1.0], [-0.16, 0.12, 1.0],
                [-0.16, -0.12, 1.5], [0.16, -0.12, 1.5], [0.16, 0.12, 1.5], [-0.16, 0.12, 1.5],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [[0, 2, 1], [0, 3, 2], [4, 6, 5], [4, 7, 6], [8, 10, 9], [8, 11, 10]],
            dtype=np.int64,
        )
        camera = {
            "projection": np.array(
                [[80.0, 0.0, 50.0, 0.0], [0.0, 80.0, 50.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            "cam_from_world": np.eye(4, dtype=np.float64),
        }
        moved_vertices = vertices.copy()
        moved_vertices[4:, 0] += 0.30  # the fitted part transform moves every candidate vertex
        candidate_lookup = np.array([False, False, True, True, True, True])
        handle_faces = np.array([2, 3], dtype=np.int64)
        rear_faces = np.array([4, 5], dtype=np.int64)

        def footprint(world, face_idx):
            layers = sp._rasterize_mesh_depth_layers(world, faces, face_idx, camera, (100, 100))
            return np.isfinite(layers[0])

        # What really happened: the handle moved, the rear surface stayed.
        evidence = np.logical_xor(footprint(vertices, handle_faces), footprint(moved_vertices, handle_faces))
        full_layers = sp._rasterize_mesh_depth_layers(vertices, faces, np.arange(6), camera, (100, 100))
        view = {
            "state_id": "s0",
            "serial": "cam",
            "group": "g0",
            "shape": (100, 100),
            "static_base": sp._joint_static_base(full_layers, candidate_lookup),
            "evidence": evidence.reshape(-1),
            "rest": {
                0: sp._sparse_top_depth(vertices, faces, handle_faces, camera, (100, 100)),
                1: sp._sparse_top_depth(vertices, faces, rear_faces, camera, (100, 100)),
            },
            "moved": {
                0: sp._sparse_top_depth(moved_vertices, faces, handle_faces, camera, (100, 100)),
                1: sp._sparse_top_depth(moved_vertices, faces, rear_faces, camera, (100, 100)),
            },
        }
        return view

    def test_joint_selection_rejects_disoccluded_static_patch(self):
        import argparse

        view = self._disocclusion_scene()
        stats_args = argparse.Namespace(
            stage_b_patch_min_observed_cameras=1, stage_b_patch_min_change_precision=0.05
        )
        stats = sp._joint_patch_stats([view], [0, 1], 1, stats_args)
        rear_pooled = sp._pooled_change_precision(
            stats[1]["predicted_change_pixels"], stats[1]["overlap_change_pixels"]
        )
        # The independent test cannot reject the revealed rear surface: hypothesized
        # to move with the handle's transform, it lands inside the handle's real
        # change region with near-perfect precision.
        self.assertEqual(stats[1]["state_count"], 1)
        self.assertGreater(rear_pooled, 0.9)

        selected, trace, totals = sp._joint_select_patches([view], [0, 1], 1, 0.25, 1, 0.05, 6)
        self.assertEqual(selected, [0])
        self.assertEqual(int(totals["spurious_pixels"]), 0)
        self.assertGreater(int(totals["explained_pixels"]), 0)
        self.assertFalse(trace[-1]["accepted"])

    def test_joint_selection_keeps_two_genuine_movers(self):
        vertices = np.array(
            [
                [-0.60, -0.60, 2.0], [0.60, -0.60, 2.0], [0.60, 0.60, 2.0], [-0.60, 0.60, 2.0],
                [-0.40, -0.12, 1.0], [-0.10, -0.12, 1.0], [-0.10, 0.12, 1.0], [-0.40, 0.12, 1.0],
                [0.10, -0.12, 1.0], [0.40, -0.12, 1.0], [0.40, 0.12, 1.0], [0.10, 0.12, 1.0],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [[0, 2, 1], [0, 3, 2], [4, 6, 5], [4, 7, 6], [8, 10, 9], [8, 11, 10]],
            dtype=np.int64,
        )
        camera = {
            "projection": np.array(
                [[80.0, 0.0, 50.0, 0.0], [0.0, 80.0, 50.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            "cam_from_world": np.eye(4, dtype=np.float64),
        }
        moved_vertices = vertices.copy()
        moved_vertices[4:, 0] += 0.10
        candidate_lookup = np.array([False, False, True, True, True, True])
        faces_a = np.array([2, 3], dtype=np.int64)
        faces_b = np.array([4, 5], dtype=np.int64)

        def footprint(world, face_idx):
            layers = sp._rasterize_mesh_depth_layers(world, faces, face_idx, camera, (100, 100))
            return np.isfinite(layers[0])

        evidence = np.logical_xor(footprint(vertices, faces_a), footprint(moved_vertices, faces_a))
        evidence |= np.logical_xor(footprint(vertices, faces_b), footprint(moved_vertices, faces_b))
        full_layers = sp._rasterize_mesh_depth_layers(vertices, faces, np.arange(6), camera, (100, 100))
        view = {
            "state_id": "s0",
            "serial": "cam",
            "group": "g0",
            "shape": (100, 100),
            "static_base": sp._joint_static_base(full_layers, candidate_lookup),
            "evidence": evidence.reshape(-1),
            "rest": {
                0: sp._sparse_top_depth(vertices, faces, faces_a, camera, (100, 100)),
                1: sp._sparse_top_depth(vertices, faces, faces_b, camera, (100, 100)),
            },
            "moved": {
                0: sp._sparse_top_depth(moved_vertices, faces, faces_a, camera, (100, 100)),
                1: sp._sparse_top_depth(moved_vertices, faces, faces_b, camera, (100, 100)),
            },
        }

        selected, _trace, totals = sp._joint_select_patches([view], [0, 1], 1, 0.25, 1, 0.05, 6)
        self.assertEqual(sorted(selected), [0, 1])
        self.assertEqual(int(totals["spurious_pixels"]), 0)

    def test_body_edge_ring_removes_boundary_flicker_keeps_area_motion(self):
        # Fixed body silhouette: square [20:60, 20:60]. Edge flicker hugs its
        # boundary; a true moving part sweeps an area well away from it.
        silhouette = np.zeros((100, 100), dtype=bool)
        silhouette[20:60, 20:60] = True
        ring = sp._body_edge_ring(silhouette, 3)
        self.assertIsNone(sp._body_edge_ring(silhouette, 0))

        motion = np.zeros((100, 100), dtype=bool)
        motion[19:22, 20:60] = True   # boundary flicker band (top edge)
        motion[70:90, 70:90] = True   # real swept-area change, far from the body edge
        cleaned = motion & ~ring
        self.assertFalse(np.any(cleaned[19:22, 20:60]))
        self.assertTrue(np.all(cleaned[70:90, 70:90]))
        # Deep interior of the body stays eligible (only the rim is suppressed).
        self.assertFalse(ring[40, 40])

    def test_joint_round_adoption_is_score_based(self):
        round0 = {"kept": [2, 7, 6, 1], "totals": {"score": 2693.2}}
        # The real failure: a refit round with kept patches but a WORSE score
        # (and a size-biased lower subset fit) must not replace round 0.
        self.assertFalse(sp._joint_round_is_better(round0, [2, 7, 1], 2523.5))
        self.assertTrue(sp._joint_round_is_better(round0, [2, 7, 1], 2800.0))
        self.assertFalse(sp._joint_round_is_better(round0, [], 9999.0))
        self.assertTrue(sp._joint_round_is_better(None, [], 0.0))
        self.assertTrue(sp._joint_round_is_better({"kept": [], "totals": {"score": 0.0}}, [3], 1.0))

    def test_cross_candidate_selection_borrows_hinge_track_and_keeps_patch_exclusive(self):
        # Two source candidates contain one patch each.  Track 00 moves both
        # patches to the wrong image locations; track 01 is the reliable hinge
        # track and lands both canonical patches on their observed change blobs.
        # A real handle patch must therefore be allowed to select track 01 even
        # though it originated in source candidate 00.
        rest_a = (np.array([0, 1], dtype=np.int64), np.ones(2, dtype=np.float32))
        rest_b = (np.array([4, 5], dtype=np.int64), np.ones(2, dtype=np.float32))
        view = {
            "state_id": "s0",
            "serial": "cam",
            "group": "g0",
            "shape": (1, 12),
            "static_base": np.full(12, np.inf, dtype=np.float32),
            "evidence": np.isin(np.arange(12), [0, 1, 2, 3, 4, 5, 6, 7]),
            "rest": {"src00_patch08": rest_a, "src01_patch00": rest_b},
            "moved": {
                "src00_patch08__track00": (np.array([10, 11], dtype=np.int64), np.ones(2, dtype=np.float32)),
                "src00_patch08__track01": (np.array([2, 3], dtype=np.int64), np.ones(2, dtype=np.float32)),
                "src01_patch00__track00": (np.array([10, 11], dtype=np.int64), np.ones(2, dtype=np.float32)),
                "src01_patch00__track01": (np.array([6, 7], dtype=np.int64), np.ones(2, dtype=np.float32)),
            },
        }
        hypotheses = {
            "src00_patch08__track00": {"patch_key": "src00_patch08", "track_part_id": 0},
            "src00_patch08__track01": {"patch_key": "src00_patch08", "track_part_id": 1},
            "src01_patch00__track00": {"patch_key": "src01_patch00", "track_part_id": 0},
            "src01_patch00__track01": {"patch_key": "src01_patch00", "track_part_id": 1},
        }
        args = argparse.Namespace(
            stage_b_patch_joint_spurious_weight=0.25,
            stage_b_cross_max_selected=4,
            stage_b_cross_min_gain=0.001,
            stage_b_cross_min_gain_frac=0.05,
        )

        selected, _trace, totals = sp._cross_select_hypotheses([view], hypotheses, 1, args)

        self.assertEqual(set(selected), {"src00_patch08__track01", "src01_patch00__track01"})
        self.assertGreater(totals["score"], 0.9)
        # The same source patch cannot be assigned to both its weak and hinge tracks.
        self.assertEqual(sum(key.startswith("src00_patch08__") for key in selected), 1)

    def test_cross_prediction_keeps_patch_at_rest_without_that_track_pose(self):
        hypothesis = "src00_patch08__track01"
        view = {
            "static_base": np.full(8, np.inf, dtype=np.float32),
            "rest": {"src00_patch08": (np.array([1, 2], dtype=np.int64), np.ones(2, dtype=np.float32))},
            # The track was not fit for this state/camera, so no moved footprint
            # exists. The patch must remain static rather than disappearing.
            "moved": {},
        }
        hypotheses = {hypothesis: {"patch_key": "src00_patch08", "track_part_id": 1}}

        predicted = sp._cross_predicted_change(view, [hypothesis], hypotheses)

        self.assertFalse(np.any(predicted))

    def test_cross_track_gate_keeps_low_fit_source_but_not_its_pose(self):
        summaries = [
            {"part_id": 0, "status": "moving_candidate", "median_fit_iou": 0.060},
            {"part_id": 1, "status": "moving_candidate", "median_fit_iou": 0.052},
        ]
        track = [
            {"status": "fit", "state_id": "s0"},
            {"status": "fit", "state_id": "s1"},
        ]
        tracks = {"part_00": track, "part_01": track}
        reg = {
            "s0": {"placement_group": "g0"},
            "s1": {"placement_group": "g1"},
        }
        args = argparse.Namespace(
            stage_b_patch_min_fit_iou=0.05,
            stage_b_patch_min_groups=2,
            stage_b_cross_min_track_fit_iou=0.05,
            stage_b_cross_track_min_fit_ratio=0.90,
            stage_b_cross_max_tracks=2,
        )

        source_ids, track_ids = sp._cross_source_and_track_ids(summaries, tracks, reg, args)

        self.assertEqual(source_ids, [0, 1])
        self.assertEqual(track_ids, [0])

    def test_cross_track_gate_rejects_the_least_bad_weak_track(self):
        summaries = [
            {"part_id": 0, "status": "moving_candidate", "median_fit_iou": 0.060},
            {"part_id": 1, "status": "moving_candidate", "median_fit_iou": 0.052},
        ]
        track = [
            {"status": "fit", "state_id": "s0"},
            {"status": "fit", "state_id": "s1"},
        ]
        args = argparse.Namespace(
            stage_b_patch_min_fit_iou=0.05,
            stage_b_patch_min_groups=2,
            stage_b_cross_min_track_fit_iou=0.08,
            stage_b_cross_track_min_fit_ratio=0.90,
            stage_b_cross_max_tracks=2,
        )

        source_ids, track_ids = sp._cross_source_and_track_ids(
            summaries,
            {"part_00": track, "part_01": track},
            {"s0": {"placement_group": "g0"}, "s1": {"placement_group": "g1"}},
            args,
        )

        self.assertEqual(source_ids, [0, 1])
        self.assertEqual(track_ids, [])

    def test_cross_track_gate_accepts_an_integer_keyed_probe_track(self):
        args = argparse.Namespace(
            stage_b_patch_min_groups=2,
            stage_b_cross_min_track_fit_iou=0.08,
            stage_b_cross_track_min_fit_ratio=0.90,
            stage_b_cross_max_tracks=2,
        )
        track = [
            {"status": "fit", "state_id": "s0"},
            {"status": "fit", "state_id": "s1"},
        ]

        track_ids = sp._cross_track_ids(
            [0],
            {0: {"median_fit_iou": 0.112}},
            {0: track},
            {"s0": {"placement_group": "g0"}, "s1": {"placement_group": "g1"}},
            args,
        )

        self.assertEqual(track_ids, [0])

    def test_cross_anchor_drift_rejects_a_refit_that_changes_the_pose(self):
        anchor = np.eye(4, dtype=np.float64)
        refit = np.eye(4, dtype=np.float64)
        angle = np.deg2rad(25.0)
        refit[:3, :3] = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        refit[0, 3] = 0.04
        args = argparse.Namespace(
            stage_b_cross_refit_max_anchor_rotation_deg=15.0,
            stage_b_cross_refit_max_anchor_translation_frac=0.03,
        )

        drift = sp._cross_anchor_drift(
            [{"status": "fit", "state_id": "s0", "T_body_part": anchor.tolist()}],
            [{"status": "fit", "state_id": "s0", "T_body_part": refit.tolist()}],
            1.0,
            args,
        )

        self.assertFalse(drift["passes"])
        self.assertEqual(drift["reason"], "anchor_rotation_drift")
        self.assertGreater(drift["max_rotation_deg"], 24.0)

    def test_cross_anchor_drift_accepts_a_nearby_refit(self):
        anchor = np.eye(4, dtype=np.float64)
        refit = np.eye(4, dtype=np.float64)
        angle = np.deg2rad(4.0)
        refit[:3, :3] = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        refit[0, 3] = 0.01
        args = argparse.Namespace(
            stage_b_cross_refit_max_anchor_rotation_deg=15.0,
            stage_b_cross_refit_max_anchor_translation_frac=0.03,
        )

        drift = sp._cross_anchor_drift(
            [{"status": "fit", "state_id": "s0", "T_body_part": anchor.tolist()}],
            [{"status": "fit", "state_id": "s0", "T_body_part": refit.tolist()}],
            1.0,
            args,
        )

        self.assertTrue(drift["passes"])
        self.assertEqual(drift["reason"], "locked")

    def test_cross_refit_faces_include_the_probe_anchor_once(self):
        hypotheses = {
            "src00_patch05__track00": {"patch_key": "src00_patch05", "track_part_id": 0},
            "src00_patch08__track00": {"patch_key": "src00_patch08", "track_part_id": 0},
            "src01_patch01__track00": {"patch_key": "src01_patch01", "track_part_id": 0},
        }
        patch_entries = {
            "src00_patch05": {"faces": np.array([1, 2], dtype=np.int64)},
            "src00_patch08": {"faces": np.array([2, 3], dtype=np.int64)},
            "src01_patch01": {"faces": np.array([4, 5], dtype=np.int64)},
        }

        faces, keys = sp._cross_refit_faces(
            ["src00_patch05__track00", "src01_patch01__track00"],
            hypotheses,
            patch_entries,
            0,
            "src00_patch08",
            True,
        )

        self.assertEqual(keys, ["src00_patch05", "src01_patch01", "src00_patch08"])
        self.assertTrue(np.array_equal(faces, np.array([1, 2, 3, 4, 5], dtype=np.int64)))

    def test_cross_objective_counts_unexplained_evidence_without_prediction(self):
        view = {
            "group": "g0",
            "static_base": np.full(4, np.inf, dtype=np.float32),
            "evidence": np.ones(4, dtype=bool),
            "rest": {},
            "moved": {},
        }

        totals = sp._cross_joint_objective([view], [], {}, 1, 0.25)

        self.assertEqual(totals["used_views"], 1)
        self.assertEqual(totals["evidence_pixels"], 4)
        self.assertEqual(totals["unexplained_pixels"], 4)
        self.assertEqual(totals["coverage"], 0.0)

    def test_cross_coverage_gate_rejects_tiny_explanation(self):
        args = argparse.Namespace(stage_b_cross_min_coverage=0.05)

        self.assertFalse(sp._cross_coverage_passes({"coverage": 0.0184}, args))
        self.assertTrue(sp._cross_coverage_passes({"coverage": 0.0500}, args))

    def test_stage_c_stops_when_rgb_patch_validation_fails(self):
        result = sp._run_stage_c_solid_parts(
            [],
            {},
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
            {"tracks": {"part_00": []}, "motion_face_part_source": "no_rgb_validated_patch"},
            None,
            "",
        )

        self.assertEqual(result["status"], "no_rgb_validated_patch")
        self.assertEqual(result["part_count"], 0)


if __name__ == "__main__":
    unittest.main()
