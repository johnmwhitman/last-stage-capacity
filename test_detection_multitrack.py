"""
Pytest tests for last_stage_capacity/_detection.py in multi-track tracking scenarios.

Validates that:
- Two overlapping synthetic tracks maintain distinct, stable IDs across frames
  when features are processed by capacity reduction modules from _detection.py.
- A third track born mid-sequence is assigned a fresh unique track ID.
- Tested using synthetic bounding boxes and synthetic features only.
"""

import pytest
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from last_stage_capacity._detection import (
    LinearProjectionReduction,
    SEReduction,
    ConditionalCapacityBlock,
    CapacityReductionStack,
)


class SimpleMultiTracker:
    """
    Lightweight synthetic multi-object tracker that associates detections across
    frames using a combination of spatial bounding-box IoU and appearance embeddings
    (processed via last-stage capacity reduction modules).
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        feature_similarity_weight: float = 0.5,
        max_dist: float = 0.7,
    ):
        self.iou_threshold = iou_threshold
        self.feature_similarity_weight = feature_similarity_weight
        self.max_dist = max_dist
        self.next_id = 1
        # Maps track_id -> dict(box=Tensor(4), feat=Tensor(C), last_frame=int)
        self.active_tracks: dict[int, dict] = {}

    def update(
        self,
        boxes: torch.Tensor,
        features: torch.Tensor,
        frame_idx: int,
    ) -> list[int]:
        """
        Update tracker with detections from current frame.

        Args:
            boxes: Tensor of shape (N, 4) in (x1, y1, x2, y2) format.
            features: Tensor of shape (N, C) representing appearance embeddings
                      (e.g., reduced capacity features).
            frame_idx: Index of current frame.

        Returns:
            assigned_ids: List of track IDs for each input detection.
        """
        N = len(boxes)
        if N == 0:
            return []

        # Normalize appearance embeddings for cosine similarity
        norm_feats = F.normalize(features, p=2, dim=-1)

        assigned_ids = [None] * N
        matched_track_ids = set()
        unmatched_dets = set(range(N))

        if self.active_tracks:
            track_ids = list(self.active_tracks.keys())
            prev_boxes = torch.stack([self.active_tracks[tid]["box"] for tid in track_ids])
            prev_feats = torch.stack([self.active_tracks[tid]["feat"] for tid in track_ids])

            # IoU matrix: shape (N, num_tracks)
            ious = box_iou(boxes, prev_boxes)

            # Cosine similarity matrix: shape (N, num_tracks)
            sims = torch.mm(norm_feats, prev_feats.t())

            # Combined affinity score: weighted combination of spatial IoU and feature similarity
            w_feat = self.feature_similarity_weight
            affinity = (1.0 - w_feat) * ious + w_feat * torch.clamp(sims, min=0.0)

            # Greedy bipartite matching
            num_candidates = min(N, len(track_ids))
            for _ in range(num_candidates):
                max_val, flat_idx = torch.max(affinity.view(-1), dim=0)
                if max_val.item() <= (1.0 - self.max_dist):
                    break
                det_idx = flat_idx.item() // len(track_ids)
                tr_idx = flat_idx.item() % len(track_ids)
                tid = track_ids[tr_idx]

                if det_idx in unmatched_dets and tid not in matched_track_ids:
                    assigned_ids[det_idx] = tid
                    matched_track_ids.add(tid)
                    unmatched_dets.remove(det_idx)
                    # Update track state (box and exponential moving average of feature)
                    self.active_tracks[tid]["box"] = boxes[det_idx]
                    self.active_tracks[tid]["feat"] = F.normalize(
                        0.7 * self.active_tracks[tid]["feat"] + 0.3 * norm_feats[det_idx],
                        p=2,
                        dim=-1,
                    )
                    self.active_tracks[tid]["last_frame"] = frame_idx

                affinity[det_idx, :] = -1.0
                affinity[:, tr_idx] = -1.0

        # Assign fresh IDs to remaining unmatched detections
        for det_idx in sorted(unmatched_dets):
            new_id = self.next_id
            self.next_id += 1
            assigned_ids[det_idx] = new_id
            self.active_tracks[new_id] = {
                "box": boxes[det_idx],
                "feat": norm_feats[det_idx],
                "last_frame": frame_idx,
            }

        return assigned_ids


def _build_synthetic_trajectory():
    """
    Constructs a 5-frame sequence:
    - Frame 0: Track A at (10, 10, 50, 50), Track B at (15, 15, 55, 55) [Overlapping]
    - Frame 1: Track A at (14, 14, 54, 54), Track B at (20, 20, 60, 60) [Still overlapping]
    - Frame 2: Track A at (18, 18, 58, 58), Track B at (25, 25, 65, 65), Track C born at (80, 80, 120, 120)
    - Frame 3: Track A at (22, 22, 62, 62), Track B at (30, 30, 70, 70), Track C moves to (85, 85, 125, 125)
    - Frame 4: Track A at (26, 26, 66, 66), Track B at (35, 35, 75, 75), Track C moves to (90, 90, 130, 130)

    Also returns distinct synthetic base feature maps for the objects.
    """
    frames = [
        # Frame 0
        {
            "boxes": torch.tensor([
                [10.0, 10.0, 50.0, 50.0],  # Track A
                [15.0, 15.0, 55.0, 55.0],  # Track B (overlapping with A)
            ]),
            "obj_keys": ["A", "B"],
        },
        # Frame 1
        {
            "boxes": torch.tensor([
                [14.0, 14.0, 54.0, 54.0],  # Track A
                [20.0, 20.0, 60.0, 60.0],  # Track B
            ]),
            "obj_keys": ["A", "B"],
        },
        # Frame 2 (Track C is born)
        {
            "boxes": torch.tensor([
                [18.0, 18.0, 58.0, 58.0],    # Track A
                [25.0, 25.0, 65.0, 65.0],    # Track B
                [80.0, 80.0, 120.0, 120.0],  # Track C (born mid-sequence)
            ]),
            "obj_keys": ["A", "B", "C"],
        },
        # Frame 3
        {
            "boxes": torch.tensor([
                [22.0, 22.0, 62.0, 62.0],    # Track A
                [30.0, 30.0, 70.0, 70.0],    # Track B
                [85.0, 85.0, 125.0, 125.0],  # Track C
            ]),
            "obj_keys": ["A", "B", "C"],
        },
        # Frame 4
        {
            "boxes": torch.tensor([
                [26.0, 26.0, 66.0, 66.0],    # Track A
                [35.0, 35.0, 75.0, 75.0],    # Track B
                [90.0, 90.0, 130.0, 130.0],  # Track C
            ]),
            "obj_keys": ["A", "B", "C"],
        },
    ]

    torch.manual_seed(42)
    # Generate distinct base feature representations for each object identity
    base_features = {
        "A": torch.randn(1, 256, 16, 16) + 2.0,
        "B": torch.randn(1, 256, 16, 16) - 2.0,
        "C": torch.randn(1, 256, 16, 16) * 3.0,
    }

    return frames, base_features


def test_multitrack_with_linear_projection_reduction():
    """
    Test that two overlapping synthetic tracks keep distinct, stable IDs across frames
    and a third track born mid-sequence receives a fresh ID using LinearProjectionReduction.
    """
    frames, base_features = _build_synthetic_trajectory()
    reducer = LinearProjectionReduction(in_channels=256, out_channels=64)
    reducer.eval()

    tracker = SimpleMultiTracker(feature_similarity_weight=0.5)

    track_history = []

    with torch.no_grad():
        for frame_idx, frame in enumerate(frames):
            boxes = frame["boxes"]
            keys = frame["obj_keys"]

            # Generate synthetic detection features per object with slight temporal noise
            feats_list = []
            for k in keys:
                noisy_map = base_features[k] + 0.05 * torch.randn_like(base_features[k])
                # Reduce capacity using LinearProjectionReduction (BCHW -> BCHW)
                reduced_map = reducer(noisy_map)
                # Global pooling to form embedding vector (1, C)
                emb = reduced_map.mean(dim=[-2, -1])
                feats_list.append(emb)

            features = torch.cat(feats_list, dim=0)  # (N, 64)
            assigned_ids = tracker.update(boxes, features, frame_idx)
            track_history.append(dict(zip(keys, assigned_ids)))

    # Verify Track A and Track B maintain distinct, stable IDs across all 5 frames
    id_a = track_history[0]["A"]
    id_b = track_history[0]["B"]
    assert id_a != id_b, f"Track A ({id_a}) and Track B ({id_b}) must have distinct IDs"

    for t, frame_record in enumerate(track_history):
        assert frame_record["A"] == id_a, f"Frame {t}: Track A changed ID to {frame_record['A']}"
        assert frame_record["B"] == id_b, f"Frame {t}: Track B changed ID to {frame_record['B']}"

    # Verify Track C (born in frame 2) gets a fresh ID distinct from A and B and stays stable
    id_c = track_history[2]["C"]
    assert id_c not in (id_a, id_b), f"Track C ID ({id_c}) must be distinct from A ({id_a}) and B ({id_b})"

    for t in range(2, 5):
        assert track_history[t]["C"] == id_c, f"Frame {t}: Track C changed ID to {track_history[t]['C']}"


def test_multitrack_with_se_reduction():
    """
    Test tracking stability with SEReduction (channel attention + reduction).
    """
    frames, base_features = _build_synthetic_trajectory()
    reducer = SEReduction(in_channels=256, reduction_ratio=0.25)
    reducer.eval()

    tracker = SimpleMultiTracker(feature_similarity_weight=0.5)

    track_history = []

    with torch.no_grad():
        for frame_idx, frame in enumerate(frames):
            boxes = frame["boxes"]
            keys = frame["obj_keys"]

            feats_list = []
            for k in keys:
                noisy_map = base_features[k] + 0.05 * torch.randn_like(base_features[k])
                reduced_map = reducer(noisy_map)
                emb = reduced_map.mean(dim=[-2, -1])
                feats_list.append(emb)

            features = torch.cat(feats_list, dim=0)
            assigned_ids = tracker.update(boxes, features, frame_idx)
            track_history.append(dict(zip(keys, assigned_ids)))

    id_a = track_history[0]["A"]
    id_b = track_history[0]["B"]
    assert id_a != id_b

    for t, frame_record in enumerate(track_history):
        assert frame_record["A"] == id_a
        assert frame_record["B"] == id_b

    id_c = track_history[2]["C"]
    assert id_c not in (id_a, id_b)

    for t in range(2, 5):
        assert track_history[t]["C"] == id_c


def test_multitrack_with_conditional_capacity_block():
    """
    Test tracking stability with ConditionalCapacityBlock (spatial gating).
    """
    frames, base_features = _build_synthetic_trajectory()
    reducer = ConditionalCapacityBlock(in_channels=256, reduction_ratio=0.25)
    reducer.eval()

    tracker = SimpleMultiTracker(feature_similarity_weight=0.5)

    track_history = []

    with torch.no_grad():
        for frame_idx, frame in enumerate(frames):
            boxes = frame["boxes"]
            keys = frame["obj_keys"]

            feats_list = []
            for k in keys:
                noisy_map = base_features[k] + 0.05 * torch.randn_like(base_features[k])
                reduced_map = reducer(noisy_map)
                emb = reduced_map.mean(dim=[-2, -1])
                feats_list.append(emb)

            features = torch.cat(feats_list, dim=0)
            assigned_ids = tracker.update(boxes, features, frame_idx)
            track_history.append(dict(zip(keys, assigned_ids)))

    id_a = track_history[0]["A"]
    id_b = track_history[0]["B"]
    assert id_a != id_b

    for t, frame_record in enumerate(track_history):
        assert frame_record["A"] == id_a
        assert frame_record["B"] == id_b

    id_c = track_history[2]["C"]
    assert id_c not in (id_a, id_b)

    for t in range(2, 5):
        assert track_history[t]["C"] == id_c


def test_multitrack_with_capacity_reduction_stack():
    """
    Test tracking stability with CapacityReductionStack across multi-stage reduction.
    """
    frames, base_features = _build_synthetic_trajectory()
    # 256 -> 128 -> 64
    stack = CapacityReductionStack(channels=[256, 128, 64], block_type="se")
    stack.eval()

    tracker = SimpleMultiTracker(feature_similarity_weight=0.5)

    track_history = []

    with torch.no_grad():
        for frame_idx, frame in enumerate(frames):
            boxes = frame["boxes"]
            keys = frame["obj_keys"]

            feats_list = []
            for k in keys:
                noisy_map = base_features[k] + 0.05 * torch.randn_like(base_features[k])
                # stack returns list of outputs per stage; take the final stage
                reduced_maps = stack(noisy_map)
                final_map = reduced_maps[-1]
                emb = final_map.mean(dim=[-2, -1])
                feats_list.append(emb)

            features = torch.cat(feats_list, dim=0)
            assigned_ids = tracker.update(boxes, features, frame_idx)
            track_history.append(dict(zip(keys, assigned_ids)))

    id_a = track_history[0]["A"]
    id_b = track_history[0]["B"]
    assert id_a != id_b

    for t, frame_record in enumerate(track_history):
        assert frame_record["A"] == id_a
        assert frame_record["B"] == id_b

    id_c = track_history[2]["C"]
    assert id_c not in (id_a, id_b)

    for t in range(2, 5):
        assert track_history[t]["C"] == id_c


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
