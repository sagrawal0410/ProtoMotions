# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Convert legged_lab AMP .pkl motions into ProtoMotions-compatible .npz files.

legged_lab stores AMP motions as joblib pickles with keys ``root_pos`` (N, 3),
``root_rot`` (N, 4, wxyz), ``dof_pos`` (N, 29) and ``fps``. The dof ordering in
those files (``LAB_ORDER``) differs from the ProtoMotions MJCF dof ordering
(``PROTO_ORDER``), so the joint columns are permuted here.

The resulting .npz files use the keys expected by
``convert_pyroki_retargeted_robot_motions_to_proto.py``:
``base_frame_pos``, ``base_frame_wxyz`` and ``joint_angles``.

Usage:
    python data/scripts/pkl_to_npz.py <in_dir_with_pkls> <out_dir_for_npz>
"""

import glob
import os
import sys

import joblib  # legged_lab saves with joblib
import numpy as np

# ProtoMotions g1_bm_box_feet.xml joint order (target)
PROTO_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# legged_lab lab_dof_names (source order of dof_pos in the pkl)
LAB_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
# Column permutation mapping lab dof order -> proto dof order.
PERM = [LAB_ORDER.index(name) for name in PROTO_ORDER]


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: python {os.path.basename(__file__)} <in_dir> <out_dir>")

    in_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    pkl_paths = sorted(glob.glob(os.path.join(in_dir, "*.pkl")))
    if not pkl_paths:
        sys.exit(f"No .pkl files found in {in_dir!r}")

    fps_seen = set()
    for p in pkl_paths:
        d = joblib.load(p)
        root_pos = np.asarray(d["root_pos"], dtype=np.float32)  # (N, 3) world
        root_quat = np.asarray(d["root_rot"], dtype=np.float32)  # (N, 4) wxyz
        dof_lab = np.asarray(d["dof_pos"], dtype=np.float32)  # (N, 29) lab order
        assert dof_lab.shape[1] == 29, f"{p}: expected 29 dofs, got {dof_lab.shape[1]}"
        dof_proto = dof_lab[:, PERM]  # reorder -> proto

        fps = int(round(float(d["fps"])))
        fps_seen.add(fps)
        name = os.path.splitext(os.path.basename(p))[0]
        np.savez(
            os.path.join(out_dir, f"{name}.npz"),
            base_frame_pos=root_pos,
            base_frame_wxyz=root_quat,
            joint_angles=dof_proto,
        )
        print(f"{name}: {root_pos.shape[0]} frames, fps={fps}")

    print("fps values across files:", fps_seen)
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
