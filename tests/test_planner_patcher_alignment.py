from __future__ import annotations

from patcher import EXACT_ALLOWED_FIELDS
from planner import ALLOWED_FIELDS


def test_planner_and_patcher_allowed_fields_stay_aligned() -> None:
    assert ALLOWED_FIELDS == EXACT_ALLOWED_FIELDS


def test_only_minimal_m1_fields_are_auto_editable() -> None:
    assert ALLOWED_FIELDS == {
        "training.num_envs",
        "training.max_iterations",
        "training.learning_rate",
        "training.entropy_coef",
        "training.clip_param",
        "training.num_mini_batches",
        "command.lin_vel_x",
        "command.lin_vel_y",
        "command.ang_vel_z",
        "command.heading",
        "reward.track_lin_vel_xy_exp.weight",
        "reward.track_ang_vel_z_exp.weight",
        "reward.feet_air_time.weight",
        "reward.flat_orientation_l2.weight",
    }
