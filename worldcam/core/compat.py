"""Compatibility patches for third-party model checkpoints."""

from ultralytics.nn.modules import head as ultralytics_head


def patch_ultralytics_pose26() -> None:
    """Map older Pose26 checkpoint heads to the available Ultralytics Pose class."""
    if not hasattr(ultralytics_head, "Pose26") and hasattr(ultralytics_head, "Pose"):
        ultralytics_head.Pose26 = ultralytics_head.Pose
