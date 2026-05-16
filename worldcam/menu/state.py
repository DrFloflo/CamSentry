"""State container for the WorldCam menu."""

from dataclasses import dataclass, field
import multiprocessing as mp

from worldcam.config import PERSON_TRACK_ENABLED, SAHI_ENABLED, SEGMENTATION_ENABLED


@dataclass
class MenuState:
    """Mutable state for the YOLO class selection menu."""

    is_open: bool = False
    index: int = 0
    pose_enabled: bool = False
    sahi_enabled: bool = SAHI_ENABLED
    tracking_enabled: bool = PERSON_TRACK_ENABLED
    segmentation_enabled: bool = SEGMENTATION_ENABLED
    display_threshold: float = 0.5
    class_selection_changed: bool = False
    pose_toggled: bool = False
    sahi_toggled: bool = False
    tracking_toggled: bool = False
    segmentation_toggled: bool = False
    threshold_changed: bool = False
    menu_process: mp.Process | None = None
    event_queue: mp.Queue = field(default_factory=mp.Queue, repr=False)
    command_queue: mp.Queue = field(default_factory=mp.Queue, repr=False)
