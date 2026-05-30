"""State container for the WorldCam menu."""

from dataclasses import dataclass, field
import multiprocessing as mp

from worldcam.core.config import COUNTING_ZONE_EDIT_ENABLED, COUNTING_ZONE_ENABLED, PERSON_TRACK_ENABLED, SAHI_ENABLED, SEGMENTATION_ENABLED


@dataclass(frozen=True)
class MenuChanges:
    """Named flags describing which menu values changed."""

    class_selection_changed: bool = False
    pose_toggled: bool = False
    sahi_toggled: bool = False
    tracking_toggled: bool = False
    segmentation_toggled: bool = False
    threshold_changed: bool = False
    counting_zone_toggled: bool = False
    counting_zone_edit_toggled: bool = False

    def merge(self, other: "MenuChanges") -> "MenuChanges":
        """Return changes containing every flag set in either instance."""
        return MenuChanges(
            class_selection_changed=self.class_selection_changed or other.class_selection_changed,
            pose_toggled=self.pose_toggled or other.pose_toggled,
            sahi_toggled=self.sahi_toggled or other.sahi_toggled,
            tracking_toggled=self.tracking_toggled or other.tracking_toggled,
            segmentation_toggled=self.segmentation_toggled or other.segmentation_toggled,
            threshold_changed=self.threshold_changed or other.threshold_changed,
            counting_zone_toggled=self.counting_zone_toggled or other.counting_zone_toggled,
            counting_zone_edit_toggled=self.counting_zone_edit_toggled or other.counting_zone_edit_toggled,
        )


@dataclass(frozen=True)
class MenuSnapshot:
    """Stable menu values used for frame analysis and display."""

    selected_class_names: set[str]
    pose_enabled: bool
    sahi_enabled: bool
    tracking_enabled: bool
    segmentation_enabled: bool
    display_threshold: float
    counting_zone_enabled: bool
    counting_zone_edit_enabled: bool


@dataclass
class MenuState:
    """Mutable state for the YOLO class selection menu."""

    is_open: bool = False
    pose_enabled: bool = False
    sahi_enabled: bool = SAHI_ENABLED
    tracking_enabled: bool = PERSON_TRACK_ENABLED
    segmentation_enabled: bool = SEGMENTATION_ENABLED
    display_threshold: float = 0.5
    counting_zone_enabled: bool = COUNTING_ZONE_ENABLED
    counting_zone_edit_enabled: bool = COUNTING_ZONE_EDIT_ENABLED
    class_selection_changed: bool = False
    pose_toggled: bool = False
    sahi_toggled: bool = False
    tracking_toggled: bool = False
    segmentation_toggled: bool = False
    threshold_changed: bool = False
    counting_zone_toggled: bool = False
    counting_zone_edit_toggled: bool = False
    menu_process: mp.Process | None = None
    event_queue: mp.Queue = field(default_factory=mp.Queue, repr=False)
    command_queue: mp.Queue = field(default_factory=mp.Queue, repr=False)
