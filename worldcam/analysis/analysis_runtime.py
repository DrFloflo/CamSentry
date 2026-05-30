"""Analysis orchestration helpers for the WorldCam main loop."""

from ultralytics import YOLO

from worldcam.analysis.counting_zone import ZonePoints
from worldcam.analysis.detection import Detection, deduplicate_detections, run_sahi_analysis, run_yolo_analysis
from worldcam.core.models import load_pose_model, load_segmentation_model
from worldcam.analysis.pose import Pose, run_pose_analysis
from worldcam.core.runtime import RuntimeState
from worldcam.analysis.segmentation import SegmentationMask, run_segmentation_analysis
from worldcam.analysis.tracking import ObjectTracker


def run_model_analysis(
    frame,
    model: YOLO,
    pose_model: YOLO | None,
    segmentation_model: YOLO | None,
    device: str,
    selected_class_names: set[str],
    latest_detections: list[Detection],
    latest_poses: list[Pose],
    latest_segmentations: list[SegmentationMask],
    pose_enabled: bool,
    segmentation_enabled: bool,
    sahi_enabled: bool = False,
    model_key: str = "",
) -> tuple[list[Detection], list[Pose], list[SegmentationMask], YOLO | None, YOLO | None]:
    """Run object and optional pose analysis while preserving previous results on errors."""
    try:
        if sahi_enabled:
            latest_detections = run_sahi_analysis(frame, model, device, selected_class_names)
        else:
            latest_detections = run_yolo_analysis(frame, model, device, selected_class_names)
    except Exception as exc:
        print(f"Erreur pendant l'analyse YOLO: {exc}")

    if segmentation_enabled:
        if segmentation_model is None:
            try:
                segmentation_model = load_segmentation_model(device, model_key)
            except Exception as exc:
                print(f"Erreur pendant le chargement du modèle segmentation YOLO: {exc}")
                latest_segmentations = []
        if segmentation_model is not None:
            try:
                latest_segmentations = run_segmentation_analysis(frame, segmentation_model, device, selected_class_names)
            except Exception as exc:
                print(f"Erreur pendant l'analyse de segmentation YOLO: {exc}")
    else:
        latest_segmentations = []

    if not pose_enabled:
        return latest_detections, [], latest_segmentations, pose_model, segmentation_model

    if pose_model is None:
        try:
            pose_model = load_pose_model(device, model_key)
        except Exception as exc:
            print(f"Erreur pendant le chargement du modèle pose YOLO: {exc}")
            return latest_detections, latest_poses, latest_segmentations, pose_model, segmentation_model

    try:
        latest_poses = run_pose_analysis(frame, pose_model, device)
    except Exception as exc:
        print(f"Erreur pendant l'analyse de pose YOLO: {exc}")

    return latest_detections, latest_poses, latest_segmentations, pose_model, segmentation_model


def update_runtime_analysis(
    runtime: RuntimeState,
    frame,
    model: YOLO,
    pose_model: YOLO | None,
    segmentation_model: YOLO | None,
    device: str,
    selected_class_names: set[str],
    pose_enabled: bool,
    segmentation_enabled: bool,
    sahi_enabled: bool,
    tracking_enabled: bool,
    object_tracker: ObjectTracker,
    counting_zone_points: ZonePoints | None = None,
    counting_zone_enabled: bool = False,
    model_key: str = "",
) -> tuple[YOLO | None, YOLO | None]:
    """Run enabled analyses and refresh runtime result caches."""
    (
        runtime.latest_detections,
        runtime.latest_poses,
        runtime.latest_segmentations,
        pose_model,
        segmentation_model,
    ) = run_model_analysis(
        frame,
        model,
        pose_model,
        segmentation_model,
        device,
        selected_class_names,
        runtime.latest_detections,
        runtime.latest_poses,
        runtime.latest_segmentations,
        pose_enabled,
        segmentation_enabled,
        sahi_enabled,
        model_key,
    )

    runtime.latest_detections = deduplicate_detections(runtime.latest_detections)

    if tracking_enabled:
        runtime.latest_object_tracks = object_tracker.update(runtime.latest_detections, counting_zone_points, counting_zone_enabled)
    else:
        runtime.latest_object_tracks = []

    return pose_model, segmentation_model
