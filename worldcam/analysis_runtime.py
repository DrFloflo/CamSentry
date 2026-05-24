"""Analysis orchestration helpers for the WorldCam main loop."""

from ultralytics import YOLO

from worldcam.detection import Detection, run_sahi_analysis, run_yolo_analysis
from worldcam.models import load_pose_model, load_segmentation_model
from worldcam.pose import Pose, run_pose_analysis
from worldcam.runtime import RuntimeState
from worldcam.segmentation import SegmentationMask, run_segmentation_analysis
from worldcam.tracking import ObjectTracker


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
                segmentation_model = load_segmentation_model(device)
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
            pose_model = load_pose_model(device)
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
    )

    if tracking_enabled:
        runtime.latest_object_tracks = object_tracker.update(runtime.latest_detections)
    else:
        runtime.latest_object_tracks = []
        object_tracker.reset()

    return pose_model, segmentation_model
