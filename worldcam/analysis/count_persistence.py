"""JSON persistence for daily WorldCam vehicle counts."""

from __future__ import annotations

from datetime import date, datetime
import json
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from worldcam.core.config import VEHICLE_COUNT_DATA_DIR, VEHICLE_COUNT_TIMEZONE

SCHEMA_VERSION = 1


class VehicleCountStore:
    """Persist one JSON vehicle-count file per local calendar day."""

    def __init__(self, data_dir: str | Path = VEHICLE_COUNT_DATA_DIR, timezone_name: str = VEHICLE_COUNT_TIMEZONE) -> None:
        self.data_dir = Path(data_dir)
        self.timezone_name = timezone_name
        self.timezone = self._load_timezone(timezone_name)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.current_date = self.today()
        self.document = self.load_day(self.current_date)

    def today(self) -> date:
        """Return today's date in the configured counting timezone."""
        return datetime.now(self.timezone).date()

    def total(self) -> int:
        """Return the persisted total for the currently loaded day."""
        return int(self.document.get("summary", {}).get("total", 0))

    def open_day(self, day: date | None = None) -> int:
        """Load a day file and return its current total."""
        self.current_date = day or self.today()
        self.document = self.load_day(self.current_date)
        return self.total()

    def record_vehicle_event(
        self,
        *,
        track_id: int | None,
        class_name: str,
        total_after_event: int,
        speed_kmh: float | None,
    ) -> None:
        """Append one vehicle passage event and atomically persist the daily JSON document."""
        current_date = self.today()
        if current_date != self.current_date:
            self.open_day(current_date)

        passed_at = self._now_isoformat()
        event = {
            "event_id": self._build_event_id(passed_at, track_id, total_after_event),
            "passed_at": passed_at,
            "track_id": track_id,
            "class_name": class_name,
            "total_after_event": total_after_event,
            "speed_kmh": None if speed_kmh is None else round(speed_kmh, 1),
        }

        events = self.document.setdefault("events", [])
        events.append(event)

        summary = self.document.setdefault("summary", {})
        summary["total"] = total_after_event
        by_class = summary.setdefault("by_class", {})
        by_class[class_name] = int(by_class.get(class_name, 0)) + 1
        self.document["updated_at"] = passed_at

        self._write_document(self.current_date, self.document)

    def load_day(self, day: date) -> dict[str, Any]:
        """Load a daily JSON document, repairing missing fields when possible."""
        file_path = self._file_path(day)
        if not file_path.exists():
            document = self._new_document(day)
            self._write_document(day, document)
            return document

        try:
            with file_path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
            if not isinstance(loaded, dict):
                raise ValueError("Daily count JSON root must be an object.")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            backup_path = file_path.with_suffix(file_path.suffix + f".corrupt-{self._backup_timestamp()}")
            try:
                file_path.replace(backup_path)
                print(f"Fichier de comptage corrompu déplacé: {backup_path}")
            except OSError as backup_exc:
                print(f"Impossible de sauvegarder le fichier de comptage corrompu: {backup_exc}")
            print(f"Réinitialisation du comptage journalier après erreur JSON: {exc}")
            document = self._new_document(day)
            self._write_document(day, document)
            return document

        return self._normalize_document(day, loaded)

    def _normalize_document(self, day: date, document: dict[str, Any]) -> dict[str, Any]:
        now = self._now_isoformat()
        events = document.get("events")
        if not isinstance(events, list):
            events = []
        normalized_events = [event for event in events if isinstance(event, dict)]

        summary = document.get("summary")
        if not isinstance(summary, dict):
            summary = {}

        total = summary.get("total")
        if not isinstance(total, int):
            total = self._derive_total(normalized_events)

        by_class = summary.get("by_class")
        if not isinstance(by_class, dict):
            by_class = self._derive_by_class(normalized_events)
        else:
            by_class = {str(class_name): int(count) for class_name, count in by_class.items() if isinstance(count, int)}

        normalized = {
            "schema_version": int(document.get("schema_version", SCHEMA_VERSION)),
            "date": day.isoformat(),
            "timezone": str(document.get("timezone", self.timezone_name)),
            "created_at": str(document.get("created_at", now)),
            "updated_at": str(document.get("updated_at", now)),
            "summary": {
                "total": total,
                "by_class": by_class,
            },
            "events": normalized_events,
        }
        self._write_document(day, normalized)
        return normalized

    def _new_document(self, day: date) -> dict[str, Any]:
        now = self._now_isoformat()
        return {
            "schema_version": SCHEMA_VERSION,
            "date": day.isoformat(),
            "timezone": self.timezone_name,
            "created_at": now,
            "updated_at": now,
            "summary": {
                "total": 0,
                "by_class": {},
            },
            "events": [],
        }

    def _write_document(self, day: date, document: dict[str, Any]) -> None:
        file_path = self._file_path(day)
        temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")
        payload = json.dumps(document, ensure_ascii=False, indent=2)
        with temporary_path.open("w", encoding="utf-8") as file:
            file.write(payload)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, file_path)

    def _file_path(self, day: date) -> Path:
        return self.data_dir / f"{day.isoformat()}.json"

    def _now_isoformat(self) -> str:
        return datetime.now(self.timezone).isoformat(timespec="microseconds")

    def _backup_timestamp(self) -> str:
        return datetime.now(self.timezone).strftime("%Y%m%d-%H%M%S")

    def _build_event_id(self, passed_at: str, track_id: int | None, total_after_event: int) -> str:
        return f"{passed_at}-track-{track_id or 'none'}-total-{total_after_event}"

    def _derive_total(self, events: list[dict[str, Any]]) -> int:
        totals = [event.get("total_after_event") for event in events if isinstance(event.get("total_after_event"), int)]
        if totals:
            return max(totals)
        return len(events)

    def _derive_by_class(self, events: list[dict[str, Any]]) -> dict[str, int]:
        by_class: dict[str, int] = {}
        for event in events:
            class_name = event.get("class_name")
            if isinstance(class_name, str):
                by_class[class_name] = by_class.get(class_name, 0) + 1
        return by_class

    def _load_timezone(self, timezone_name: str):
        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            print(f"Fuseau horaire introuvable ({timezone_name}), utilisation du fuseau local.")
            return datetime.now().astimezone().tzinfo
