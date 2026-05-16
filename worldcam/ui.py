"""On-screen display and keyboard interaction helpers."""

from dataclasses import dataclass, field
import multiprocessing as mp
import queue as queue_module

import cv2
import numpy as np

from worldcam.config import SAHI_ENABLED

MENU_WINDOW_TITLE = "WorldCam - Menu"
MENU_WINDOW_GEOMETRY = "420x520+1320+40"
MENU_EVENT_CLOSE = "close"
MENU_EVENT_CLASS = "class"
MENU_EVENT_POSE = "pose"
MENU_EVENT_SAHI = "sahi"
MENU_EVENT_CLOSED = "closed"

MenuEvent = tuple[str, object]


@dataclass
class MenuState:
    """Mutable state for the YOLO class selection menu."""

    is_open: bool = False
    index: int = 0
    pose_enabled: bool = False
    sahi_enabled: bool = SAHI_ENABLED
    class_selection_changed: bool = False
    pose_toggled: bool = False
    sahi_toggled: bool = False
    menu_process: mp.Process | None = None
    event_queue: mp.Queue = field(default_factory=mp.Queue, repr=False)
    command_queue: mp.Queue = field(default_factory=mp.Queue, repr=False)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Draw the current FPS in yellow at the top-right corner."""
    label = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    margin = 12
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_width, text_height = text_size
    x = max(frame.shape[1] - text_width - margin, margin)
    y = margin + text_height
    cv2.putText(frame, label, (x, y), font, font_scale, (0, 255, 255), thickness)


def _run_class_menu_process(
    class_names: list[str],
    selected_class_names: set[str],
    pose_enabled: bool,
    sahi_enabled: bool,
    event_queue: mp.Queue,
    command_queue: mp.Queue,
) -> None:
    """Run the class selection menu in a separate process with its own Tk main loop."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title(MENU_WINDOW_TITLE)
    root.geometry(MENU_WINDOW_GEOMETRY)
    root.minsize(360, 360)

    selected_local = set(selected_class_names)
    current_index = 0

    style = ttk.Style(root)
    style.configure("WorldCam.TCheckbutton", font=("Segoe UI", 10))

    title_label = ttk.Label(root, text="Menu analyse", font=("Segoe UI", 13, "bold"))
    title_label.pack(anchor="w", padx=10, pady=(10, 2))

    help_label = ttk.Label(root, text="Molette ou flèches: scroller | clic/espace: cocher | M: fermer")
    help_label.pack(anchor="w", padx=10, pady=(0, 8))

    top_frame = ttk.Frame(root)
    top_frame.pack(fill="x", padx=10, pady=(0, 8))

    pose_var = tk.BooleanVar(value=pose_enabled)
    sahi_var = tk.BooleanVar(value=sahi_enabled)

    def toggle_pose() -> None:
        event_queue.put((MENU_EVENT_POSE, bool(pose_var.get())))

    def toggle_sahi() -> None:
        event_queue.put((MENU_EVENT_SAHI, bool(sahi_var.get())))

    ttk.Checkbutton(top_frame, text="Pose", variable=pose_var, command=toggle_pose, style="WorldCam.TCheckbutton").pack(side="left")
    ttk.Checkbutton(top_frame, text="SAHI", variable=sahi_var, command=toggle_sahi, style="WorldCam.TCheckbutton").pack(side="left", padx=(24, 0))

    list_frame = ttk.Frame(root)
    list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    canvas = tk.Canvas(list_frame, highlightthickness=0)
    scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
    rows_frame = ttk.Frame(canvas)
    rows_window = canvas.create_window((0, 0), window=rows_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    class_vars: list[tk.BooleanVar] = []
    class_rows: list[ttk.Checkbutton] = []

    def select_class(class_index: int) -> None:
        nonlocal current_index
        if not 0 <= class_index < len(class_names):
            return
        current_index = class_index
        class_name = class_names[class_index]
        is_selected = bool(class_vars[class_index].get())
        if is_selected:
            selected_local.add(class_name)
        else:
            selected_local.discard(class_name)
        event_queue.put((MENU_EVENT_CLASS, (class_name, is_selected)))

    for class_index, class_name in enumerate(class_names):
        class_var = tk.BooleanVar(value=class_name in selected_local)
        class_vars.append(class_var)
        row = ttk.Checkbutton(
            rows_frame,
            text=class_name,
            variable=class_var,
            command=lambda index=class_index: select_class(index),
            style="WorldCam.TCheckbutton",
        )
        row.pack(anchor="w", fill="x", pady=1)
        class_rows.append(row)

    def refresh_scroll_region(_event=None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def resize_rows_window(event) -> None:
        canvas.itemconfigure(rows_window, width=event.width)

    def scroll_units(units: int) -> None:
        canvas.yview_scroll(units, "units")

    def show_selected_row() -> None:
        if not class_rows:
            return
        selected_row = class_rows[max(0, min(current_index, len(class_rows) - 1))]
        rows_frame.update_idletasks()
        canvas_height = max(canvas.winfo_height(), 1)
        content_height = max(rows_frame.winfo_height(), 1)
        row_top = selected_row.winfo_y()
        row_bottom = row_top + selected_row.winfo_height()
        visible_top = canvas.canvasy(0)
        visible_bottom = visible_top + canvas_height
        if row_top < visible_top:
            canvas.yview_moveto(row_top / content_height)
        elif row_bottom > visible_bottom:
            canvas.yview_moveto(max(0, (row_bottom - canvas_height) / content_height))

    def handle_mousewheel(event) -> str:
        if event.delta:
            scroll_units(-1 if event.delta > 0 else 1)
        return "break"

    def handle_linux_scroll_up(_event) -> str:
        scroll_units(-1)
        return "break"

    def handle_linux_scroll_down(_event) -> str:
        scroll_units(1)
        return "break"

    def handle_key(event) -> str | None:
        nonlocal current_index
        key = event.keysym.lower()
        if key in {"m", "escape"}:
            close_window()
            return "break"
        if key in {"up", "down"}:
            if class_names:
                direction = -1 if key == "up" else 1
                current_index = max(0, min(current_index + direction, len(class_names) - 1))
                show_selected_row()
            return "break"
        if key in {"prior", "page_up"}:
            scroll_units(-8)
            return "break"
        if key in {"next", "page_down"}:
            scroll_units(8)
            return "break"
        if key == "space":
            if 0 <= current_index < len(class_vars):
                class_vars[current_index].set(not class_vars[current_index].get())
                select_class(current_index)
            return "break"
        return None

    def poll_commands() -> None:
        try:
            while True:
                command = command_queue.get_nowait()
                if command == MENU_EVENT_CLOSE:
                    root.destroy()
                    return
        except queue_module.Empty:
            pass
        root.after(50, poll_commands)

    def close_window() -> None:
        event_queue.put((MENU_EVENT_CLOSED, None))
        root.destroy()

    rows_frame.bind("<Configure>", refresh_scroll_region)
    canvas.bind("<Configure>", resize_rows_window)
    root.bind_all("<MouseWheel>", handle_mousewheel)
    root.bind_all("<Button-4>", handle_linux_scroll_up)
    root.bind_all("<Button-5>", handle_linux_scroll_down)
    root.bind_all("<Key>", handle_key)
    root.protocol("WM_DELETE_WINDOW", close_window)

    root.after(100, root.focus_force)
    root.after(50, poll_commands)
    try:
        root.mainloop()
    finally:
        event_queue.put((MENU_EVENT_CLOSED, None))


def start_class_menu_window(class_names: list[str], selected_class_names: set[str], menu_state: MenuState) -> None:
    """Start the independent class menu process if it is not already running."""
    if menu_state.menu_process is not None and menu_state.menu_process.is_alive():
        menu_state.is_open = True
        return

    menu_state.is_open = True
    menu_process = mp.Process(
        target=_run_class_menu_process,
        args=(
            class_names,
            set(selected_class_names),
            menu_state.pose_enabled,
            menu_state.sahi_enabled,
            menu_state.event_queue,
            menu_state.command_queue,
        ),
        daemon=True,
    )
    menu_state.menu_process = menu_process
    menu_process.start()


def close_class_menu_window(menu_state: MenuState) -> None:
    """Request the class menu process to close, then force it if needed."""
    menu_state.is_open = False
    process = menu_state.menu_process
    if process is None:
        return

    if process.is_alive():
        menu_state.command_queue.put(MENU_EVENT_CLOSE)
        process.join(timeout=1.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
    menu_state.menu_process = None


def consume_menu_changes(menu_state: MenuState, selected_class_names: set[str]) -> tuple[bool, bool, bool]:
    """Apply pending menu-process events and return change flags."""
    try:
        while True:
            event_name, payload = menu_state.event_queue.get_nowait()
            if event_name == MENU_EVENT_CLASS:
                class_name, is_selected = payload
                if is_selected:
                    selected_class_names.add(class_name)
                else:
                    selected_class_names.discard(class_name)
                menu_state.class_selection_changed = True
            elif event_name == MENU_EVENT_POSE:
                menu_state.pose_enabled = bool(payload)
                menu_state.pose_toggled = True
            elif event_name == MENU_EVENT_SAHI:
                menu_state.sahi_enabled = bool(payload)
                menu_state.sahi_toggled = True
            elif event_name == MENU_EVENT_CLOSED:
                menu_state.is_open = False
    except queue_module.Empty:
        pass

    if menu_state.menu_process is not None and not menu_state.menu_process.is_alive():
        menu_state.menu_process.join(timeout=0.1)
        menu_state.menu_process = None
        menu_state.is_open = False

    changes = (menu_state.class_selection_changed, menu_state.pose_toggled, menu_state.sahi_toggled)
    menu_state.class_selection_changed = False
    menu_state.pose_toggled = False
    menu_state.sahi_toggled = False
    return changes


def snapshot_menu_state(menu_state: MenuState, selected_class_names: set[str]) -> tuple[set[str], bool, bool]:
    """Return a stable snapshot of selected classes and feature toggles."""
    return set(selected_class_names), menu_state.pose_enabled, menu_state.sahi_enabled


def handle_class_menu_key(
    key: int,
    class_names: list[str],
    selected_class_names: set[str],
    menu_state: MenuState,
) -> tuple[bool, bool, bool]:
    """Update menu state and return class-change, pose-toggle, and SAHI-toggle flags."""
    if key == ord("m"):
        if menu_state.is_open:
            close_class_menu_window(menu_state)
        else:
            start_class_menu_window(class_names, selected_class_names, menu_state)
        return False, False, False

    if not menu_state.is_open:
        return False, False, False

    if key == ord("p"):
        menu_state.pose_enabled = not menu_state.pose_enabled
        return False, True, False

    if key == ord("a"):
        menu_state.sahi_enabled = not menu_state.sahi_enabled
        return False, False, True

    return False, False, False
