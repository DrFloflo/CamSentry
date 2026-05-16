"""Tkinter process implementation for the class selection menu."""

import multiprocessing as mp
import queue as queue_module

from worldcam.menu.constants import (
    MENU_EVENT_CLASS,
    MENU_EVENT_CLOSE,
    MENU_EVENT_CLOSED,
    MENU_EVENT_POSE,
    MENU_EVENT_SAHI,
    MENU_EVENT_TRACKING,
    MENU_EVENT_SEGMENTATION,
    MENU_EVENT_THRESHOLD,
    MENU_WINDOW_GEOMETRY,
    MENU_WINDOW_TITLE,
)


def run_class_menu_process(
    class_names: list[str],
    selected_class_names: set[str],
    pose_enabled: bool,
    sahi_enabled: bool,
    tracking_enabled: bool,
    segmentation_enabled: bool,
    display_threshold: float,
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
    tracking_var = tk.BooleanVar(value=tracking_enabled)
    segmentation_var = tk.BooleanVar(value=segmentation_enabled)
    threshold_var = tk.DoubleVar(value=display_threshold)
    threshold_label_var = tk.StringVar(value=f"Threshold: {display_threshold:.2f}")

    def toggle_pose() -> None:
        event_queue.put((MENU_EVENT_POSE, bool(pose_var.get())))

    def toggle_sahi() -> None:
        event_queue.put((MENU_EVENT_SAHI, bool(sahi_var.get())))

    def toggle_tracking() -> None:
        event_queue.put((MENU_EVENT_TRACKING, bool(tracking_var.get())))

    def toggle_segmentation() -> None:
        event_queue.put((MENU_EVENT_SEGMENTATION, bool(segmentation_var.get())))

    def update_threshold(value: str) -> None:
        threshold = round(float(value), 2)
        threshold_label_var.set(f"Threshold: {threshold:.2f}")
        event_queue.put((MENU_EVENT_THRESHOLD, threshold))

    ttk.Checkbutton(top_frame, text="Pose", variable=pose_var, command=toggle_pose, style="WorldCam.TCheckbutton").pack(side="left")
    ttk.Checkbutton(top_frame, text="SAHI", variable=sahi_var, command=toggle_sahi, style="WorldCam.TCheckbutton").pack(side="left", padx=(24, 0))
    ttk.Checkbutton(top_frame, text="Tracking", variable=tracking_var, command=toggle_tracking, style="WorldCam.TCheckbutton").pack(side="left", padx=(24, 0))
    ttk.Checkbutton(top_frame, text="Seg", variable=segmentation_var, command=toggle_segmentation, style="WorldCam.TCheckbutton").pack(side="left", padx=(24, 0))

    threshold_frame = ttk.Frame(root)
    threshold_frame.pack(fill="x", padx=10, pady=(0, 8))
    ttk.Label(threshold_frame, textvariable=threshold_label_var).pack(anchor="w")
    ttk.Scale(
        threshold_frame,
        from_=0.0,
        to=1.0,
        variable=threshold_var,
        command=update_threshold,
    ).pack(fill="x")

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
        if key == "t":
            tracking_var.set(not tracking_var.get())
            toggle_tracking()
            return "break"
        if key == "g":
            segmentation_var.set(not segmentation_var.get())
            toggle_segmentation()
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
