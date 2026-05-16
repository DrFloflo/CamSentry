"""Constants and event names for the WorldCam menu."""

MENU_WINDOW_TITLE = "WorldCam - Menu"
MENU_WINDOW_GEOMETRY = "420x520+1320+40"
MENU_EVENT_CLOSE = "close"
MENU_EVENT_CLASS = "class"
MENU_EVENT_POSE = "pose"
MENU_EVENT_SAHI = "sahi"
MENU_EVENT_THRESHOLD = "threshold"
MENU_EVENT_CLOSED = "closed"

MenuEvent = tuple[str, object]
