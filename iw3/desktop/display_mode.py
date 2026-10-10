import atexit
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .display_mode_guard import ModeWatchdog


@dataclass(frozen=True)
class Mode:
    width: int
    height: int
    frequency: float

    @classmethod
    def from_native(cls, value: Any) -> "Mode":
        return cls(int(value.width), int(value.height), float(value.frequency or 0))

    def matches(self, other: "Mode") -> bool:
        return (self.width, self.height) == (other.width, other.height) and abs(self.frequency - other.frequency) < 0.1


def choose_fullscreen_mode(current: Mode, available: list[Mode]) -> Mode | None:
    target_width, target_height = current.width * 2, current.height * 2
    exact = [mode for mode in available if (mode.width, mode.height) == (target_width, target_height)]
    if exact:
        return min(exact, key=lambda mode: abs(mode.frequency - current.frequency))

    candidates = [
        mode
        for mode in available
        if mode.width >= current.width
        and mode.height >= current.height
        and (mode.width, mode.height) != (current.width, current.height)
    ]
    proportional = [mode for mode in candidates if mode.width * current.height == mode.height * current.width]

    def preference(mode: Mode) -> tuple[bool, int, float, int]:
        area = mode.width * mode.height
        return (
            mode.width < target_width or mode.height < target_height,
            abs(area - target_width * target_height),
            abs(mode.frequency - current.frequency),
            -area,
        )

    best = min(proportional or candidates, key=preference, default=None)
    return best


def monitor_api() -> ModuleType:
    if sys.platform not in {"win32", "linux"}:
        raise RuntimeError("Fullscreen display mode switching supports Windows and Linux X11 only")
    if sys.platform == "linux" and (
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
        or not os.environ.get("DISPLAY")
        or shutil.which("xrandr") is None
    ):
        raise RuntimeError("Fullscreen display mode switching requires X11 and xrandr; Wayland is unsupported")
    try:
        import pymonctl
    except (ImportError, SystemExit) as exc:
        raise RuntimeError(
            "Cannot load PyMonCtl; install requirements-gui.txt and use a supported desktop session"
        ) from exc
    return pymonctl


def apply_mode(api: ModuleType, monitor: Any, mode: Mode) -> None:
    current = monitor.mode
    if current is not None and mode.matches(Mode.from_native(current)):
        return
    frequency = round(mode.frequency) if sys.platform == "win32" else mode.frequency
    monitor.setMode(api.DisplayMode(mode.width, mode.height, frequency))
    deadline = time.monotonic() + 3.0
    while True:
        actual = monitor.mode
        if actual is not None and mode.matches(Mode.from_native(actual)):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Display {monitor.name} did not accept {mode.width}x{mode.height} at {mode.frequency:g} Hz"
            )
        time.sleep(0.1)


@dataclass(frozen=True)
class DisplaySnapshot:
    monitor_name: str
    center: tuple[int, int]
    original: Mode

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DisplaySnapshot":
        return cls(value["monitor_name"], (value["center"][0], value["center"][1]), Mode(**value["original"]))

    def restore(self) -> None:
        api = monitor_api()
        monitor = api.findMonitorWithName(self.monitor_name)
        if monitor is None:
            raise RuntimeError(f"Cannot restore disconnected display {self.monitor_name}")
        apply_mode(api, monitor, self.original)


class FullscreenDisplayMode:
    """Own one temporary monitor mode and its independent recovery process."""

    def __init__(self) -> None:
        self.snapshot: DisplaySnapshot | None = None
        self.watchdog: ModeWatchdog | None = None
        atexit.register(self.close)

    def activate(self, center: tuple[int, int]) -> bool:
        if self.snapshot is not None:
            return False
        api = monitor_api()
        monitors = api.findMonitorsAtPoint(*center)
        if not monitors or monitors[0].mode is None:
            raise RuntimeError("Cannot determine the Local Viewer monitor or its current mode")
        monitor = monitors[0]
        original = Mode.from_native(monitor.mode)
        target = choose_fullscreen_mode(original, [Mode.from_native(mode) for mode in monitor.allModes])
        if target is None:
            print(f"No larger display mode available for {monitor.name}; keeping the current mode", file=sys.stderr)
            return False

        from .display_mode_guard import ModeWatchdog

        snapshot = DisplaySnapshot(str(monitor.name), center, original)
        watchdog = ModeWatchdog(snapshot)
        self.snapshot, self.watchdog = snapshot, watchdog
        try:
            apply_mode(api, monitor, target)
        except Exception:
            try:
                self.restore()
            except Exception as recovery_error:
                print(f"Display mode rollback failed: {recovery_error}", file=sys.stderr)
                watchdog.release(restored=False)
                self.watchdog = None
            raise
        print(
            f"Local Viewer display mode: {original.width}x{original.height} -> {target.width}x{target.height}",
            file=sys.stderr,
        )
        return True

    def restore(self) -> None:
        if self.snapshot is None:
            return
        self.snapshot.restore()
        self.snapshot = None
        if self.watchdog is not None:
            self.watchdog.release(restored=True)
            self.watchdog = None

    def close(self) -> None:
        try:
            self.restore()
        except Exception as exc:
            print(f"Local Viewer display restoration failed: {exc}; requesting watchdog recovery", file=sys.stderr)
            if self.watchdog is not None:
                self.watchdog.release(restored=False)
                self.watchdog = None
        finally:
            atexit.unregister(self.close)
