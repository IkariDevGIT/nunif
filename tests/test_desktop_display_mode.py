import io
import json
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import psutil

from iw3.desktop import display_mode as modes
from iw3.desktop import display_mode_guard as guard


class ModeSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = modes.Mode(1920, 1080, 60)

    def test_exact_double_prefers_nearest_refresh_rate(self) -> None:
        best = modes.Mode(3840, 2160, 59.94)
        self.assertEqual(modes.choose_fullscreen_mode(self.original, [modes.Mode(3840, 2160, 120), best]), best)

    def test_same_aspect_ratio_before_other_sizes(self) -> None:
        best = modes.Mode(2560, 1440, 60)
        self.assertEqual(modes.choose_fullscreen_mode(self.original, [modes.Mode(4096, 2160, 60), best]), best)

    def test_size_above_target_before_size_below_target(self) -> None:
        best = modes.Mode(7680, 4320, 60)
        self.assertEqual(modes.choose_fullscreen_mode(self.original, [modes.Mode(2560, 1440, 60), best]), best)

    def test_different_aspect_ratio_is_available_as_last_resort(self) -> None:
        best = modes.Mode(4096, 2160, 60)
        self.assertEqual(modes.choose_fullscreen_mode(self.original, [self.original, best]), best)

    def test_neither_dimension_may_shrink(self) -> None:
        self.assertIsNone(modes.choose_fullscreen_mode(self.original, [modes.Mode(3840, 720, 60)]))

    def test_current_size_and_empty_list_do_not_switch(self) -> None:
        for choices in ([], [self.original], [modes.Mode(1920, 1080, 120)]):
            with self.subTest(choices=choices):
                self.assertIsNone(modes.choose_fullscreen_mode(self.original, choices))


class DisplayModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = modes.Mode(1920, 1080, 60)
        self.target = modes.Mode(3840, 2160, 60)
        self.monitor = SimpleNamespace(name="viewer-monitor", mode=self.original, allModes=[self.original, self.target])
        self.monitor.setMode = MagicMock(side_effect=lambda mode: setattr(self.monitor, "mode", mode))
        self.api = MagicMock()
        self.api.DisplayMode = modes.Mode
        self.api.findMonitorsAtPoint = MagicMock(return_value=[self.monitor])
        self.api.findMonitorWithName = MagicMock(return_value=self.monitor)
        self.backend = patch.object(modes, "monitor_api", return_value=self.api)
        self.backend.start()
        self.addCleanup(self.backend.stop)
        self.watchdog = patch.object(guard, "ModeWatchdog")
        self.watchdog_class = self.watchdog.start()
        self.addCleanup(self.watchdog.stop)
        self.controller = modes.FullscreenDisplayMode()
        self.addCleanup(self.controller.close)

    def test_round_trip_and_multiple_activations(self) -> None:
        for _ in range(2):
            self.assertTrue(self.controller.activate((-100, 200)))
            self.assertEqual(self.monitor.mode, self.target)
            self.assertFalse(self.controller.activate((0, 0)))
            self.controller.restore()
            self.assertEqual(self.monitor.mode, self.original)
            self.assertIsNone(self.controller.snapshot)
        self.watchdog_class.return_value.release.assert_called_with(restored=True)
        self.api.findMonitorsAtPoint.assert_called_with(-100, 200)

    def test_watchdog_is_ready_before_mode_change(self) -> None:
        order: list[str] = []

        def ready(snapshot: modes.DisplaySnapshot) -> MagicMock:
            order.append("watchdog")
            return MagicMock()

        def switch(mode: modes.Mode) -> None:
            order.append("switch")
            self.monitor.mode = mode

        self.watchdog_class.side_effect = ready
        self.monitor.setMode.side_effect = switch
        self.controller.activate((0, 0))
        self.assertEqual(order, ["watchdog", "switch"])

    def test_watchdog_startup_failure_does_not_change_mode(self) -> None:
        self.watchdog_class.side_effect = RuntimeError("startup failed")
        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            self.controller.activate((0, 0))
        self.monitor.setMode.assert_not_called()
        self.assertIsNone(self.controller.snapshot)

    def test_failed_switch_rolls_back_even_after_partial_change(self) -> None:
        def change(mode: modes.Mode) -> None:
            self.monitor.mode = mode
            if mode == self.target:
                raise RuntimeError("driver failed after changing mode")

        self.monitor.setMode.side_effect = change
        with self.assertRaisesRegex(RuntimeError, "driver failed"):
            self.controller.activate((0, 0))
        self.assertEqual(self.monitor.mode, self.original)
        self.assertIsNone(self.controller.snapshot)
        self.watchdog_class.return_value.release.assert_called_once_with(restored=True)

    def test_failed_restore_keeps_snapshot_and_guard_for_retry(self) -> None:
        self.controller.activate((0, 0))
        self.monitor.setMode.side_effect = RuntimeError("temporarily unavailable")
        with self.assertRaisesRegex(RuntimeError, "temporarily unavailable"):
            self.controller.restore()
        self.assertIsNotNone(self.controller.snapshot)
        self.watchdog_class.return_value.release.assert_not_called()
        self.monitor.setMode.side_effect = lambda mode: setattr(self.monitor, "mode", mode)
        self.controller.restore()
        self.assertEqual(self.monitor.mode, self.original)

    def test_close_hands_failed_restoration_to_guard(self) -> None:
        self.controller.activate((0, 0))
        self.monitor.setMode.side_effect = RuntimeError("driver unavailable")
        self.controller.close()
        self.watchdog_class.return_value.release.assert_called_once_with(restored=False)
        self.monitor.setMode.side_effect = lambda mode: setattr(self.monitor, "mode", mode)

    def test_disconnected_monitor_does_not_restore_a_different_monitor(self) -> None:
        snapshot = modes.DisplaySnapshot("viewer-monitor", (0, 0), self.original)
        self.api.findMonitorWithName.return_value = None
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            snapshot.restore()
        self.monitor.setMode.assert_not_called()

    def test_silent_driver_rejection_is_detected(self) -> None:
        self.monitor.setMode.side_effect = None
        with patch.object(modes.time, "monotonic", side_effect=[0.0, 4.0]):
            with self.assertRaisesRegex(RuntimeError, "did not accept"):
                modes.apply_mode(self.api, self.monitor, self.target)

    def test_delayed_mode_change_is_allowed(self) -> None:
        monitor = MagicMock()
        type(monitor).mode = PropertyMock(side_effect=[self.original, self.original, self.target])
        with patch.object(modes.time, "monotonic", side_effect=[0.0, 1.0]), patch.object(modes.time, "sleep"):
            modes.apply_mode(self.api, monitor, self.target)
        monitor.setMode.assert_called_once()

    def test_windows_backend_receives_integer_refresh_rate(self) -> None:
        with patch.object(modes.sys, "platform", "win32"):
            modes.apply_mode(self.api, self.monitor, self.target)
        passed = self.monitor.setMode.call_args.args[0]
        self.assertIsInstance(passed.frequency, int)
        self.assertEqual(passed.frequency, 60)

    def test_linux_backend_keeps_fractional_refresh_rate(self) -> None:
        target = modes.Mode(3840, 2160, 59.94)
        with patch.object(modes.sys, "platform", "linux"):
            modes.apply_mode(self.api, self.monitor, target)
        passed = self.monitor.setMode.call_args.args[0]
        self.assertEqual(passed.frequency, 59.94)

    @unittest.skipUnless(sys.platform == "win32", "Windows backend regression")
    def test_real_windows_setter_accepts_the_converted_mode(self) -> None:
        try:
            import pymonctl
        except ImportError:
            self.skipTest("PyMonCtl is not installed")

        native_monitor = MagicMock()
        native_monitor.name = "test-display"

        def native_setter(mode: pymonctl.DisplayMode) -> None:
            pymonctl.Monitor.setMode(native_monitor, mode)
            self.monitor.mode = mode

        self.monitor.setMode.side_effect = native_setter
        with patch(f"{pymonctl.Monitor.__module__}.win32api.ChangeDisplaySettingsEx") as change_display:
            modes.apply_mode(pymonctl, self.monitor, self.target)
            self.assertEqual(change_display.call_args.args[1].DisplayFrequency, 60)
            modes.apply_mode(pymonctl, self.monitor, self.original)
            self.assertEqual(change_display.call_args.args[1].DisplayFrequency, 60)
        self.assertEqual(modes.Mode.from_native(self.monitor.mode), self.original)

    def test_snapshot_json_round_trip(self) -> None:
        snapshot = modes.DisplaySnapshot("viewer-monitor", (-1920, 540), self.original)
        self.assertEqual(modes.DisplaySnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict()))), snapshot)

    def test_no_larger_mode_does_not_start_guard(self) -> None:
        self.monitor.allModes = [self.original]
        self.assertFalse(self.controller.activate((0, 0)))
        self.watchdog_class.assert_not_called()
        self.monitor.setMode.assert_not_called()


class PlatformTests(unittest.TestCase):
    def test_unsupported_sessions_do_not_import_pymonctl(self) -> None:
        for platform, environment, xrandr in (
            ("darwin", {}, "/bin/xrandr"),
            ("linux", {"XDG_SESSION_TYPE": "wayland", "DISPLAY": ":0"}, "/bin/xrandr"),
            ("linux", {"XDG_SESSION_TYPE": "x11"}, "/bin/xrandr"),
            ("linux", {"DISPLAY": ":0"}, None),
        ):
            with (
                self.subTest(platform=platform, environment=environment),
                patch.object(modes.sys, "platform", platform),
                patch.dict(os.environ, environment, clear=True),
                patch.object(modes.shutil, "which", return_value=xrandr),
                patch.dict(sys.modules, {"pymonctl": None}),
            ):
                with self.assertRaises(RuntimeError):
                    modes.monitor_api()

    def test_supported_sessions_load_pymonctl(self) -> None:
        fake = ModuleType("pymonctl")
        for platform in ("win32", "linux"):
            with (
                self.subTest(platform=platform),
                patch.object(modes.sys, "platform", platform),
                patch.dict(os.environ, {"DISPLAY": ":0", "XDG_SESSION_TYPE": "x11"}, clear=True),
                patch.object(modes.shutil, "which", return_value="/bin/xrandr"),
                patch.dict(sys.modules, {"pymonctl": fake}),
            ):
                self.assertIs(modes.monitor_api(), fake)


class WatchdogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot = modes.DisplaySnapshot("viewer-monitor", (0, 0), modes.Mode(1920, 1080, 60))
        self.backend = patch.object(guard, "monitor_api")
        self.backend.start()
        self.addCleanup(self.backend.stop)
        self.restore = patch.object(modes.DisplaySnapshot, "restore")
        self.restore_mock = self.restore.start()
        self.addCleanup(self.restore.stop)

    def run_watchdog(self, commands: str, alive: bool = True) -> int:
        with (
            patch.object(guard.sys, "stdin", io.StringIO(commands)),
            patch.object(guard.sys, "stdout", io.StringIO()),
            patch.object(guard, "parent_alive", return_value=alive),
        ):
            return guard.watch_parent(123, 10.0, self.snapshot)

    def test_clean_completion_does_not_restore_again(self) -> None:
        self.assertEqual(self.run_watchdog("ping\ndone\n"), 0)
        self.restore_mock.assert_not_called()

    def test_closed_pipe_triggers_recovery(self) -> None:
        self.assertEqual(self.run_watchdog("ping\n"), 0)
        self.restore_mock.assert_called_once()

    def test_parent_death_triggers_recovery(self) -> None:
        self.assertEqual(self.run_watchdog("ping\n", alive=False), 0)
        self.restore_mock.assert_called_once()

    def test_recovery_retries_and_reports_exhaustion(self) -> None:
        self.restore_mock.side_effect = RuntimeError("disconnected")
        with patch.object(guard.time, "sleep"):
            self.assertEqual(self.run_watchdog(""), 1)
        self.assertEqual(self.restore_mock.call_count, 3)

    def test_parent_identity_includes_creation_time(self) -> None:
        created = psutil.Process().create_time()
        self.assertTrue(guard.parent_alive(os.getpid(), created))
        self.assertFalse(guard.parent_alive(os.getpid(), created - 10))


class FullscreenTransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from iw3.desktop import local_viewer
        except ImportError:
            self.skipTest("Local Viewer GUI dependencies are not installed")
        self.viewer = local_viewer
        self.window = MagicMock()
        self.window.display_mode = MagicMock()
        self.window.fullscreen_requested = False
        self.window.fullscreen_sequence = 0
        self.window.canvas.closed = False
        self.window.IsFullScreen.return_value = False
        self.window.IsShown.return_value = True
        self.window.GetScreenRect.return_value = SimpleNamespace(x=-1920, y=0, width=960, height=540)
        self.window.set_fullscreen.side_effect = lambda value: self.viewer.LocalViewerWindow.set_fullscreen(
            self.window, value
        )
        self.window.escape_fullscreen.side_effect = lambda: self.viewer.LocalViewerWindow.escape_fullscreen(self.window)
        self.window.toggle_fullscreen.side_effect = lambda: self.viewer.LocalViewerWindow.toggle_fullscreen(self.window)
        self.window.complete_fullscreen_entry.side_effect = lambda sequence: (
            self.viewer.LocalViewerWindow.complete_fullscreen_entry(self.window, sequence)
        )
        self.queue_patch = patch.object(self.viewer.wx, "CallAfter")
        self.queue = self.queue_patch.start()
        self.addCleanup(self.queue_patch.stop)

    def enter(self) -> None:
        self.viewer.LocalViewerWindow.toggle_fullscreen(self.window)

    def dispatch_entry(self, index: int = -1) -> None:
        callback, sequence = self.queue.call_args_list[index].args
        callback(sequence)

    def test_mode_changes_and_confirms_before_fullscreen(self) -> None:
        def activate(center: tuple[int, int]) -> bool:
            self.window.ShowFullScreen.assert_not_called()
            self.assertEqual(center, (-1440, 270))
            return True

        self.window.display_mode.activate.side_effect = activate
        self.enter()
        self.window.ShowFullScreen.assert_not_called()
        self.dispatch_entry()
        self.window.ShowFullScreen.assert_called_once_with(True, style=self.viewer.wx.FULLSCREEN_ALL)

    def test_escape_cancels_queued_fullscreen_entry(self) -> None:
        self.enter()
        event = MagicMock()
        event.GetKeyCode.return_value = self.viewer.wx.WXK_ESCAPE
        self.viewer.LocalViewerWindow.on_char(self.window, event)
        self.dispatch_entry()
        self.window.ShowFullScreen.assert_called_once_with(False, style=self.viewer.wx.FULLSCREEN_ALL)
        self.window.display_mode.restore.assert_called_once()

    def test_close_cancels_queued_fullscreen_entry(self) -> None:
        self.enter()
        self.viewer.LocalViewerWindow.on_close(self.window, MagicMock())
        self.dispatch_entry()
        self.window.ShowFullScreen.assert_not_called()
        self.window.display_mode.close.assert_called_once()
        self.window.canvas.destroy.assert_called_once()

    def test_earlier_entry_cannot_override_a_new_transition(self) -> None:
        self.enter()
        self.window.escape_fullscreen()
        self.window.ShowFullScreen.reset_mock()
        self.enter()
        self.dispatch_entry(0)
        self.window.ShowFullScreen.assert_not_called()
        self.dispatch_entry(1)
        self.window.ShowFullScreen.assert_called_once_with(True, style=self.viewer.wx.FULLSCREEN_ALL)

    def test_leaves_fullscreen_before_restoring_mode(self) -> None:
        order: list[str] = []
        self.window.ShowFullScreen.side_effect = lambda *_args, **_kwargs: order.append("windowed")
        self.window.display_mode.restore.side_effect = lambda: order.append("restore")
        self.window.escape_fullscreen()
        self.assertEqual(order, ["windowed", "restore"])

    def test_feature_disabled_enters_fullscreen_immediately(self) -> None:
        self.window.display_mode = None
        self.enter()
        self.queue.assert_not_called()
        self.window.ShowFullScreen.assert_called_once_with(True, style=self.viewer.wx.FULLSCREEN_ALL)

    def test_no_available_mode_enters_fullscreen_without_delay(self) -> None:
        self.window.display_mode.activate.return_value = False
        self.enter()
        self.queue.assert_not_called()
        self.window.ShowFullScreen.assert_called_once_with(True, style=self.viewer.wx.FULLSCREEN_ALL)

    def test_failed_switch_preserves_ordinary_fullscreen(self) -> None:
        self.window.display_mode.activate.side_effect = RuntimeError("unsupported desktop")
        self.enter()
        self.queue.assert_not_called()
        self.window.ShowFullScreen.assert_called_once_with(True, style=self.viewer.wx.FULLSCREEN_ALL)


if __name__ == "__main__":
    unittest.main()
