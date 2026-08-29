"""Stage 1 — root logging configuration.

The bug being locked out: nothing ever configured the root logger, so every
logger.info() in src/ fell through to logging.lastResort (level WARNING) and was
discarded — including the [EntryV2][ReID] line the entry calibration depends on.
"""
import io
import logging
import unittest
from unittest import mock

from src.logging_setup import (
    DEFAULT_FORMAT,
    configure_logging,
    is_configured,
    _resolve_level,
)


class LoggingSetupTests(unittest.TestCase):
    def setUp(self):
        root = logging.getLogger()
        self._saved_handlers = list(root.handlers)
        self._saved_level = root.level
        root.handlers = []

    def tearDown(self):
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
        root.handlers = list(self._saved_handlers)
        root.setLevel(self._saved_level)

    # -- scenario 1: an info line from src/ actually reaches the stream ---- #
    def test_info_from_a_src_logger_is_emitted(self):
        stream = io.StringIO()
        configure_logging("INFO", stream=stream)

        logging.getLogger("src.entry.coordinator").info("[EntryV2][ReID] score=0.81")

        self.assertIn("[EntryV2][ReID] score=0.81", stream.getvalue())
        self.assertIn("src.entry.coordinator", stream.getvalue())

    def test_without_configuration_an_info_line_is_discarded(self):
        # The pre-fix behaviour, asserted so the regression is visible if the
        # call is ever removed from main().
        self.assertFalse(is_configured())
        record_shown = logging.getLogger("src.entry.coordinator").isEnabledFor(
            logging.INFO
        )
        self.assertFalse(record_shown)

    # -- scenario 2: idempotent, no duplicate lines ------------------------ #
    def test_second_call_does_not_duplicate_the_handler(self):
        stream = io.StringIO()
        configure_logging("INFO", stream=stream)
        configure_logging("INFO")

        self.assertEqual(1, len(logging.getLogger().handlers))

        logging.getLogger("src.demo").info("once")
        self.assertEqual(1, stream.getvalue().count("once"))

    def test_second_call_still_updates_the_level(self):
        stream = io.StringIO()
        configure_logging("INFO", stream=stream)
        configure_logging("DEBUG")

        self.assertEqual(logging.DEBUG, logging.getLogger().level)
        logging.getLogger("src.demo").debug("verbose")
        self.assertIn("verbose", stream.getvalue())

    def test_force_replaces_the_handler_without_stacking(self):
        first = io.StringIO()
        second = io.StringIO()
        configure_logging("INFO", stream=first)
        configure_logging("INFO", stream=second, force=True)

        self.assertEqual(1, len(logging.getLogger().handlers))
        logging.getLogger("src.demo").info("only-second")
        self.assertNotIn("only-second", first.getvalue())
        self.assertIn("only-second", second.getvalue())

    # -- scenario 3: LOG_LEVEL, including a malformed one ------------------ #
    def test_level_comes_from_the_environment(self):
        stream = io.StringIO()
        with mock.patch.dict("os.environ", {"LOG_LEVEL": "WARNING"}, clear=False):
            configure_logging(stream=stream)
        self.assertEqual(logging.WARNING, logging.getLogger().level)

        logging.getLogger("src.demo").info("suppressed")
        self.assertNotIn("suppressed", stream.getvalue())

    def test_malformed_level_falls_back_to_info_rather_than_raising(self):
        # An operator typo must not stop the engine booting.
        self.assertEqual(logging.INFO, _resolve_level("NOT_A_LEVEL"))
        self.assertEqual(logging.INFO, _resolve_level(""))
        self.assertEqual(logging.INFO, _resolve_level(None))

        stream = io.StringIO()
        with mock.patch.dict("os.environ", {"LOG_LEVEL": "chatty"}, clear=False):
            configure_logging(stream=stream)
        self.assertEqual(logging.INFO, logging.getLogger().level)

    def test_numeric_and_lowercase_levels_are_accepted(self):
        self.assertEqual(logging.DEBUG, _resolve_level("debug"))
        self.assertEqual(logging.ERROR, _resolve_level(" error "))

    # -- scenario 4: library noise is capped, uvicorn is untouched --------- #
    def test_noisy_libraries_are_capped_at_warning(self):
        logging.getLogger("urllib3").setLevel(logging.NOTSET)
        configure_logging("INFO", stream=io.StringIO())
        self.assertEqual(logging.WARNING, logging.getLogger("urllib3").level)

    def test_an_explicitly_set_library_level_is_not_overruled(self):
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        try:
            configure_logging("INFO", stream=io.StringIO())
            self.assertEqual(logging.DEBUG, logging.getLogger("httpx").level)
        finally:
            logging.getLogger("httpx").setLevel(logging.NOTSET)

    def test_uvicorn_loggers_are_left_alone(self):
        # uvicorn.run(log_level="info") owns these; clobbering them would change
        # request logging as a side effect of fixing src/ logging.
        configure_logging("INFO", stream=io.StringIO())
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            self.assertEqual(logging.NOTSET, logging.getLogger(name).level)
            self.assertTrue(logging.getLogger(name).propagate)

    # -- format ------------------------------------------------------------ #
    def test_format_carries_level_and_logger_name(self):
        stream = io.StringIO()
        configure_logging("INFO", stream=stream)
        logging.getLogger("src.entry.runtime").warning("disk is full")
        line = stream.getvalue()
        self.assertIn("WARNING", line)
        self.assertIn("src.entry.runtime", line)
        self.assertIn("disk is full", line)
        self.assertIn("%(asctime)s", DEFAULT_FORMAT)



class PerFrameCapTests(unittest.TestCase):
    """Turning the root logger up to INFO switches on 146 logger.info() calls
    at once, and a few sit in the frame loop rather than on an event. On a
    CPU-starved fleet that is paid on every frame of every camera, and it would
    bury the entry decisions the logging was turned on to see."""

    def setUp(self):
        root = logging.getLogger()
        self._saved = list(root.handlers)
        self._level = root.level
        root.handlers = []
        for name in ("src.core.engine.engine_tracking", "src.ocr"):
            logging.getLogger(name).setLevel(logging.NOTSET)

    def tearDown(self):
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
        root.handlers = list(self._saved)
        root.setLevel(self._level)
        for name in ("src.core.engine.engine_tracking", "src.ocr"):
            logging.getLogger(name).setLevel(logging.NOTSET)

    def test_per_frame_loggers_are_capped_at_warning(self):
        configure_logging("INFO", stream=io.StringIO())
        capped = logging.getLogger("src.core.engine.engine_tracking")
        self.assertEqual(logging.WARNING, capped.level)

    def test_the_entry_pipeline_is_not_capped(self):
        # The whole point of configuring logging was to see these.
        stream = io.StringIO()
        configure_logging("INFO", stream=stream)
        logging.getLogger("src.entry.coordinator").info("[EntryV2][ReID] visible")
        self.assertIn("[EntryV2][ReID] visible", stream.getvalue())

    def test_an_explicit_override_beats_the_cap(self):
        stream = io.StringIO()
        with mock.patch.dict(
            "os.environ",
            {"LOG_LEVEL_OVERRIDES": "src.core.engine.engine_tracking=INFO"},
            clear=False,
        ):
            configure_logging("INFO", stream=stream)
        self.assertEqual(
            logging.INFO,
            logging.getLogger("src.core.engine.engine_tracking").level,
        )

    def test_a_malformed_override_does_not_cost_the_others(self):
        with mock.patch.dict(
            "os.environ",
            {"LOG_LEVEL_OVERRIDES": "garbage,,src.ocr=DEBUG,also=NOPE"},
            clear=False,
        ):
            configure_logging("INFO", stream=io.StringIO())
        self.assertEqual(logging.DEBUG, logging.getLogger("src.ocr").level)


if __name__ == "__main__":
    unittest.main()
