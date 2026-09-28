import logging
import os
import sys

import structlog
from structlog.types import Processor

LEVEL = os.environ.get("CHESSGPT_LOG_LEVEL", "INFO").upper()
FORMAT = os.environ.get("CHESSGPT_LOG_FORMAT", "console")


def configure_logging() -> None:
    level = logging.getLevelNamesMapping()[LEVEL]

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console_renderer = structlog.dev.ConsoleRenderer() if FORMAT == "console" else structlog.processors.JSONRenderer()
    console = logging.StreamHandler(sys.stdout)

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, console_renderer],
    )
    console.setFormatter(formatter)
    root.addHandler(console)
