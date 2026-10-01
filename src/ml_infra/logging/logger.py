import json
import logging as std_logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

from ml_infra.results_db import ResultsDatabase

LogLevel = Literal[
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]


@dataclass(frozen=True)
class LogEvent:
    """
    Immutable structured operational event.
    """

    timestamp: str

    level: LogLevel

    logger_name: str

    event: str

    message: str

    run_id: str | None = None

    evaluation_id: str | None = None

    data: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the event into a JSON-compatible dictionary.
        """

        return asdict(self)

    def to_json(self) -> str:
        """
        Serialize the event deterministically.
        """

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )


class ResearchLogger:
    """
    Structured logger shared by all ml_infra components.

    Events can be persisted to the results database and
    optionally written as JSON Lines.

    The standard Python logging system is used for console output.
    """

    _LEVELS: ClassVar[dict[str, int]] = {
        "DEBUG": std_logging.DEBUG,
        "INFO": std_logging.INFO,
        "WARNING": std_logging.WARNING,
        "ERROR": std_logging.ERROR,
        "CRITICAL": std_logging.CRITICAL,
    }

    def __init__(
        self,
        db: ResultsDatabase,
        *,
        name: str = "ml_infra",
        level: LogLevel = "INFO",
        jsonl_path: str | Path | None = None,
        run_id: str | None = None,
        evaluation_id: str | None = None,
    ):
        self.db = db
        self.name = name
        self.level = level
        self.jsonl_path = (
            Path(jsonl_path)
            if jsonl_path is not None
            else None
        )

        self.run_id = run_id
        self.evaluation_id = evaluation_id

        if self.level not in self._LEVELS:
            raise ValueError(
                f"Unsupported log level: {self.level}"
            )

        self._logger = (
            std_logging.getLogger(name)
        )

        self._logger.setLevel(
            self._LEVELS[level]
        )

        self._configure_console()

        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

    def _configure_console(self) -> None:
        """
        Install a simple console handler if this logger
        does not already have one.
        """

        if self._logger.handlers:
            return

        handler = (
            std_logging.StreamHandler()
        )

        formatter = (
            std_logging.Formatter(
                "%(asctime)s | "
                "%(levelname)s | "
                "%(name)s | "
                "%(message)s"
            )
        )

        handler.setFormatter(formatter)

        self._logger.addHandler(
            handler
        )

        self._logger.propagate = False

    @staticmethod
    def _utc_now() -> str:
        """Return current UTC timestamp."""

        return datetime.now(
            timezone.utc
        ).isoformat()

    def bind(
        self,
        *,
        run_id: str | None = None,
        evaluation_id: str | None = None,
    ) -> "ResearchLogger":
        """
        Return a new logger carrying additional context.

        The original logger remains unchanged.
        """

        return ResearchLogger(
            self.db,
            name=self.name,
            level=self.level,
            jsonl_path=self.jsonl_path,
            run_id=(
                run_id
                if run_id is not None
                else self.run_id
            ),
            evaluation_id=(
                evaluation_id
                if evaluation_id is not None
                else self.evaluation_id
            ),
        )

    def _should_emit(
        self,
        level: LogLevel,
    ) -> bool:
        return (
            self._LEVELS[level]
            >= self._LEVELS[self.level]
        )

    def log(
        self,
        level: LogLevel,
        event: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
        run_id: str | None = None,
        evaluation_id: str | None = None,
    ) -> LogEvent:
        """
        Emit one structured event.

        Explicit IDs take precedence over bound context.
        """

        if level not in self._LEVELS:
            raise ValueError(
                f"Unsupported log level: {level}"
            )

        if not event.strip():
            raise ValueError(
                "event cannot be empty."
            )

        if not message.strip():
            raise ValueError(
                "message cannot be empty."
            )

        if not self._should_emit(level):
            # Return an event without persisting it only
            # so callers still receive a valid object.
            return LogEvent(
                timestamp=self._utc_now(),
                level=level,
                logger_name=self.name,
                event=event,
                message=message,
                run_id=(
                    run_id
                    if run_id is not None
                    else self.run_id
                ),
                evaluation_id=(
                    evaluation_id
                    if evaluation_id is not None
                    else self.evaluation_id
                ),
                data=data or {},
            )

        log_event = LogEvent(
            timestamp=self._utc_now(),
            level=level,
            logger_name=self.name,
            event=event,
            message=message,
            run_id=(
                run_id
                if run_id is not None
                else self.run_id
            ),
            evaluation_id=(
                evaluation_id
                if evaluation_id is not None
                else self.evaluation_id
            ),
            data=data or {},
        )

        # Persistent structured record.
        self.db.log_event(
            timestamp=log_event.timestamp,
            level=log_event.level,
            logger_name=log_event.logger_name,
            event=log_event.event,
            message=log_event.message,
            run_id=log_event.run_id,
            evaluation_id=log_event.evaluation_id,
            data=log_event.data,
        )

        # Optional JSONL event stream.
        if self.jsonl_path is not None:
            with self.jsonl_path.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    log_event.to_json()
                    + "\n"
                )

        # Human-readable console log.
        numeric_level = self._LEVELS[
            level
        ]

        console_message = message

        context = []

        if log_event.run_id is not None:
            context.append(
                f"run_id={log_event.run_id}"
            )

        if (
            log_event.evaluation_id
            is not None
        ):
            context.append(
                "evaluation_id="
                f"{log_event.evaluation_id}"
            )

        if context:
            console_message = (
                f"[{event}] "
                f"{message} "
                f"({' '.join(context)})"
            )
        else:
            console_message = (
                f"[{event}] {message}"
            )

        self._logger.log(
            numeric_level,
            console_message,
        )

        return log_event

    def debug(
        self,
        event: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> LogEvent:
        return self.log(
            "DEBUG",
            event,
            message,
            data=data,
        )

    def info(
        self,
        event: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> LogEvent:
        return self.log(
            "INFO",
            event,
            message,
            data=data,
        )

    def warning(
        self,
        event: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> LogEvent:
        return self.log(
            "WARNING",
            event,
            message,
            data=data,
        )

    def error(
        self,
        event: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> LogEvent:
        return self.log(
            "ERROR",
            event,
            message,
            data=data,
        )

    def critical(
        self,
        event: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> LogEvent:
        return self.log(
            "CRITICAL",
            event,
            message,
            data=data,
        )