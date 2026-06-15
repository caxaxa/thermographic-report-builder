"""Structured logging configuration for CloudWatch.

``setup_logging`` and ``get_logger`` now live in the shared
``solar_report_utils`` library; this module re-exports them so existing
``from ...utils.logger import X`` call sites keep working.
"""

from solar_report_utils import get_logger, setup_logging

__all__ = ["setup_logging", "get_logger"]
