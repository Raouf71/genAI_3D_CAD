"""
Solar Panel Comparison Backend

This package provides functionality for extracting data from solar panel datasheets
and comparing them against design requirements.
"""

from .schemas import (
    PowerRange,
    SolarPanelSpec,
    SolarPanelSchema,
    DetailItem,
    ComparisonDetails,
    ComparisonReportOutput,
)

from .workflows import (
    SolarPanelComparisonWorkflow,
    DatasheetParseEvent,
    RequirementsLoadEvent,
    ComparisonReportEvent,
    LogEvent,
)

from .extraction import (
    create_extraction_agent,
    extract_datasheet,
    aextract_datasheet,
)

from . import config

__version__ = "1.0.0"

__all__ = [
    # Schemas
    "PowerRange",
    "SolarPanelSpec",
    "SolarPanelSchema",
    "DetailItem",
    "ComparisonDetails",
    "ComparisonReportOutput",
    
    # Workflows
    "SolarPanelComparisonWorkflow",
    "DatasheetParseEvent",
    "RequirementsLoadEvent",
    "ComparisonReportEvent",
    "LogEvent",
    
    # Extraction
    "create_extraction_agent",
    "extract_datasheet",
    "aextract_datasheet",
    
    # Config
    "config",
]