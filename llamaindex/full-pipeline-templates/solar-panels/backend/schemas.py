"""
Pydantic schemas for solar panel datasheet extraction and comparison reporting.
"""

from pydantic import BaseModel, Field
from typing import List, Literal


# ============================================================================
# DATASHEET EXTRACTION SCHEMAS
# ============================================================================

class PowerRange(BaseModel):
    """Power output range specification."""
    min_power: float = Field(..., description="Minimum power output in Watts")
    max_power: float = Field(..., description="Maximum power output in Watts")
    unit: str = Field("W", description="Power unit")


class SolarPanelSpec(BaseModel):
    """Specification for a single solar panel model."""
    module_name: str = Field(..., description="Name or model of the solar panel module")
    power_output: PowerRange = Field(..., description="Power output range")
    maximum_efficiency: float = Field(
        ..., description="Maximum module efficiency in percentage"
    )
    temperature_coefficient: float = Field(
        ..., description="Temperature coefficient in %/K"
    )
    max_length: int = Field(..., description="Maximum length of product in mm")
    max_weight: int = Field(..., description="Maximum weight of product in kg")
    warranty: int = Field(
        ..., description="Minimum number of years for product to be in warranty"
    )
    certifications: List[str] = Field([], description="List of certifications")
    page_citations: dict = Field(
        ..., description="Mapping of each extracted field to its page numbers"
    )


class SolarPanelSchema(BaseModel):
    """Schema for extracted solar panel specifications from datasheet."""
    specs: List[SolarPanelSpec] = Field(
        ..., description="List of extracted solar panel specifications"
    )


# ============================================================================
# COMPARISON REPORT SCHEMAS
# ============================================================================

class DetailItem(BaseModel):
    """Individual parameter comparison result."""
    status: Literal["PASS", "FAIL"] = Field(..., description="PASS or FAIL")
    explanation: str = Field(..., description="Why it passed or failed")


class ComparisonDetails(BaseModel):
    """Detailed comparison for all parameters."""
    maximum_power: DetailItem
    minimum_power: DetailItem
    max_length: DetailItem
    max_weight: DetailItem
    certification: DetailItem
    efficiency: DetailItem
    temperature_coefficient: DetailItem
    warranty: DetailItem


class ComparisonReportOutput(BaseModel):
    """Final comparison report output."""
    component_name: str
    meets_requirements: bool
    summary: str
    details: ComparisonDetails