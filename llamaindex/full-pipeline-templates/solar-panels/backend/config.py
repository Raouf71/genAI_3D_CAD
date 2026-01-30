"""
Configuration settings for the solar panel comparison backend.
"""

import os
from pathlib import Path

# ============================================================================
# API CONFIGURATION
# ============================================================================

# LlamaCloud API settings
EU_BASE_URL = "https://api.cloud.llamaindex.ai"
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"

# ============================================================================
# EXTRACTION CONFIGURATION
# ============================================================================

# Extraction mode: "BALANCED", "FAST", or "ACCURATE"
EXTRACTION_MODE = "BALANCED"

# Agent name for LlamaExtract
AGENT_NAME = "solar-panel-datasheet"

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base project directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
SOLAR_PANELS_DIR = DATA_DIR / "solar-panels"
REQUIREMENTS_DIR = DATA_DIR / "requirements"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
TEMP_DIR = OUTPUT_DIR / "temp"

# Log directory
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, SOLAR_PANELS_DIR, REQUIREMENTS_DIR, 
                  OUTPUT_DIR, REPORTS_DIR, TEMP_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DEFAULT REQUIREMENTS
# ============================================================================

DEFAULT_REQUIREMENTS = {
    "max_power": 450,  # Watts
    "min_power": 400,  # Watts
    "max_length": 2000,  # mm
    "max_weight": 25,  # kg
    "warranty": 12,  # years
}

# ============================================================================
# WORKFLOW CONFIGURATION
# ============================================================================

# Workflow timeout in seconds
WORKFLOW_TIMEOUT = 180

# Enable verbose logging
WORKFLOW_VERBOSE = True