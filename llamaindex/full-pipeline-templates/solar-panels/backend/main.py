"""
Main entry point for the solar panel comparison backend.

This script demonstrates how to use the modularized backend components.
"""

import os
import asyncio
from getpass import getpass
from pathlib import Path

import nest_asyncio

from backend.extraction import create_extraction_agent
from backend.workflows import SolarPanelComparisonWorkflow
from backend.config import (
    WORKFLOW_TIMEOUT,
    WORKFLOW_VERBOSE,
    DEFAULT_REQUIREMENTS,
)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()


def setup_api_keys():
    """
    Setup API keys from environment or user input.
    
    Returns:
        Tuple of (llama_api_key, openai_api_key)
    """
    # LlamaCloud API Key
    if "LLAMA_CLOUD_API_KEY" not in os.environ:
        llama_key = getpass("Enter your Llama Cloud API Key: ")
        os.environ["LLAMA_CLOUD_API_KEY"] = llama_key
    else:
        llama_key = os.environ["LLAMA_CLOUD_API_KEY"]
    
    # OpenAI API Key
    if "OPENAI_API_KEY" not in os.environ:
        openai_key = getpass("Enter your OpenAI API Key: ")
    else:
        openai_key = os.environ["OPENAI_API_KEY"]
    
    return llama_key, openai_key


def create_requirements_text(
    max_power: int = None,
    min_power: int = None,
    max_length: int = None,
    max_weight: int = None,
    warranty: int = None
) -> str:
    """
    Create requirements text from parameters.
    
    Args:
        max_power: Maximum power in Watts
        min_power: Minimum power in Watts
        max_length: Maximum length in mm
        max_weight: Maximum weight in kg
        warranty: Minimum warranty in years
        
    Returns:
        Formatted requirements text
    """
    # Use defaults if not provided
    max_power = max_power or DEFAULT_REQUIREMENTS["max_power"]
    min_power = min_power or DEFAULT_REQUIREMENTS["min_power"]
    max_length = max_length or DEFAULT_REQUIREMENTS["max_length"]
    max_weight = max_weight or DEFAULT_REQUIREMENTS["max_weight"]
    warranty = warranty or DEFAULT_REQUIREMENTS["warranty"]
    
    return f"""
Solar Panel Design Requirements:

1. Power Output:
   - Maximum Power: {max_power} W
   - Minimum Power: {min_power} W

2. Physical Specifications:
   - Maximum Length: {max_length} mm
   - Maximum Weight: {max_weight} kg

3. Warranty:
   - Minimum Warranty Period: {warranty} years

4. Additional Requirements:
   - Must have relevant certifications
   - High efficiency preferred
   - Good temperature coefficient preferred
"""


async def run_comparison(
    datasheet_path: str,
    requirements_text: str = None,
    max_power: int = None,
    min_power: int = None,
    max_length: int = None,
    max_weight: int = None,
    warranty: int = None,
    verbose: bool = WORKFLOW_VERBOSE,
    timeout: int = WORKFLOW_TIMEOUT
) -> dict:
    """
    Run the complete comparison workflow.
    
    Args:
        datasheet_path: Path to the PDF datasheet
        requirements_text: Pre-formatted requirements text (optional)
        max_power: Maximum power in Watts (used if requirements_text not provided)
        min_power: Minimum power in Watts
        max_length: Maximum length in mm
        max_weight: Maximum weight in kg
        warranty: Minimum warranty in years
        verbose: Enable verbose logging
        timeout: Workflow timeout in seconds
        
    Returns:
        Dictionary with report and datasheet_content
    """
    # Setup API keys
    llama_key, openai_key = setup_api_keys()
    
    # Create extraction agent
    print("\n📊 Creating extraction agent...")
    agent = create_extraction_agent()
    
    # Create requirements text if not provided
    if requirements_text is None:
        requirements_text = create_requirements_text(
            max_power=max_power,
            min_power=min_power,
            max_length=max_length,
            max_weight=max_weight,
            warranty=warranty
        )
    
    # Initialize workflow
    print("\n🔄 Initializing workflow...")
    workflow = SolarPanelComparisonWorkflow(
        agent=agent,
        requirements_text=requirements_text,
        openai_api_key=openai_key,
        verbose=verbose,
        timeout=timeout
    )
    
    # Run workflow
    print(f"\n🚀 Processing datasheet: {Path(datasheet_path).name}")
    result = await workflow.run(datasheet_path=datasheet_path)
    
    return result


def main():
    """
    Main function demonstrating backend usage.
    """
    print("=" * 70)
    print("🌞 Solar Panel Comparison Backend")
    print("=" * 70)
    
    # Example datasheet path
    # Modify this to point to your actual datasheet
    datasheet_path = "data/solar-panels/EU_Datasheet_HoneyM_DE08M.08(II)_2021_A.pdf"
    
    # Check if file exists
    if not Path(datasheet_path).exists():
        print(f"\n❌ Error: Datasheet not found at {datasheet_path}")
        print("Please update the datasheet_path variable with a valid PDF path.")
        return
    
    # Run comparison
    result = asyncio.run(
        run_comparison(
            datasheet_path=datasheet_path,
            max_power=450,
            min_power=400,
            max_length=2000,
            max_weight=25,
            warranty=12
        )
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 COMPARISON REPORT")
    print("=" * 70)
    print(result["report"].model_dump_json(indent=4))
    
    print("\n" + "=" * 70)
    print("✅ Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()