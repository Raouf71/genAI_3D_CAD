"""
Data extraction utilities using LlamaExtract.
"""

from llama_cloud_services import LlamaExtract, EU_BASE_URL
from llama_cloud.core.api_error import ApiError
from llama_cloud import ExtractConfig

from .schemas import SolarPanelSchema
from .config import AGENT_NAME, EXTRACTION_MODE


def create_extraction_agent(
    base_url: str = EU_BASE_URL,
    agent_name: str = AGENT_NAME,
    extraction_mode: str = EXTRACTION_MODE,
    recreate: bool = True
) -> LlamaExtract:
    """
    Create or retrieve a LlamaExtract agent for solar panel datasheet extraction.
    
    Args:
        base_url: LlamaCloud API base URL
        agent_name: Name for the extraction agent
        extraction_mode: Extraction mode ("BALANCED", "FAST", or "ACCURATE")
        recreate: If True, delete existing agent and create new one
        
    Returns:
        Configured LlamaExtract agent
        
    Raises:
        ApiError: If agent creation fails
    """
    # Initialize LlamaExtract client
    llama_extract = LlamaExtract(base_url=base_url)
    
    # Handle existing agent
    if recreate:
        try:
            existing_agent = llama_extract.get_agent(name=agent_name)
            if existing_agent:
                llama_extract.delete_agent(existing_agent.id)
                print(f"Deleted existing agent: {agent_name}")
        except ApiError as e:
            if e.status_code != 404:
                raise
    
    # Create extraction configuration
    extract_config = ExtractConfig(
        extraction_mode=extraction_mode,
    )
    
    # Create new agent
    agent = llama_extract.create_agent(
        name=agent_name,
        data_schema=SolarPanelSchema,
        config=extract_config
    )
    
    print(f"Created extraction agent: {agent_name}")
    
    return agent


def extract_datasheet(agent: LlamaExtract, datasheet_path: str) -> dict:
    """
    Synchronously extract data from a solar panel datasheet.
    
    Args:
        agent: Configured LlamaExtract agent
        datasheet_path: Path to the PDF datasheet
        
    Returns:
        Extracted data as dictionary
        
    Raises:
        Exception: If extraction fails
    """
    try:
        extraction_result = agent.extract(datasheet_path)
        return extraction_result.data
    except Exception as e:
        print(f"Extraction failed for {datasheet_path}: {str(e)}")
        raise


async def aextract_datasheet(agent: LlamaExtract, datasheet_path: str) -> dict:
    """
    Asynchronously extract data from a solar panel datasheet.
    
    Args:
        agent: Configured LlamaExtract agent
        datasheet_path: Path to the PDF datasheet
        
    Returns:
        Extracted data as dictionary
        
    Raises:
        Exception: If extraction fails
    """
    try:
        extraction_result = await agent.aextract(datasheet_path)
        return extraction_result.data
    except Exception as e:
        print(f"Extraction failed for {datasheet_path}: {str(e)}")
        raise