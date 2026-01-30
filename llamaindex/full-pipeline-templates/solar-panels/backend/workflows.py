"""
Workflow definitions for solar panel datasheet comparison.
"""

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Context,
    Workflow,
    step,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import ChatPromptTemplate
from llama_cloud_services import LlamaExtract

from .schemas import ComparisonReportOutput


# ============================================================================
# WORKFLOW EVENTS
# ============================================================================

class DatasheetParseEvent(Event):
    """Event triggered when datasheet parsing is complete."""
    datasheet_content: dict


class RequirementsLoadEvent(Event):
    """Event triggered when requirements are loaded."""
    requirements_text: str


class ComparisonReportEvent(Event):
    """Event triggered when comparison report is generated."""
    report: ComparisonReportOutput


class LogEvent(Event):
    """Event for logging messages."""
    msg: str
    delta: bool = False


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

class SolarPanelComparisonWorkflow(Workflow):
    """
    Workflow to extract data from a solar panel datasheet and generate a comparison report
    against provided design requirements.
    
    Steps:
    1. Parse datasheet using LlamaExtract
    2. Load design requirements
    3. Generate comparison report using LLM
    """

    def __init__(
        self, 
        agent: LlamaExtract, 
        requirements_text: str,
        openai_api_key: str,
        **kwargs
    ):
        """
        Initialize the workflow.
        
        Args:
            agent: LlamaExtract agent for datasheet parsing
            requirements_text: Design requirements as text
            openai_api_key: OpenAI API key for LLM
            **kwargs: Additional workflow arguments (verbose, timeout, etc.)
        """
        super().__init__(**kwargs)
        self.agent = agent
        self.requirements_text = requirements_text
        self.openai_api_key = openai_api_key

    @step
    async def parse_datasheet(
        self, ctx: Context, ev: StartEvent
    ) -> DatasheetParseEvent:
        """
        Step 1: Parse the datasheet PDF and extract structured data.
        
        Args:
            ctx: Workflow context
            ev: StartEvent containing datasheet_path
            
        Returns:
            DatasheetParseEvent with extracted content
        """
        datasheet_path = ev.datasheet_path
        
        # Extract data using LlamaExtract agent
        extraction_result = await self.agent.aextract(datasheet_path)
        datasheet_dict = extraction_result.data
        
        # Store in context for later steps
        await ctx.store.set("datasheet_content", datasheet_dict)
        
        # Log progress
        ctx.write_event_to_stream(LogEvent(msg="Datasheet parsed successfully."))
        
        return DatasheetParseEvent(datasheet_content=datasheet_dict)

    @step
    async def load_requirements(
        self, ctx: Context, ev: DatasheetParseEvent
    ) -> RequirementsLoadEvent:
        """
        Step 2: Load design requirements.
        
        Args:
            ctx: Workflow context
            ev: DatasheetParseEvent from previous step
            
        Returns:
            RequirementsLoadEvent with requirements text
        """
        req_text = self.requirements_text
        
        # Log progress
        ctx.write_event_to_stream(LogEvent(msg="Design requirements loaded."))
        
        return RequirementsLoadEvent(requirements_text=req_text)

    @step
    async def generate_comparison_report(
        self, ctx: Context, ev: RequirementsLoadEvent
    ) -> StopEvent:
        """
        Step 3: Generate comparison report using LLM.
        
        Args:
            ctx: Workflow context
            ev: RequirementsLoadEvent from previous step
            
        Returns:
            StopEvent with final report
        """
        # Retrieve datasheet content from context
        datasheet_content = await ctx.store.get("datasheet_content")
        
        # Build prompt for LLM
        prompt_str = """
You are an expert renewable energy engineer.

Compare the following solar panel datasheet information with the design requirements.

Design Requirements:
{requirements_text}

Extracted Datasheet Information:
{datasheet_content}

Generate a detailed comparison report in JSON format with the following schema:
  - component_name: string
  - meets_requirements: boolean
  - summary: string
  - details: dictionary of comparisons for each parameter

For each parameter (Maximum Power, Minimum Power, Max Length, Max Weight, Certification, Efficiency, Temperature Coefficient, Warranty),
indicate PASS or FAIL and provide brief explanations and recommendations.
"""

        prompt = ChatPromptTemplate.from_messages([("user", prompt_str)])
        
        # Initialize LLM
        llm = OpenAI(model="gpt-4o", api_key=self.openai_api_key)
        
        # Generate structured report
        report_output = await llm.astructured_predict(
            ComparisonReportOutput,
            prompt,
            requirements_text=ev.requirements_text,
            datasheet_content=str(datasheet_content),
        )
        
        # Log completion
        ctx.write_event_to_stream(LogEvent(msg="Comparison report generated."))
        
        return StopEvent(
            result={"report": report_output, "datasheet_content": datasheet_content}
        )