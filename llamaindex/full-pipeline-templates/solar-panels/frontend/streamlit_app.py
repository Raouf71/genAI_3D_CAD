import streamlit as st
import os
import json
import asyncio
from pathlib import Path
import tempfile
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="Solar Panel Expert",
    page_icon="☀️",
    layout="wide"
)

# Import your backend modules
# Note: These imports should work if your backend code is in the same directory
# or properly installed as a package
try:
    from llama_cloud_services import LlamaExtract, EU_BASE_URL
    from llama_cloud.core.api_error import ApiError
    from llama_cloud import ExtractConfig
    from llama_index.llms.openai import OpenAI
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    st.warning("⚠️ Backend dependencies not fully installed. Some features may not work.")

# Import your workflow classes (assuming they're in a separate file or same directory)
# If your backend code is in a file called 'backend.py', uncomment and adjust:
# from backend import SolarPanelComparisonWorkflow, SolarPanelSchema, ComparisonReportOutput

# Initialize session state
if 'reports' not in st.session_state:
    st.session_state.reports = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'llama_api_key' not in st.session_state:
    st.session_state.llama_api_key = ""
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'max_power' not in st.session_state:
    st.session_state.max_power = 450
if 'min_power' not in st.session_state:
    st.session_state.min_power = 400
if 'max_length' not in st.session_state:
    st.session_state.max_length = 2000
if 'max_weight' not in st.session_state:
    st.session_state.max_weight = 25
if 'warranty' not in st.session_state:
    st.session_state.warranty = 12

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        display: flex;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .header-icon {
        font-size: 3rem;
        margin-right: 1rem;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF8C00;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
        margin: 10px 0;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        color: #721c24;
        margin: 10px 0;
    }
    .report-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        max-height: 600px;
        overflow-y: auto;
    }
    .section-divider {
        border-top: 2px solid #e9ecef;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <div class="header-icon">☀️</div>
        <div class="header-title">Your Solar Panel Expert</div>
    </div>
""", unsafe_allow_html=True)

# Setup Section - Tabs
st.markdown("### ⚙️ Setup & Configuration")

tab1, tab2, tab3 = st.tabs(["📤 Upload Docs", "🔑 Setup API Keys", "⚡ Configure Parameters"])

# Tab 1: Upload Documents
with tab1:
    st.markdown("#### Upload Solar Panel Datasheets")
    st.markdown("Upload PDF datasheets for solar panels you want to analyze.")
    
    uploaded_files = st.file_uploader(
        "Drag and drop files here or click to browse",
        type=['pdf'],
        accept_multiple_files=True,
        key='file_uploader'
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.markdown(f'<div class="status-success">✅ {len(uploaded_files)} document(s) uploaded successfully and ready to use.</div>', 
                   unsafe_allow_html=True)
        
        # Display uploaded file names
        st.markdown("**Uploaded files:**")
        for file in uploaded_files:
            st.markdown(f"- {file.name} ({file.size / 1024:.2f} KB)")
    else:
        if st.session_state.uploaded_files:
            st.info(f"ℹ️ {len(st.session_state.uploaded_files)} document(s) previously uploaded.")
        else:
            st.info("ℹ️ No documents uploaded yet. Please upload PDF datasheets to proceed.")

# Tab 2: API Keys
with tab2:
    st.markdown("#### Configure API Keys")
    st.markdown("Enter your API keys to enable datasheet extraction and comparison analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        llama_key = st.text_input(
            "🦙 LlamaExtract API Key",
            type="password",
            value=st.session_state.llama_api_key,
            help="Required for extracting structured data from PDF datasheets"
        )
        if llama_key:
            st.session_state.llama_api_key = llama_key
            os.environ["LLAMA_CLOUD_API_KEY"] = llama_key
    
    with col2:
        openai_key = st.text_input(
            "🤖 OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
            help="Required for generating comparison reports using GPT-4"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key
    
    # Status indicator
    if st.session_state.llama_api_key and st.session_state.openai_api_key:
        st.markdown('<div class="status-success">✅ All API keys configured successfully.</div>', 
                   unsafe_allow_html=True)
    elif st.session_state.llama_api_key or st.session_state.openai_api_key:
        st.markdown('<div class="status-error">⚠️ Please configure all required API keys.</div>', 
                   unsafe_allow_html=True)
    else:
        st.info("ℹ️ API keys not configured yet.")

# Tab 3: Configure Parameters
with tab3:
    st.markdown("#### Design Requirements")
    st.markdown("Set the requirements for your solar panel specifications.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.max_power = st.slider(
            "⚡ Maximum Power (W)",
            min_value=100,
            max_value=1000,
            value=st.session_state.max_power,
            step=10,
            help="Maximum acceptable power output in Watts"
        )
        
        st.session_state.min_power = st.slider(
            "🔋 Minimum Power (W)",
            min_value=100,
            max_value=1000,
            value=st.session_state.min_power,
            step=10,
            help="Minimum required power output in Watts"
        )
        
        st.session_state.max_length = st.slider(
            "📏 Maximum Length (mm)",
            min_value=500,
            max_value=3000,
            value=st.session_state.max_length,
            step=50,
            help="Maximum acceptable length in millimeters"
        )
    
    with col2:
        st.session_state.max_weight = st.slider(
            "⚖️ Maximum Weight (kg)",
            min_value=5,
            max_value=50,
            value=st.session_state.max_weight,
            step=1,
            help="Maximum acceptable weight in kilograms"
        )
        
        st.session_state.warranty = st.slider(
            "🛡️ Warranty (years)",
            min_value=1,
            max_value=30,
            value=st.session_state.warranty,
            step=1,
            help="Minimum required warranty period in years"
        )
    
    st.markdown('<div class="status-success">✅ Parameters configured successfully.</div>', 
               unsafe_allow_html=True)
    
    # Show current configuration
    with st.expander("📋 View Current Configuration"):
        config_data = {
            "Maximum Power": f"{st.session_state.max_power} W",
            "Minimum Power": f"{st.session_state.min_power} W",
            "Maximum Length": f"{st.session_state.max_length} mm",
            "Maximum Weight": f"{st.session_state.max_weight} kg",
            "Warranty": f"{st.session_state.warranty} years"
        }
        st.json(config_data)

# Section Divider
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Action Button
st.markdown("### 🚀 Generate Comparison Report")

col1, col2, col3 = st.columns([2, 1, 2])

with col2:
    generate_button = st.button(
        "🔍 Analyze & Compare",
        type="primary",
        use_container_width=True
    )

# Generate Report Logic
if generate_button:
    # Validation
    if not st.session_state.uploaded_files:
        st.error("❌ Please upload at least one datasheet PDF.")
    elif not st.session_state.llama_api_key:
        st.error("❌ Please configure your LlamaExtract API key.")
    elif not st.session_state.openai_api_key:
        st.error("❌ Please configure your OpenAI API key.")
    elif not IMPORTS_AVAILABLE:
        st.error("❌ Backend dependencies are not installed. Please install required packages.")
    else:
        # Process each uploaded file
        with st.spinner("🔄 Processing datasheets and generating comparison reports..."):
            try:
                # Import backend components here to avoid issues if not installed
                from pydantic import BaseModel, Field
                from typing import List, Literal
                from llama_index.core.workflow import (
                    Event, StartEvent, StopEvent, Context, Workflow, step
                )
                from llama_index.llms.openai import OpenAI
                from llama_index.core.prompts import ChatPromptTemplate
                
                # Define schemas (copying from your backend)
                class PowerRange(BaseModel):
                    min_power: float = Field(..., description="Minimum power output in Watts")
                    max_power: float = Field(..., description="Maximum power output in Watts")
                    unit: str = Field("W", description="Power unit")
                
                class SolarPanelSpec(BaseModel):
                    module_name: str = Field(..., description="Name or model of the solar panel module")
                    power_output: PowerRange = Field(..., description="Power output range")
                    maximum_efficiency: float = Field(..., description="Maximum module efficiency in percentage")
                    temperature_coefficient: float = Field(..., description="Temperature coefficient in %/K")
                    max_length: int = Field(..., description="Maximum length of product in mm")
                    max_weight: int = Field(..., description="Maximum weight of product in kg")
                    warranty: int = Field(..., description="Minimum number of years for product to be in warranty")
                    certifications: List[str] = Field([], description="List of certifications")
                    page_citations: dict = Field(..., description="Mapping of each extracted field to its page numbers")
                
                class SolarPanelSchema(BaseModel):
                    specs: List[SolarPanelSpec] = Field(..., description="List of extracted solar panel specifications")
                
                class DetailItem(BaseModel):
                    status: Literal["PASS", "FAIL"] = Field(..., description="PASS or FAIL")
                    explanation: str = Field(..., description="Why it passed or failed")
                
                class ComparisonDetails(BaseModel):
                    maximum_power: DetailItem
                    minimum_power: DetailItem
                    max_length: DetailItem
                    max_weight: DetailItem
                    certification: DetailItem
                    efficiency: DetailItem
                    temperature_coefficient: DetailItem
                    warranty: DetailItem
                
                class ComparisonReportOutput(BaseModel):
                    component_name: str
                    meets_requirements: bool
                    summary: str
                    details: ComparisonDetails
                
                # Initialize LlamaExtract
                llama_extract = LlamaExtract(base_url=EU_BASE_URL)
                
                # Create or get agent
                try:
                    existing_agent = llama_extract.get_agent(name="solar-panel-datasheet")
                    if existing_agent:
                        llama_extract.delete_agent(existing_agent.id)
                except ApiError as e:
                    if e.status_code != 404:
                        raise
                
                extract_config = ExtractConfig(extraction_mode="BALANCED")
                agent = llama_extract.create_agent(
                    name="solar-panel-datasheet",
                    data_schema=SolarPanelSchema,
                    config=extract_config
                )
                
                # Create requirements text from parameters
                requirements_text = f"""
Solar Panel Design Requirements:

1. Power Output:
   - Maximum Power: {st.session_state.max_power} W
   - Minimum Power: {st.session_state.min_power} W

2. Physical Specifications:
   - Maximum Length: {st.session_state.max_length} mm
   - Maximum Weight: {st.session_state.max_weight} kg

3. Warranty:
   - Minimum Warranty Period: {st.session_state.warranty} years

4. Additional Requirements:
   - Must have relevant certifications
   - High efficiency preferred
   - Good temperature coefficient preferred
"""
                
                # Define workflow (simplified inline version)
                class DatasheetParseEvent(Event):
                    datasheet_content: dict
                
                class RequirementsLoadEvent(Event):
                    requirements_text: str
                
                class SolarPanelComparisonWorkflow(Workflow):
                    def __init__(self, agent: LlamaExtract, requirements_text: str, **kwargs):
                        super().__init__(**kwargs)
                        self.agent = agent
                        self.requirements_text = requirements_text
                    
                    @step
                    async def parse_datasheet(self, ctx: Context, ev: StartEvent) -> DatasheetParseEvent:
                        datasheet_path = ev.datasheet_path
                        extraction_result = await self.agent.aextract(datasheet_path)
                        datasheet_dict = extraction_result.data
                        await ctx.store.set("datasheet_content", datasheet_dict)
                        return DatasheetParseEvent(datasheet_content=datasheet_dict)
                    
                    @step
                    async def load_requirements(self, ctx: Context, ev: DatasheetParseEvent) -> RequirementsLoadEvent:
                        return RequirementsLoadEvent(requirements_text=self.requirements_text)
                    
                    @step
                    async def generate_comparison_report(self, ctx: Context, ev: RequirementsLoadEvent) -> StopEvent:
                        datasheet_content = await ctx.store.get("datasheet_content")
                        
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
                        
                        llm = OpenAI(model="gpt-4o", api_key=st.session_state.openai_api_key)
                        
                        report_output = await llm.astructured_predict(
                            ComparisonReportOutput,
                            prompt,
                            requirements_text=ev.requirements_text,
                            datasheet_content=str(datasheet_content),
                        )
                        
                        return StopEvent(result={"report": report_output, "datasheet_content": datasheet_content})
                
                # Process files
                import nest_asyncio
                nest_asyncio.apply()
                
                for uploaded_file in st.session_state.uploaded_files:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Run workflow
                        workflow = SolarPanelComparisonWorkflow(
                            agent=agent,
                            requirements_text=requirements_text,
                            verbose=False,
                            timeout=180
                        )
                        
                        result = asyncio.run(workflow.run(datasheet_path=tmp_path))
                        
                        # Store report
                        report_data = {
                            'filename': uploaded_file.name,
                            'report': result["report"].model_dump_json(indent=4),
                            'report_obj': result["report"]
                        }
                        st.session_state.reports.append(report_data)
                    
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                
                st.success(f"✅ Successfully generated {len(st.session_state.uploaded_files)} comparison report(s)!")
            
            except Exception as e:
                st.error(f"❌ Error generating reports: {str(e)}")
                st.exception(e)

# Comparison Reports Section
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("### 📊 Comparison Report(s)")

if st.session_state.reports:
    st.markdown(f"**{len(st.session_state.reports)} report(s) generated**")
    
    # Display each report
    for idx, report_data in enumerate(st.session_state.reports):
        with st.expander(f"📄 Report {idx + 1}: {report_data['filename']}", expanded=(idx == 0)):
            # Summary card
            report_obj = report_data['report_obj']
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if report_obj.meets_requirements:
                    st.success("✅ MEETS REQUIREMENTS")
                else:
                    st.error("❌ DOES NOT MEET REQUIREMENTS")
            
            with col2:
                st.markdown(f"**Component:** {report_obj.component_name}")
                st.markdown(f"**Summary:** {report_obj.summary}")
            
            st.markdown("---")
            
            # Detailed comparison
            st.markdown("#### Detailed Parameter Comparison")
            
            details = report_obj.details
            
            # Create a table-like display
            params = [
                ("⚡ Maximum Power", details.maximum_power),
                ("🔋 Minimum Power", details.minimum_power),
                ("📏 Max Length", details.max_length),
                ("⚖️ Max Weight", details.max_weight),
                ("📜 Certification", details.certification),
                ("📈 Efficiency", details.efficiency),
                ("🌡️ Temperature Coefficient", details.temperature_coefficient),
                ("🛡️ Warranty", details.warranty),
            ]
            
            for param_name, detail_item in params:
                col1, col2, col3 = st.columns([2, 1, 4])
                
                with col1:
                    st.markdown(f"**{param_name}**")
                
                with col2:
                    if detail_item.status == "PASS":
                        st.success("✓ PASS")
                    else:
                        st.error("✗ FAIL")
                
                with col3:
                    st.markdown(detail_item.explanation)
            
            st.markdown("---")
            
            # Raw JSON output
            with st.expander("🔍 View Raw JSON Report"):
                st.json(report_data['report'])
    
    # Clear reports button
    if st.button("🗑️ Clear All Reports", type="secondary"):
        st.session_state.reports = []
        st.rerun()

else:
    st.info("ℹ️ No comparison reports generated yet. Upload datasheets and click 'Analyze & Compare' to generate reports.")

# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 1rem;'>
        <small>Solar Panel Expert • Powered by LlamaExtract & OpenAI • Built with Streamlit ☀️</small>
    </div>
""", unsafe_allow_html=True)