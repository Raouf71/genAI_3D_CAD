# Solar Panel Expert - Streamlit UI

A comprehensive web-based UI for comparing solar panel datasheets against design requirements.

## Features

### 🎯 Main Components

1. **Upload Docs Tab**
   - Drag-and-drop or browse to upload PDF datasheets
   - Support for multiple files
   - Real-time upload status
   - File size display

2. **Setup API Keys Tab**
   - Secure input for LlamaExtract API key
   - Secure input for OpenAI API key
   - Configuration status indicators

3. **Configure Parameters Tab**
   - Interactive sliders for 5 key specifications:
     - Maximum Power (W)
     - Minimum Power (W)
     - Maximum Length (mm)
     - Maximum Weight (kg)
     - Warranty (years)
   - Live configuration preview

4. **Comparison Report Section**
   - Detailed pass/fail analysis for each parameter
   - Summary cards with overall compliance status
   - Expandable reports for multiple datasheets
   - Raw JSON view option
   - Scrollable output area

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

0. **Create and activate separate python venv**
   ```bash
   python -m venv solar-panels-venv
   source solar-panels-venv/bin/activate
   ```

1. **Clone or download the project files**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify your backend code**
   Make sure your backend workflow code is accessible. The app expects the following to be importable:
   - `llama_cloud_services`
   - `llama_index.core.workflow`
   - `llama_index.llms.openai`

## Running the App

### Start the Streamlit server:
```bash
streamlit run frontend/streamlit_app.py
```

### The app will automatically open in your default browser at:
```
http://localhost:8501
```

## Usage Guide

### Step 1: Configure API Keys
1. Navigate to the "🔑 Setup API Keys" tab
2. Enter your LlamaExtract API key
3. Enter your OpenAI API key
4. Wait for the success confirmation

### Step 2: Set Design Requirements
1. Go to the "⚡ Configure Parameters" tab
2. Adjust sliders for your requirements:
   - Set maximum and minimum power thresholds
   - Define physical constraints (length, weight)
   - Set minimum warranty period
3. View your configuration in the expandable section

### Step 3: Upload Datasheets
1. Click on the "📤 Upload Docs" tab
2. Drag and drop PDF files or click to browse
3. Confirm upload success message
4. Review uploaded file names and sizes

### Step 4: Generate Reports
1. Click the "🔍 Analyze & Compare" button
2. Wait for processing (progress indicator will show)
3. Review generated reports in the "Comparison Report(s)" section

### Step 5: Review Results
Each report shows:
- ✅/❌ Overall compliance status
- Component name and summary
- Detailed parameter-by-parameter analysis:
  - Maximum Power
  - Minimum Power
  - Max Length
  - Max Weight
  - Certification
  - Efficiency
  - Temperature Coefficient
  - Warranty
- Raw JSON data (expandable)

<!-- ## Features & Functionality

### UI Elements
- **Solar panel icon (☀️)** in the header for branding
- **Three tabbed sections** for setup and configuration
- **Status messages** for upload and configuration states
- **Large scrollable report area** with vertical scrollbar
- **Color-coded pass/fail indicators**
- **Expandable sections** for detailed information

### Workflow
1. **Initialization**: Configure API keys and parameters
2. **Upload**: Add solar panel datasheet PDFs
3. **Processing**: Backend extracts data and generates comparison
4. **Display**: Results shown in structured, readable format

### Session Management
- Configuration persists across interactions
- Multiple reports can be generated in one session
- Clear all reports option available

## Troubleshooting

### "Backend dependencies not fully installed"
- Run: `pip install -r requirements.txt`
- Ensure all packages install successfully

### API Key Errors
- Verify your LlamaExtract API key is valid
- Verify your OpenAI API key has sufficient credits
- Check for typos in API keys

### Upload Issues
- Ensure files are valid PDFs
- Check file size isn't too large
- Verify file isn't corrupted

### Processing Errors
- Check internet connection
- Verify API keys are configured
- Review error message for specific issues
- Check console output for detailed error logs -->

## Technical Architecture

### Frontend (Streamlit)
- Single-page application
- Reactive UI components
- Session state management
- File upload handling

### Backend Integration
- LlamaExtract for PDF data extraction
- OpenAI GPT-4 for comparison analysis
- Pydantic schemas for data validation
- Async workflow execution

<!-- ## Customization

### Modify Parameter Ranges
Edit the slider configurations in the "Configure Parameters" tab section:
```python
st.slider(
    "⚡ Maximum Power (W)",
    min_value=100,      # Adjust minimum
    max_value=1000,     # Adjust maximum
    value=450,          # Adjust default
    step=10             # Adjust increment
)
```

### Add New Parameters
1. Add to session state initialization
2. Create new slider in the tab
3. Update requirements text generation
4. Modify comparison schema if needed -->

<!-- ### Styling
Modify the custom CSS in the `st.markdown()` section at the top of the file. -->

<!-- ## Support

For issues or questions:
- Check the console output for detailed error messages
- Review API documentation for LlamaExtract and OpenAI
- Verify all dependencies are correctly installed

## Version Information
- Streamlit: 1.30.0+
- Python: 3.8+
- LlamaExtract: Latest
- OpenAI: 1.0.0+ -->

[⬆ Back to Top](#-main-components)