# BioVis: An Integrated Tool for Data Visualization, AI-Powered Interpretation, and Scholarly Literature Search via Natural Language Query (NLQ)

## Overview

BioVis is a Streamlit-based web application that combines data visualization with AI-powered insights. It enables users to upload datasets, generate interactive charts using natural language queries, and discover relevant academic research papers.
![User Interface](https://github.com/MuhammadZainButt0/BioVis/blob/main/overview.jpg)

Input: Query + Upload Data File
Ouput:

    i.   Required Graph 
    
    ii.  AI generated graph explanation 
    
    iii. Five most relevant articles related to graph or data 
       
## Features

- **Data Upload & Management**: Support for CSV, TSV, and Excel (.xlsx) formats
- **AI-Powered Chart Generation**: Generate Plotly visualizations from natural language queries
- **AI Insights**: Automatic analysis and explanation of generated charts
- **Academic Research Integration**: Auto-discovery of relevant research papers via Semantic Scholar
- **Graph Interpreter**: Analyze uploaded graph images with AI
- **Data Q&A**: Ask questions about your dataset and get AI-driven answers
- **Sample Datasets**: Pre-loaded datasets for quick testing (Apple Stock, Gene Expression, Hospital Data)

## Prerequisites

- [Python 3.8+](https://www.python.org/downloads/)
- API keys for:
  - [DeepSeek API (via OpenRouter)](https://openrouter.ai/deepseek/deepseek-chat-v3.1)
  - [GPT API (via OpenRouter)](https://openrouter.ai/openai/gpt-oss-20b:free)
  - [Qwen API (via OpenRouter)](https://openrouter.ai/qwen/qwen2.5-vl-32b-instruct)
  - [Semantic Scholar API](https://www.semanticscholar.org/product/api#api-key)

## Installation

### 1. Clone or Download the Project

```bash
git clone https://github.com/MuhammadZainButt0/BioVis.git
cd BioVis
```

### 2. Create a Virtual Environment (Recommended)

```powershell
python -m venv env_name
.\env_name\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory (as in BioVis):

```
DEEPSEEK_API_KEY="your_deepseek_key_here"
GPT_API_KEY="your_gpt_key_here"
QWEN_API_KEY="your_qwen_key_here"
SEMANTIC_SCHOLAR_API_KEY="your_semantic_scholar_key_here"
```
If you don't have above API's, then create account on [OpenRouter](https://openrouter.ai/models) and [Semantic Scholar](https://www.semanticscholar.org/product/api) to generate the above keys.

## 5. Running the Application

```powershell
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

