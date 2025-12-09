# BioVis 

Transform Your Data Into Interactive Visualizations with AI

## Overview

BioVis is a Streamlit-based web application that combines data visualization with AI-powered insights. It enables users to upload datasets, generate interactive charts using natural language queries, and discover relevant academic research papers.

## Features

- **Data Upload & Management**: Support for CSV, TSV, and Excel (.xlsx) formats
- **AI-Powered Chart Generation**: Generate Plotly visualizations from natural language queries
- **Interactive Visualizations**: Explore data with Plotly's interactive charts
- **AI Insights**: Automatic analysis and explanation of generated charts
- **Academic Research Integration**: Auto-discovery of relevant research papers via Semantic Scholar
- **Graph Interpreter**: Analyze uploaded graph images with AI
- **Data Q&A**: Ask questions about your dataset and get AI-driven answers
- **Sample Datasets**: Pre-loaded datasets for quick testing (Apple Stock, Gene Expression, Hospital Data)

## Prerequisites

- Python 3.8+
- API keys for:
  - DeepSeek API (via OpenRouter)
  - GPT API (via OpenRouter)
  - Qwen API (via OpenRouter)
  - Semantic Scholar API

## Installation

### 1. Clone or Download the Project

```bash
cd c:\Users\PMLS\Desktop\BioVis_Code
```

### 2. Create a Virtual Environment (Recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory (as in BioVis_Code):

```
DEEPSEEK_API_KEY=your_deepseek_key_here
GPT_API_KEY=your_gpt_key_here
QWEN_API_KEY=your_qwen_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key_here
```

## 5. Running the Application

```powershell
streamlit run BioVis.py
```

The app will open in your browser at `http://localhost:8501`

