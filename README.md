# BioVis: An Integrated Tool for Data Visualization, AI-Powered Interpretation, and Scholarly Literature Search via Natural Language Query (NLQ)

## Overview

BioVis is a Streamlit-based web application that combines data visualization with AI-powered insights. It enables users to upload datasets, generate interactive charts using natural language queries, and discover relevant academic research papers.mIt uses the plotly library for data vizualization, deepseek V3.1 for NLP and semantic scholar for literature search. And for the conversational feature we use the gpt oss 20b due to it's consistent and structured output format and qwen2.5-vl-32b-instruct is selected for processing the user quey with his uploaded graph to help the undersatnding realted it's graph pattern or trends.


![User Interface](https://github.com/MuhammadZainButt0/BioVis/blob/main/overview.jpg)

       
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

### 1. Clone the repository

```bash
git clone https://github.com/MuhammadZainButt0/BioVis.git
cd BioVis
```

### 2. Create a Virtual Environment (Recommended)
#### For Windows:
```powershell
python -m venv env_name
.\env_name\Scripts\Activate.ps1
```
#### For Linux:
```powershell
python3 -m venv env_name
source env_name\bin\activate
```
User can also skip the above step if he don't have the knowlegde of virtual environment, and can proceed to the next step.

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

## Results

Following are the some outcomes of BioVis, on different datasets. In the following panels (section A) show the dataset, (section B) shows the interactive visualization with plotly, (section D) shows the AI-generated explanation and formulated query out of graph or data  and (section C)
shows the relevant research papers which BioVis retrieve from the Semantic scholar upon thhe formulated query as: 

- **Figures**:
  1. **On Gene Expression Dataset**
      
     ![Gene Expression Dataset](https://github.com/MuhammadZainButt0/BioVis/blob/main/results/Gene_expression.png)
     
  2. **Chromatin Accesible dataset**
     
     ![Chromatin Accesible dataset](https://github.com/MuhammadZainButt0/BioVis/blob/main/results/Chromatin_accessible.png)
     
  3. **Clinical Diabetic Dataset**
     
     ![Clinical Diabetic Dataset](https://github.com/MuhammadZainButt0/BioVis/blob/main/results/Clinical_Diabetic.png)



## Deployment

BioVis is deployed on hugging face. If user wants to test or use the BioVis, [click here](https://huggingface.co/spaces/ml4genomics/BioVis.)

# Tips for Success

- Ensure your input files are correctly formatted such as csv, xlsx or tsv,  and contain all information required for visualization.
- Try to make query in simple and detail manner, avoid informal words. 
- On every run the results may be different, so check the reliabilty before the usage of information.
  
# Contributing
We welcome contributions! Please fork the repository and submit pull requests for any enhancements or bug fixes.

# Contact Us
For any questions or issues, please contact us at:
- **Mr. Muhammad Zain Butt**: [zain.202302328@gcuf.edu.pk](mailto:zain.202302328@gcuf.edu.pk)
- **Mr. Rana Sheraz Ahmad**: [ranasheraz.202101902@gcuf.edu.pk](mailto:ranasheraz.202101902@gcuf.edu.pk)
- **Dr. Muhammad Tahir ul Qamar**: [m.tahirulqamar@hotmail.com](mailto:tahirulqamar@gcuf.edu.pk)
- **Eman Fatima**: [eman.202204127@gcuf.edu.pk](mailto:eman.202204127@gcuf.edu.pk)



