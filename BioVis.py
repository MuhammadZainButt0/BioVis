import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import requests
from dotenv import load_dotenv
import time
import io
import base64
# from streamlit_option_menu import option_menu
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #2A24CE;">
  <h2 style="color: white; margin-right: 20px;">BioVis</h2>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav ml-auto">  <!-- ml-auto pushes items to the right -->
      <li class="nav-item active">
        <a class="nav-link" href="#">Resources <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#">Contact Us</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)



st.set_page_config(
    page_title="BioVis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    
    .main-header {
        background: #0800D4;
        padding: 2rem;
        border-radius: 30px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.3rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .status-card {
        padding: 1.5rem;
        border-radius: 12px;
        color: black;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    .section-header {
        background: #0800D4;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        border-left: 5px solid #1338B3;
        
    }
    
    .section-header h3 {
        margin: 0;
        color: white;
        font-weight: 600;
    }
    
    .sidebar-section {
        background: #f8f9ff;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 5px solid black;
    }
    
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        flex: 1;
        background: #0800D4;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(19, 56, 179, 0.3);
        color: white;
    }
    
    .metric-card h4 {
        margin: 0 0 0.5rem 0;
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .metric-card p {
        margin: 0;
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e3e8ff;
        padding: 0.7rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .uploadedFile {
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .custom-warning {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 8px;
        color: #2d3436;
        margin: 1rem 0;
        border-left: 4px solid #e17055;
    }
    
    .custom-info {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        color: black;
        margin: 1rem 0;
        border-left: 4px solid #0056b3;
    }
    
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .example-query {
        background: #f8f9ff;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* 1. Hide Streamlit’s own top header */
header[data-testid="stHeader"] {
    display: none !important;
}

/* 2. Bring your Bootstrap navbar to the very front */
nav.navbar {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 999999 !important;
}

/* 3. Push the whole Streamlit app content downward */
.stApp {
    padding-top: 75px !important;
}

/* 4. Fix Streamlit’s main container (prevents overlapping) */
main[data-testid="stAppViewContainer"] {
    padding-top: 75px !important;
}

</style>
""", unsafe_allow_html=True)


load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
gpt_api_key = os.getenv("GPT_API_KEY")
qwen_api_key = os.getenv("QWEN_API_KEY")  
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

api_errors = []
if not deepseek_api_key:
    api_errors.append("DeepSeek API Key")
if not gpt_api_key:
    api_errors.append("GPT API Key")
if not qwen_api_key:
    api_errors.append("QWEN API Key")
if not SEMANTIC_SCHOLAR_API_KEY:
    api_errors.append("Semantic Scholar API Key")

if api_errors:
    st.markdown(f"""
    <div class="custom-warning">
        <h4> Configuration Required</h4>
        <p>Missing API keys: {', '.join(api_errors)}</p>
        <p>Please configure your .env file with the required API keys to continue.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

deepseek_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=deepseek_api_key
)

gpt_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=gpt_api_key
)

qwen_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=qwen_api_key
)

def clean_code(text: str) -> str:
    """Remove markdown fences or extra text from code output."""
    return text.strip().replace("```python", "").replace("```", "")

def generate_chart_code(columns: str, query: str) -> str:
    messages = [
        {"role": "system", "content": "You are a data visualization assistant."},
        {"role": "user", "content": f"The user uploaded a dataset with the following columns:\n{columns}\n\nThe user asked: '{query}'\n\nWrite valid Python code that:\n1. Uses Plotly (px or go) to create the most appropriate chart.\n2. The dataframe is available as `df`.\n3. Save the chart as a variable named `fig`.\n\nOnly return the code. Do not include explanations."}
    ]
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1",
            messages=messages,
            temperature=0,
        )
        return clean_code(response.choices[0].message.content)
    except Exception as e:
        st.error(f" Error generating chart code: {e}")
        return ""

def fix_chart_code(code: str, error: str) -> str:
    messages = [
        {"role": "system", "content": (
            "You are a data visualization assistant. "
            "IMPORTANT RULES:\n"
            "1. Never create or inject fake/sample dataframes (like df = pd.DataFrame(...)).\n"
            "2. Always assume the user already uploaded a dataframe named `df`.\n"
            "3. If certain required columns are missing, insert a validation check that raises a ValueError with a clear message.\n"
            "4. Always return corrected Python code that produces a Plotly chart named `fig`.\n"
            "5. Only return the corrected code, no explanations."
        )},
        {"role": "user", "content": f"The following code failed:\n\n{code}\n\nError:\n{error}\n\nFix the code according to the rules."}
    ]
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1",
            messages=messages,
            temperature=0,
        )
        return clean_code(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error fixing chart code: {e}")
        return code

def generate_explanation(query: str, columns: str, stats_summary: str) -> str:
    code = generate_chart_code(columns, query)
    messages = [
        {"role": "system", "content": "You are a data visualization expert. Provide clear, concise explanations of generated chart.Don't show the <｜begin▁of▁sentence｜>"},
        {"role": "user", "content": f"User query: '{query}'\nGenerated Graph code: '{code}'\nDataset columns: {columns}\nKey statistics from the chart data: {stats_summary}\n\nExplain the chart in natural language: Describe what it shows, key trends, insights, and implications. Keep it to 3-5 sentences. Be engaging and insightful."}
    ]
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1",
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating explanation: {e}")
        return "Could not generate explanation due to API error."

def analyze_explanation_for_query(explanation: str, original_query: str, columns: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic and literature research assistant."
                
                
                
            
            ),
        },
        {
            "role": "user",
            "content": (
                f"Dataset columns: {columns}\n"
                f"Chart explanation: {explanation}\n"
                f"Original user query: {original_query}\n\n"
                "Craft precise,  a domain-specific query (5–12 words) for Semantic Scholar "
                "based on the dataset columns, chart explanation, and user query. "
                "Generate a concise academic search query specific to this dataset."
                "If dataset columns contain abbreviations or symbols (e.g., APPL), expand them into their real-world entity names (e.g., Apple). "
                "Queries must incorporate relevant dataset entities (company names, regions, populations, etc.) "
                "Focus on methods, variables, and domain terminology of dataset. "
                "Target literature that can support or challenge the observed trend. "
                "Do not include chart names or explanations. "
                "your response should be a concise academic search query only, no reasoning and explanation! ."
            
            ),
        },
    ]
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1",
            messages=messages,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating academic query: {e}")
        return original_query

def search_semantic_scholar(search_query: str, retries: int = 3, backoff: int = 5):
    """Search Semantic Scholar API for the most relevant papers with retries & exponential backoff."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": search_query,
        "limit": 5,
        "sort": "relevance",
        "fields": "title,authors,year,abstract,citationCount,url"
    }

    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)

            if response.status_code == 429:
                wait_time = backoff * (2 ** attempt)
                st.warning(f"Semantic Scholar rate limit hit (429). Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            if "data" in data and data["data"]:
                papers = []
                for paper in data["data"]:
                    papers.append({
                        "Title": paper.get("title", "N/A"),
                        "Authors": ", ".join([a.get("name", "N/A") for a in paper.get("authors", [])]),
                        "Year": paper.get("year", "N/A"),
                        "Citations": paper.get("citationCount", 0),
                        "Abstract Snippet": paper.get("abstract", "N/A")[:200] + "..." if paper.get("abstract") else "N/A",
                        "URL": paper.get("url", "N/A")
                    })
                return pd.DataFrame(papers)
            else:
                return pd.DataFrame({"Message": ["No relevant papers found."]})

        except Exception as e:
            st.error(f"Semantic Scholar API error: {e}")
            if attempt == retries - 1:
                return pd.DataFrame({"Message": ["Search failed."]})
            time.sleep(backoff)

    return pd.DataFrame({"Message": ["Failed after retries due to rate limits."]})

def interpret_graph_with_qwen(image_base64: str, question: str) -> str:
    """Interpret a graph image"""
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please analyze this graph/chart image and answer the following question: {question}\n\nProvide a concise and precise interpretation including trends, patterns, key insights, and any notable observations. If you are not clear about the graph/chart, say so and ask the user to provide more information.Answers should be no more than 6 to 7 lines."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        response = qwen_client.chat.completions.create(
            model="qwen/qwen2.5-vl-32b-instruct:free",
            messages=messages,
            max_tokens=1000,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error interpreting graph with qwen: {str(e)}"

def create_sample_data_from_drive(sample_choice: str) -> pd.DataFrame:
    """Load dataset from Google Drive for given sample_choice."""
    
    try:
        if sample_choice == "Population Data":
            # Use the share link you provided
            share_url = "https://drive.google.com/file/d/1NttDmesL-sInl27PvYlFZRCUZNgpJQL8/view?usp=sharing"
            file_id = share_url.split("/")[-2]
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            df = pd.read_csv(download_url)
            return df
        if sample_choice == "Immigration Data":
            # Use the share link you provided
            share_url = "https://drive.google.com/file/d/1DB9107DwAphw9cfTnkO6LxqbHH6SgGzy/view?usp=drive_link"
            file_id = share_url.split("/")[-2]
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            df = pd.read_csv(download_url)
            return df
        if sample_choice == "Clinical_Diabetes Data":
            # Use the share link you provided
            share_url = "https://drive.google.com/file/d/1kdhxtQSsbU0xdgv4jPgYvEX-DDRIEOqc/view?usp=drive_link"
            file_id = share_url.split("/")[-2]
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            df = pd.read_csv(download_url)
            return df
        # fallback or other sample choices ...
        return None

    except Exception as e:
        st.error(f"⚠️ Error creating sample data: {e}")
        return None

if 'df' not in st.session_state:
    st.session_state.df = None
if 'fig' not in st.session_state:
    st.session_state.fig = None


# Show intro only if no dataset is loaded
if st.session_state.df is None:
    st.markdown("""
    <div class="main-header">
        <h1> What is BioVis?</h1>
        <p>BioVis is an innovative web-based tool that integrates data visualization, AI-powered interpretation, and relevant literature search—all from a single query.</p>
        <p>It allows users to visualize their uploaded data and discover pertinent research articles seamlessly. Additionally, BioVis supports both conversational and uploaded</p>
        <p> graph interpretation to enhance users’ understanding of their data trends and patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Overview")

    with st.container():
        # Centering using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("overview.png", width=900)
            if st.button(" Get Started ➡", use_container_width=True):
                st.write()



with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h2 style="color: #667eea; margin: 0;">Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("### Upload Your File")
    uploaded_file = st.file_uploader(
        "Choose a data file",
        type=["csv", "tsv", "xlsx"],
        help="Supported formats: CSV, TSV, Excel (.xlsx)"
    )
    st.markdown("### Sample Datasets")
    sample_choice = st.radio(
        "Or load a sample dataset:",
        ( "None","Population Data", "Immigration Data", "Clinical_Diabetes Data"),
        
    )
    

    st.markdown("### AI Assistant")
    with st.expander("Data Q&A", expanded=False):
        qa_query = st.text_area(
            "Your question:",
            placeholder="e.g., What patterns do you see in this dataset?",
            height=80
        )
        if st.button("Ask AI", use_container_width=True):
            if not qa_query.strip():
                st.warning("Please enter a question first.")
            elif st.session_state.df is None:
                st.error("No data loaded yet.")
            else:
                with st.spinner("AI is analyzing your data..."):
                    try:
                        df = st.session_state.df
                        
                        # Prepare data summary
                        data_summary = f"""
Dataset Info:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}

Data Preview (first 5 rows):
{df.head().to_string()}

Statistical Summary:
{df.describe().to_string() if len(df.select_dtypes(include=['number']).columns) > 0 else 'No numeric columns'}

Missing Values:
{df.isnull().sum().to_dict()}
"""
                        
                        messages = [
                            {
                                "role": "system", 
                                "content": "You are a data analysis expert. Analyze the provided dataset and answer the user's question with clear, insightful responses. Focus on patterns, trends, and actionable insights."
                            },
                            {
                                "role": "user",
                                "content": f"Your Answer should not be more than 6 to 7 lines. Here is my dataset:\n\n{data_summary}\n\nQuestion: {qa_query}\n\nProvide a well-structured but concise and precise answer based on the data.Summarize key findings and insights.don't use the *"
                            }
                        ]
                        
                        response = gpt_client.chat.completions.create(
                            model="openai/gpt-oss-20b:free",
                            messages=messages,
                            temperature=0.7,
                            max_tokens=800
                        )
                        
                        answer = response.choices[0].message.content.strip()
                        
                        st.markdown("**AI Analysis:**")
                        st.markdown(f"""
                        <div class="success-card">
                            {answer}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error getting AI response: {str(e)}")

    with st.expander("Graph Interpreter", expanded=False):
        uploaded_graph = st.file_uploader(
            "Upload Graph Image",
            type=["png", "jpg", "jpeg"],
            key="graph_upload"
        )
        if uploaded_graph:
            st.image(uploaded_graph, caption="Uploaded Graph", use_container_width=True)
        
        graph_question = st.text_area(
            "Question about the graph:",
            placeholder="e.g., What trends does this graph show?",
            key="graph_question",
            height=80
        )
        
        if st.button("Analyze Graph", use_container_width=True):
            if uploaded_graph is None:
                st.warning("Please upload a graph image first.")
            elif not graph_question.strip():
                st.warning("Please enter a question first.")
            else:
                with st.spinner("Qwen is interpreting your graph..."):
                    try:
                        img_bytes = uploaded_graph.read()
                        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                        interpretation = interpret_graph_with_qwen(img_b64, graph_question)
                        
                        st.markdown("**Qwen's Analysis:**")
                        st.markdown(f"""
                        <div class="success-card">
                            {interpretation}
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error interpreting graph: {e}")

current_df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            current_df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".tsv"):
            current_df = pd.read_csv(uploaded_file, sep="\t")
        else:
            current_df = pd.read_excel(uploaded_file)

        current_df.columns = current_df.columns.str.strip()
        current_df.columns = current_df.columns.str.replace(r"\s+", "_", regex=True)
        current_df.columns = current_df.columns.str.replace(r"[^\w]", "", regex=True)

        st.session_state.df = current_df
        
    except Exception as e:
        st.markdown(f"""
        <div class="custom-warning">
            <h4>File Processing Error</h4>
            <p>Unable to read the uploaded file: {str(e)}</p>
            <p>Please ensure your file is in the correct format (CSV, TSV, or Excel).</p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.df = None

elif sample_choice != "None" and uploaded_file is None:
    current_df = create_sample_data_from_drive(sample_choice)
    if current_df is not None:
        st.session_state.df = current_df
        st.sidebar.success(f"Loaded {sample_choice} dataset.")
    else:
        st.sidebar.error("Failed to create sample dataset.")

if st.session_state.df is not None:
    df = st.session_state.df
    rows, columns = df.shape
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>{rows:,}</h4>
            <p>Rows</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>{columns}</h4>
            <p>Columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        st.markdown(f"""
        <div class="metric-card">
            <h4>{numeric_cols}</h4>
            <p>Numeric</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        text_cols = len(df.select_dtypes(include=['object']).columns)
        st.markdown(f"""
        <div class="metric-card">
            <h4>{text_cols}</h4>
            <p>Text</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
        <h3>Data Preview</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(
        df.head(20), 
        use_container_width=True,
        height=400
    )

    st.markdown("""
    <div class="section-header">
        <h3>Chart Generation Studio</h3>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "Enter your Query",
        placeholder="e.g., 'Create a line chart showing sales trends over time with different colors for each product category'",
        help="Be specific about the type of chart, columns to use, and any styling preferences"
    )

    # Add Submit button
    if st.button("Submit Query", use_container_width=True):
        if not query.strip():
            st.warning("⚠️ Please enter a query first.")
        elif st.session_state.df is None:
            st.error("❌ No dataset loaded yet.")
        else:
            df = st.session_state.df
            columns = ", ".join(df.columns)

            with st.spinner("🎨 Generating Graph..."):
                code = generate_chart_code(columns, query)

            if code:
                with st.expander("View Generated Code", expanded=False):
                    st.code(code, language="python")

                max_attempts = 3
                attempt = 0
                success = False
                local_vars = {"df": df, "px": px, "go": go}

                while attempt < max_attempts and not success:
                    try:
                        with st.spinner(f"⚙️ Executing visualization (attempt {attempt + 1})..."):
                            exec(code, {}, local_vars)

                        if "fig" not in local_vars:
                            raise ValueError("No `fig` variable was created in the code.")

                        st.markdown("""
                        <div class="section-header">
                            <h3>📊 Your Interactive Visualization</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        local_vars["fig"].update_layout(height=600)

                        st.plotly_chart(
                            local_vars["fig"], 
                            use_container_width=True, 
                            config={"displayModeBar": False}
                            )

                        st.session_state.fig = local_vars["fig"]
                        success = True

                        # AI Insights and Research Section
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.markdown("""
                            <div class="section-header">
                                <h3>🤖 AI Insights</h3>
                            </div>
                            """, unsafe_allow_html=True)

                            with st.spinner("🧠 Generating insights..."):
                                numeric_cols = df.select_dtypes(include=['number']).columns
                                stats_summary = f"Dataset shape: {df.shape}. Numeric columns: {list(numeric_cols)}. Sample stats: {df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else 'No numeric data.'}"
                                explanation = generate_explanation(query, columns, stats_summary)

                            st.markdown(f"""
                            <div class="custom-info">
                                {explanation}
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown("""
                            <div class="section-header">
                                <h3> Academic Research</h3>
                            </div>
                            """, unsafe_allow_html=True)

                            with st.spinner("🔬 Generating research query..."):
                                academic_query = analyze_explanation_for_query(explanation, query, columns)

                            st.markdown(f"""
                            <div class="status-card">
                                <h4>Generated Research Query</h4>
                                <p style="font-style: italic;">{academic_query}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Semantic Scholar Search
                        st.markdown("""
                        <div class="section-header">
                            <h3>📖 Relevant Research Papers (Semantic Scholar)</h3>
                        </div>
                        """, unsafe_allow_html=True)

                        with st.spinner("🔍 Searching Semantic Scholar..."):
                            papers_df = search_semantic_scholar(academic_query)

                        if not papers_df.empty and "Title" in papers_df.columns:
                            st.markdown("### 📄 Semantic Scholar Papers")
                            for idx, row in papers_df.iterrows():
                                st.markdown(f"""
                                <div style="background: #f8f9ff; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid #667eea;">
                                    <h4 style="margin: 0 0 0.5rem 0; color: #2c3e50;">
                                        <a href="{row['URL']}" target="_blank" style="text-decoration: none; color: #667eea;">
                                            📎 {row['Title']}
                                        </a>
                                    </h4>
                                    <p style="margin: 0; color: #666; font-size: 0.9rem;">
                                        <strong>Authors:</strong> {row['Authors']} | 
                                        <strong>Year:</strong> {row['Year']} | 
                                        <strong>Citations:</strong> {row['Citations']}
                                    </p>
                                    <p style="margin: 0.5rem 0 0 0; color: #555; font-size: 0.85rem;">
                                        {row['Abstract Snippet']}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="custom-warning">
                                <h4>📭 No Semantic Scholar Papers Found</h4>
                                <p>No relevant research papers were found for this query.</p>
                            </div>
                            """, unsafe_allow_html=True)

                    except Exception as e:
                        attempt += 1
                        st.markdown(f"""
                        <div class="custom-warning">
                            <h4>⚠️ Attempt {attempt} Failed</h4>
                            <p>Error: {str(e)}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        if attempt < max_attempts:
                            with st.spinner("🔧 Auto-fixing code..."):
                                current_hash = hash(code)
                                new_code = fix_chart_code(code, str(e))

                            if hash(new_code) == current_hash:
                                st.markdown("""
                                <div class="custom-warning">
                                    <h4>❌ Unable to Auto-Fix</h4>
                                    <p>The code correction system returned the same code. Please try rephrasing your query.</p>
                                </div>
                                """, unsafe_allow_html=True)
                                break

                            code = new_code
                            st.markdown(f"""
                            <div class="section-header">
                                <h3>🔧 Auto-Corrected Code (Attempt {attempt})</h3>
                            </div>
                            """, unsafe_allow_html=True)

                            with st.expander("View Corrected Code", expanded=True):
                                st.code(code, language="python")
                            local_vars = {"df": df, "px": px, "go": go}
                        else:
                            st.markdown("""
                            <div class="custom-warning">
                                <h4>❌ Visualization Failed</h4>
                                <p>Unable to generate a valid chart after multiple attempts. Please try:</p>
                                <ul>
                                    <li>Rephrasing your query with more specific column names</li>
                                    <li>Checking if your data format is compatible</li>
                                    <li>Simplifying your visualization request</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)

                if not success:
                    st.markdown("""
                    <div class="custom-warning">
                        <h4>💡 Suggestion</h4>
                        <p>Rephrase your request with different terminology.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="custom-warning">
                    <h4>❌ Code Generation Failed</h4>
                    <p>Unable to generate chart code. Please check your API configuration or try a simpler query.</p>
                </div>
                """, unsafe_allow_html=True)

