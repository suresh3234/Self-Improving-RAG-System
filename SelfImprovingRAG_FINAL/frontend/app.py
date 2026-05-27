import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os

# Page Config
st.set_page_config(
    page_title="Self-Improving RAG Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; border: 1px solid #464b5d; }
    .stButton>button:hover { border-color: #ff4b4b; color: #ff4b4b; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    .stChatMessage { border-radius: 10px; padding: 10px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("🧠 Self-Improving RAG")
    st.markdown("---")
    
    menu = st.radio("Navigation", ["Query Interface", "System Analytics", "Data Ingestion"])
    
    st.markdown("---")
    if st.button("Refresh System Stats"):
        st.rerun()

# --- Functions ---
def get_stats():
    try:
        response = requests.get(f"{API_URL}/stats")
        return response.json() if response.status_code == 200 else None
    except:
        return None

def api_error_message(response):
    try:
        data = response.json()
        return data.get("detail", response.text)
    except Exception:
        return response.text

# --- Main Logic ---

if menu == "Query Interface":
    st.header("🔍 Intelligent Query Interface")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("metadata"):
                with st.expander("System Reasoning & Sources"):
                    st.json(message["metadata"])

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agentic workflow in progress..."):
                try:
                    response = requests.post(f"{API_URL}/query", json={"query": prompt})
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["generated_answer"]["answer"]
                        
                        # Prepare metadata for display
                        meta = {
                            "Strategy Used": data["rewritten_query"]["strategy_used"],
                            "Retrieval Score": f"{data['retrieval_evaluation']['overall_score']:.2f}",
                            "Faithfulness": f"{data['evaluation_scores']['faithfulness']:.2f}",
                            "Answer Relevancy": f"{data['evaluation_scores']['answer_relevancy']:.2f}",
                            "Sources": data["generated_answer"]["sources"]
                        }
                        
                        st.markdown(answer)
                        with st.expander("System Reasoning & Sources"):
                            st.json(meta)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "metadata": meta
                        })
                        
                        # Feedback Buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("👍 Useful"):
                                st.success("Feedback saved!")
                        with col2:
                            if st.button("👎 Not Useful"):
                                st.error("Feedback saved!")
                    else:
                        st.error(f"Error: {api_error_message(response)}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

elif menu == "System Analytics":
    st.header("📊 Real-time Performance Analytics")
    
    stats = get_stats()
    if stats:
        # Top Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Indexed Chunks", stats["indexed_chunks"])
        avg_faith = pd.DataFrame(stats["recent_feedback"])["faithfulness_score"].mean() if stats["recent_feedback"] else 0
        m2.metric("Avg Faithfulness", f"{avg_faith:.2f}")
        m3.metric("Active Strategies", len(stats["strategy_weights"]))
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Strategy Performance (Weight Distribution)")
            weights = stats["strategy_weights"]
            df_weights = pd.DataFrame([{"Strategy": k, "Weight": v} for k, v in weights.items()])
            fig = px.pie(df_weights, values='Weight', names='Strategy', hole=.3,
                         color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Recent Evaluation History")
            if stats["recent_feedback"]:
                df_history = pd.DataFrame(stats["recent_feedback"])
                st.dataframe(df_history[["query", "strategy_used", "faithfulness_score", "relevancy_score", "overall_reward"]])
            else:
                st.info("No feedback records yet.")

        # Timeline Chart
        if stats["recent_feedback"]:
            st.subheader("Quality Scores Timeline")
            df_history = pd.DataFrame(stats["recent_feedback"])
            df_history["created_at"] = pd.to_datetime(df_history["created_at"])
            fig = px.line(df_history, x="created_at", y=["faithfulness_score", "relevancy_score"], 
                          title="Quality Metrics Over Time", markers=True)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not connect to API for statistics.")

elif menu == "Data Ingestion":
    st.header("📁 Knowledge Ingestion")
    st.info("Upload documents to the 'data/raw' folder and trigger ingestion here.")
    
    default_data_dir = os.path.abspath("data/raw")
    folder_path = st.text_input("Local Folder Path", value=default_data_dir)
    
    if st.button("Start Ingestion Process"):
        with st.spinner("Analyzing and indexing documents..."):
            try:
                clean_folder_path = folder_path.strip().strip('"').strip("'")
                response = requests.post(f"{API_URL}/ingest", json={"directory_path": clean_folder_path})
                if response.status_code == 200:
                    st.success("Ingestion started! Check terminal for progress.")
                else:
                    st.error(f"Failed: {api_error_message(response)}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
    
    st.markdown("---")
    st.subheader("Current Index Status")
    stats = get_stats()
    if stats:
        st.write(f"Total Chunks in Index: **{stats['indexed_chunks']}**")
