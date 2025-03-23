import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest
from openai import OpenAI
from rag_vectorstore import load_documents, split_into_chunks, create_faiss_index, load_faiss_index, retrieve_context

# --- Streamlit Title & Intro ---
st.title("Smart Performance Forecasting for OpenShift Pods")
st.write("""
This application predicts the optimal number of pods required in OpenShift based on LoadRunner performance data. 
It uses ML + RAG (Retrieval-Augmented Generation) to enrich GPT predictions with historical knowledge base content.
""")

# --- Upload LoadRunner CSV File ---
uploaded_csv = st.file_uploader("Upload LoadRunner Performance Report", type=["csv"])

# --- Upload Optional Knowledge Base Documents ---
st.subheader("Upload Knowledge Base Files (txt, csv, md)")
kb_files = st.file_uploader("Upload documents to enhance GPT context", type=["txt", "csv", "md"], accept_multiple_files=True)

if kb_files:
    os.makedirs("knowledge_base", exist_ok=True)
    for file in kb_files:
        with open(os.path.join("knowledge_base", file.name), "wb") as f:
            f.write(file.getbuffer())
    docs = load_documents("knowledge_base")
    chunks = split_into_chunks(docs)
    db = create_faiss_index(chunks)
else:
    try:
        db = load_faiss_index()
    except:
        db = None

# --- OpenAI API ---
openai_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_key)

# --- Run if CSV file is uploaded ---
if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    required_cols = ["TPS", "CPU_Cores", "Memory_GB", "ResponseTime_ms", "CPU_Load", "Memory_Load"]
    if not all(col in df.columns for col in required_cols):
        st.error("Missing required columns.")
        st.stop()

    st.write("Sample Data:", df.head())

    features = ["TPS", "CPU_Cores", "Memory_GB", "ResponseTime_ms"]
    target_cpu = "CPU_Load"
    target_mem = "Memory_Load"

    X = df[features]
    y_cpu = df[target_cpu]
    y_mem = df[target_mem]

    X_train, X_test, y_cpu_train, y_cpu_test = train_test_split(X, y_cpu, test_size=0.2, random_state=42)
    _, _, y_mem_train, y_mem_test = train_test_split(X, y_mem, test_size=0.2, random_state=42)

    cpu_model = LinearRegression().fit(X_train, y_cpu_train)
    mem_model = LinearRegression().fit(X_train, y_mem_train)

    cpu_r2 = r2_score(y_cpu_test, cpu_model.predict(X_test))
    mem_r2 = r2_score(y_mem_test, mem_model.predict(X_test))

    st.write(f"Model Accuracy: CPU R² = {cpu_r2:.2f}, Memory R² = {mem_r2:.2f}")

    # Prediction UI
    tps = st.slider("Expected TPS", 10, 200, 40, 10)
    cpu = st.slider("CPU Cores per Pod", 1, 4, 1)
    mem = st.slider("Memory per Pod (GB)", 2, 8, 2)
    response = st.slider("Target Response Time (seconds)", 1, 5, 2, 1)
    response_ms = response * 1000  # Convert to ms for model

    def predict_pods(tps, cpu, mem, response_ms, max_cpu=75, max_mem=75):
        for pods in range(1, 50):
            avg_tps = tps / pods
            sample = pd.DataFrame([[avg_tps, cpu, mem, response_ms]], columns=features)
            cpu_pred = cpu_model.predict(sample)[0]
            mem_pred = mem_model.predict(sample)[0]
            if cpu_pred <= max_cpu and mem_pred <= max_mem:
                return pods, cpu_pred, mem_pred, True
        # Still return best-effort if not within limits
        sample = pd.DataFrame([[tps / 1, cpu, mem, response_ms]], columns=features)
        return 1, cpu_model.predict(sample)[0], mem_model.predict(sample)[0], False

    pods_needed, pred_cpu, pred_mem, is_recommended = predict_pods(tps, cpu, mem, response_ms)

    st.success(f"Estimated Pods Required: {pods_needed}")
    st.write(f"Estimated CPU Utilization: {pred_cpu:.2f}%")
    st.write(f"Estimated Memory Utilization: {pred_mem:.2f}%")

    if not is_recommended:
        st.warning("⚠️ This configuration exceeds resource limits. Not recommended for production use.")

    # --- GPT-4 + RAG Context ---
    user_question = st.text_input("Ask a performance-related question")
    if user_question:
        if db:
            context = retrieve_context(db, user_question)
        else:
            context = "No vector database found. Only using GPT without retrieval context."

        example_prompt = f"""
You are a senior performance engineer.
Based on the following LoadRunner sample data:

{df.head(3).to_string(index=False)}

Answer the user's query: "{user_question}"

If applicable, identify a potential performance problem using this data, explain the root cause, and provide tuning or optimization suggestions. Also summarize insights using historical context below:

{context}

Respond professionally and clearly.
"""

        res = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": example_prompt}],
            temperature=0.2
        )
        st.write("### AI Response:")
        st.write(res.choices[0].message.content)

# Footer
st.markdown("<br><br><p style='font-size:14px; text-align:center; color:gray;'>Developed by Devesh Kumar</p>", unsafe_allow_html=True)
