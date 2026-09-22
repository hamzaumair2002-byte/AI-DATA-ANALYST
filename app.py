import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst")
st.caption("Upload Excel/CSV → automatic analysis → charts → AI questions")

uploaded = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])

def make_profile(df):
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    profile = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": [{"name": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()), "unique": int(df[c].nunique(dropna=True))} for c in df.columns],
        "numeric_statistics": {},
        "categorical_top_values": {},
        "sample_rows": df.head(50).fillna("").astype(str).to_dict(orient="records")
    }
    for c in numeric[:30]:
        s = df[c].dropna()
        if len(s):
            profile["numeric_statistics"][c] = {
                "mean": round(float(s.mean()), 4), "median": round(float(s.median()), 4),
                "min": round(float(s.min()), 4), "max": round(float(s.max()), 4),
                "std": round(float(s.std()), 4) if len(s) > 1 else 0
            }
    for c in categorical[:20]:
        profile["categorical_top_values"][c] = {str(k): int(v) for k, v in df[c].value_counts(dropna=False).head(10).items()}
    return profile

def ask_ai(question, profile):
    if OpenAI is None:
        return "OpenAI package is not installed. Add `openai` to requirements.txt and redeploy."
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        return "AI is not connected yet. Add OPENAI_API_KEY to Streamlit Secrets, then restart the app."
    client = OpenAI(api_key=api_key)
    instructions = """You are an expert data analyst. Analyze the supplied dataset profile and answer the user's question using only the available data. Do not invent values. If the profile is insufficient, say what is missing. Give calculations/comparisons when supported. Use clear headings and bullets. Mention important data-quality limitations."""
    prompt = f"DATASET PROFILE:\n{profile}\n\nUSER QUESTION:\n{question}"
    try:
        response = client.responses.create(model="gpt-5.6-luna", instructions=instructions, input=prompt)
        return response.output_text
    except Exception as e:
        return f"AI request failed: {e}"

if uploaded:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.success(f"Loaded: {uploaded.name}")
        clean = df.copy()
        text_cols = clean.select_dtypes(include=["object", "string"]).columns
        for col in text_cols:
            clean[col] = clean[col].astype("string").str.strip()
        original_rows = len(clean)
        clean = clean.dropna(how="all").reset_index(drop=True)
        numeric = clean.select_dtypes(include=np.number).columns.tolist()
        categorical = clean.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        tabs = st.tabs(["Overview", "Data Quality", "Statistics", "Charts", "Insights", "🤖 Ask AI"])
        with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(clean):,}")
            c2.metric("Columns", f"{len(clean.columns):,}")
            c3.metric("Missing cells", f"{int(clean.isna().sum().sum()):,}")
            c4.metric("Duplicate rows", f"{int(clean.duplicated().sum()):,}")
            st.subheader("Data Preview")
            st.dataframe(clean.head(100), use_container_width=True)
            st.subheader("Column Information")
            info = pd.DataFrame({"Column": clean.columns, "Data Type": [str(clean[c].dtype) for c in clean.columns], "Missing": [int(clean[c].isna().sum()) for c in clean.columns], "Unique": [int(clean[c].nunique(dropna=True)) for c in clean.columns]})
            st.dataframe(info, use_container_width=True)

        with tabs[1]:
            missing = clean.isna().sum().sort_values(ascending=False)
            quality = pd.DataFrame({"Missing": missing, "Missing %": (missing / max(len(clean), 1) * 100).round(2), "Unique": [clean[c].nunique(dropna=True) for c in missing.index]})
            st.subheader("Missing Values")
            st.dataframe(quality, use_container_width=True)
            st.write("Completely blank rows removed:", original_rows - len(clean))
            st.write("Duplicate rows:", int(clean.duplicated().sum()))

        with tabs[2]:
            if numeric:
                st.subheader("Descriptive Statistics")
                st.dataframe(clean[numeric].describe().T, use_container_width=True)
                st.subheader("Correlation Matrix")
                if len(numeric) >= 2:
                    st.dataframe(clean[numeric].corr().round(3), use_container_width=True)
                else:
                    st.info("At least two numeric columns are needed for correlation.")
            else:
                st.info("No numeric columns were detected.")

        with tabs[3]:
            st.subheader("Automatic Visual Analysis")
            if numeric:
                selected_num = st.selectbox("Choose a numeric column", numeric)
                fig, ax = plt.subplots(figsize=(9, 4))
                clean[selected_num].dropna().plot(kind="hist", bins=20, ax=ax)
                ax.set_title(f"Distribution: {selected_num}")
                ax.set_xlabel(selected_num)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
                plt.close(fig)
            if categorical:
                selected_cat = st.selectbox("Choose a categorical column", categorical)
                counts = clean[selected_cat].value_counts(dropna=False).head(15)
                fig, ax = plt.subplots(figsize=(9, 5))
                counts.sort_values().plot(kind="barh", ax=ax)
                ax.set_title(f"Top categories: {selected_cat}")
                ax.set_xlabel("Count")
                st.pyplot(fig)
                plt.close(fig)

        with tabs[4]:
            st.subheader("Automatic Analytical Insights")
            insights = [f"The dataset contains {len(clean):,} records and {len(clean.columns):,} columns."]
            missing_total = int(clean.isna().sum().sum())
            if missing_total:
                worst = clean.isna().sum().sort_values(ascending=False).head(3)
                insights.append("Missing values are concentrated in: " + ", ".join([f"{c} ({int(v)})" for c, v in worst.items() if v > 0]) + ".")
            else:
                insights.append("No missing cells were detected.")
            insights.append(f"{int(clean.duplicated().sum()):,} duplicate rows were detected.")
            for col in numeric[:8]:
                s = clean[col].dropna()
                if len(s):
                    insights.append(f"{col}: mean={s.mean():,.2f}, median={s.median():,.2f}, min={s.min():,.2f}, max={s.max():,.2f}.")
            for item in insights:
                st.write("• " + item)

        with tabs[5]:
            st.subheader("🤖 Ask your AI Data Analyst")
            st.write("Ask questions about the uploaded dataset. The AI receives a compact data profile, statistics, category frequencies, and sample rows.")
            question = st.text_area("Your question", placeholder="Example: What are the most important insights from this dataset?", height=120)
            col_a, col_b = st.columns(2)
            with col_a:
                ask_button = st.button("🤖 Analyze with AI", type="primary")
            with col_b:
                insights_button = st.button("✨ Generate AI Insights")
            if ask_button and question.strip():
                with st.spinner("AI is analyzing your dataset..."):
                    st.markdown(ask_ai(question.strip(), make_profile(clean)))
            if insights_button:
                with st.spinner("Generating AI insights..."):
                    q = "Give me the 10 most important insights from this dataset, including data-quality issues, distributions, notable categories, useful comparisons, and practical next steps. Do not invent information."
                    st.markdown(ask_ai(q, make_profile(clean)))
            st.info("AI features require OPENAI_API_KEY in Streamlit Secrets. Never put the API key in public GitHub code.")

        st.divider()
        st.subheader("Download cleaned data")
        out = BytesIO()
        clean.to_excel(out, index=False)
        out.seek(0)
        st.download_button("⬇️ Download Cleaned Excel", data=out, file_name="cleaned_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Could not analyze this file: {e}")
else:
    st.info("Upload a file above to start the analysis.")
