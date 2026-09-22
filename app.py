
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

st.title("📊 AI Data Analyst")
st.write("Upload Excel or CSV data and get an automatic advanced analysis.")

uploaded = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])

if uploaded:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.success(f"Loaded: {uploaded.name}")

        # Basic cleaning
        clean = df.copy()
        text_cols = clean.select_dtypes(include=["object", "string"]).columns
        for col in text_cols:
            clean[col] = clean[col].astype("string").str.strip()

        original_rows = len(clean)
        clean = clean.dropna(how="all").reset_index(drop=True)

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Overview", "Data Quality", "Statistics", "Charts", "Insights"]
        )

        with tab1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(clean):,}")
            c2.metric("Columns", f"{len(clean.columns):,}")
            c3.metric("Missing cells", f"{int(clean.isna().sum().sum()):,}")
            c4.metric("Duplicate rows", f"{int(clean.duplicated().sum()):,}")

            st.subheader("Data Preview")
            st.dataframe(clean.head(100), use_container_width=True)

            st.subheader("Column Information")
            info = pd.DataFrame({
                "Column": clean.columns,
                "Data Type": [str(clean[c].dtype) for c in clean.columns],
                "Missing": [int(clean[c].isna().sum()) for c in clean.columns],
                "Unique": [int(clean[c].nunique(dropna=True)) for c in clean.columns],
            })
            st.dataframe(info, use_container_width=True)

        with tab2:
            missing = clean.isna().sum().sort_values(ascending=False)
            quality = pd.DataFrame({
                "Missing": missing,
                "Missing %": (missing / len(clean) * 100).round(2),
                "Unique": [clean[c].nunique(dropna=True) for c in missing.index]
            })
            st.subheader("Missing Values")
            st.dataframe(quality, use_container_width=True)

            st.write("Completely blank rows removed:", original_rows - len(clean))
            st.write("Duplicate rows:", int(clean.duplicated().sum()))

        numeric = clean.select_dtypes(include=np.number).columns.tolist()
        categorical = clean.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        with tab3:
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

        with tab4:
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

        with tab5:
            st.subheader("Automatic Analytical Insights")

            insights = []

            if len(clean) > 0:
                insights.append(f"The dataset contains {len(clean):,} records and {len(clean.columns):,} columns.")

            missing_total = int(clean.isna().sum().sum())
            if missing_total:
                worst = clean.isna().sum().sort_values(ascending=False).head(3)
                insights.append(
                    "Missing values are concentrated in: " +
                    ", ".join([f"{c} ({int(v)})" for c, v in worst.items() if v > 0]) + "."
                )
            else:
                insights.append("No missing cells were detected.")

            duplicates = int(clean.duplicated().sum())
            insights.append(f"{duplicates:,} duplicate rows were detected.")

            for col in numeric[:8]:
                s = clean[col].dropna()
                if len(s):
                    insights.append(
                        f"{col}: mean={s.mean():,.2f}, median={s.median():,.2f}, "
                        f"min={s.min():,.2f}, max={s.max():,.2f}."
                    )

            for item in insights:
                st.write("• " + item)

        # Downloads
        st.divider()
        st.subheader("Download cleaned data")
        out = BytesIO()
        clean.to_excel(out, index=False)
        out.seek(0)
        st.download_button(
            "⬇️ Download Cleaned Excel",
            data=out,
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Could not analyze this file: {e}")
else:
    st.info("Upload a file above to start the analysis.")
