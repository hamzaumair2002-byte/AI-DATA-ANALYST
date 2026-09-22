
import io, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Data Analyst", page_icon="🤖", layout="wide")

def clean(df):
    df=df.dropna(how="all").copy()
    df.columns=[str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype=="object":
            df[c]=df[c].astype(str).str.strip().replace({"":"", "nan":np.nan, "None":np.nan})
    return df

def profile(df):
    return pd.DataFrame({
        "column":df.columns,
        "dtype":[str(df[c].dtype) for c in df.columns],
        "missing":[int(df[c].isna().sum()) for c in df.columns],
        "unique":[int(df[c].nunique(dropna=True)) for c in df.columns]
    })

def stats(df):
    n=df.select_dtypes(include=np.number)
    if n.empty: return pd.DataFrame()
    x=n.describe().T
    x["median"]=n.median()
    x["missing"]=n.isna().sum()
    x["range"]=n.max()-n.min()
    return x.reset_index().rename(columns={"index":"column"})

def outliers(df):
    rows=[]
    for c in df.select_dtypes(include=np.number):
        s=df[c].dropna()
        if len(s)<4: continue
        q1,q3=s.quantile([.25,.75]); iqr=q3-q1
        n=0 if iqr==0 else int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum())
        rows.append({"column":c,"outliers":n,"outlier_pct":round(n/len(s)*100,2)})
    return pd.DataFrame(rows).sort_values("outliers",ascending=False) if rows else pd.DataFrame()

def correlations(df):
    n=df.select_dtypes(include=np.number)
    if n.shape[1]<2: return pd.DataFrame()
    c=n.corr()
    rows=[]
    for i,a in enumerate(c.columns):
        for b in c.columns[i+1:]:
            if pd.notna(c.loc[a,b]):
                rows.append({"column_1":a,"column_2":b,"correlation":round(float(c.loc[a,b]),3)})
    return pd.DataFrame(rows).sort_values("correlation",key=lambda x:x.abs(),ascending=False) if rows else pd.DataFrame()

def category_patterns(df, limit=8):
    rows=[]
    for c in df.select_dtypes(exclude=np.number).columns:
        vc=df[c].value_counts(dropna=True).head(limit)
        for value,count in vc.items():
            rows.append({"column":c,"value":str(value),"count":int(count),"share_pct":round(count/len(df)*100,2)})
    return pd.DataFrame(rows)

def automatic_insights(df):
    x=[
        f"Dataset contains {len(df):,} rows and {len(df.columns):,} columns.",
        f"Missing cells: {int(df.isna().sum().sum()):,}.",
        f"Duplicate rows: {int(df.duplicated().sum()):,}."
    ]
    s=stats(df)
    if not s.empty:
        r=s.sort_values("mean",ascending=False).iloc[0]
        x.append(f"Highest numeric mean: {r['column']} = {r['mean']:,.2f}.")
    o=outliers(df)
    if not o.empty and o.iloc[0]["outliers"]>0:
        r=o.iloc[0]
        x.append(f"Most outliers: {r['column']} ({int(r['outliers'])}, {r['outlier_pct']:.2f}%).")
    c=correlations(df)
    if not c.empty:
        r=c.iloc[0]
        x.append(f"Strongest numeric relationship: {r['column_1']} vs {r['column_2']} (r={r['correlation']}).")
    for col in df.select_dtypes(exclude=np.number).columns[:6]:
        vc=df[col].value_counts(dropna=True)
        if not vc.empty:
            x.append(f"Most common {col}: {vc.index[0]} ({int(vc.iloc[0]):,} rows).")
    return x

def business_analysis(df):
    result = {}
    nums=df.select_dtypes(include=np.number)
    result["kpis"] = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": len(nums.columns),
        "categorical_columns": len(df.select_dtypes(exclude=np.number).columns)
    }
    result["numeric_summary"]=stats(df).to_dict("records")
    result["outliers"]=outliers(df).to_dict("records")
    result["correlations"]=correlations(df).head(20).to_dict("records")
    result["category_patterns"]=category_patterns(df).to_dict("records")
    result["automatic_insights"]=automatic_insights(df)
    return result

def ask_ai(question, context):
    try:
        from openai import OpenAI
        key=st.secrets.get("OPENAI_API_KEY","").strip()
        if not key:
            return None,"OPENAI_API_KEY is missing from Streamlit Secrets."
        prompt=f"""You are an advanced data analyst.
Use ONLY the supplied computed dataset context. Never invent numbers.
Explain findings clearly, distinguish facts from recommendations, and give exact numbers where available.

USER REQUEST:
{question}

COMPUTED DATASET CONTEXT:
{json.dumps(context,default=str)}
"""
        r=OpenAI(api_key=key).responses.create(model="gpt-5.6-luna",input=prompt)
        return r.output_text,None
    except Exception as e:
        return None,str(e)

st.title("🤖 AI Data Analyst")
st.caption("Upload → analyze → AI insights → recommendations → export")

file=st.file_uploader("Upload Excel or CSV",type=["csv","xlsx","xls"])
if not file:
    st.info("Upload a dataset to begin.")
    st.stop()

try:
    df=pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    df=clean(df)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.success(f"Loaded: {file.name}")
t=st.tabs(["Overview","Data Quality","Statistics","Charts","Insights","🎯 Business Analysis","Ask AI","Report"])

with t[0]:
    a,b,c,d=st.columns(4)
    a.metric("Rows",f"{len(df):,}")
    b.metric("Columns",f"{len(df.columns):,}")
    c.metric("Missing",f"{int(df.isna().sum().sum()):,}")
    d.metric("Duplicates",f"{int(df.duplicated().sum()):,}")
    st.dataframe(df.head(100),use_container_width=True)
    st.dataframe(profile(df),use_container_width=True)

with t[1]:
    m=pd.DataFrame({"column":df.columns,"missing":[int(df[c].isna().sum()) for c in df.columns]})
    m["missing_pct"]=(m["missing"]/max(len(df),1)*100).round(2)
    st.subheader("Missing Values")
    st.dataframe(m.sort_values("missing",ascending=False),use_container_width=True)
    st.subheader("Outliers (IQR)")
    st.dataframe(outliers(df),use_container_width=True)

with t[2]:
    st.subheader("Numeric Statistics")
    st.dataframe(stats(df),use_container_width=True)
    st.subheader("Correlations")
    st.dataframe(correlations(df).head(20),use_container_width=True)

with t[3]:
    nums=list(df.select_dtypes(include=np.number).columns)
    if nums:
        col=st.selectbox("Numeric column",nums)
        fig,ax=plt.subplots()
        ax.hist(df[col].dropna(),bins=30)
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col); ax.set_ylabel("Frequency")
        st.pyplot(fig); plt.close(fig)
    cats=list(df.select_dtypes(exclude=np.number).columns)
    if cats:
        col=st.selectbox("Categorical column",cats)
        vc=df[col].value_counts().head(15)
        fig,ax=plt.subplots()
        vc.plot(kind="bar",ax=ax)
        ax.set_title(f"Top categories: {col}")
        ax.set_ylabel("Count")
        st.pyplot(fig); plt.close(fig)

with t[4]:
    st.subheader("Automatic Insights")
    for i,x in enumerate(automatic_insights(df),1):
        st.write(f"**{i}.** {x}")

with t[5]:
    st.subheader("🎯 Automatic Business Analysis")
    ba=business_analysis(df)
    k=ba["kpis"]
    a,b,c,d,e,f=st.columns(6)
    a.metric("Rows",f"{k['rows']:,}")
    b.metric("Columns",f"{k['columns']:,}")
    c.metric("Missing",f"{k['missing_cells']:,}")
    d.metric("Duplicates",f"{k['duplicate_rows']:,}")
    e.metric("Numeric",k["numeric_columns"])
    f.metric("Categorical",k["categorical_columns"])

    st.markdown("### Key Findings")
    for i,x in enumerate(ba["automatic_insights"],1):
        st.write(f"**{i}.** {x}")

    st.markdown("### Outlier Signals")
    st.dataframe(outliers(df),use_container_width=True)

    st.markdown("### Strong Relationships")
    st.dataframe(correlations(df).head(10),use_container_width=True)

    st.markdown("### Top Category Patterns")
    st.dataframe(category_patterns(df).head(30),use_container_width=True)

    if st.button("🤖 Generate Business Recommendations",type="primary"):
        with st.spinner("Generating recommendations..."):
            ans,err=ask_ai(
                "Based on these computed results, give 6 practical recommendations. "
                "For each recommendation, cite the exact evidence/number from the data. "
                "Do not invent business context that is not provided.",
                ba
            )
        if err: st.error(err)
        else: st.markdown(ans)

with t[6]:
    q=st.text_area("Ask your AI Data Analyst",placeholder="Which findings are most important and why?")
    if st.button("🤖 Analyze with AI",type="primary"):
        if not q.strip():
            st.warning("Write a question first.")
        else:
            with st.spinner("Analyzing..."):
                ans,err=ask_ai(q,business_analysis(df))
            if err: st.error(err)
            else: st.markdown(ans)

    if st.button("✨ Generate AI Insights"):
        with st.spinner("Generating..."):
            ans,err=ask_ai(
                "Give 8 important insights from this dataset. Include data quality, outliers, "
                "relationships and category patterns with exact numbers.",
                business_analysis(df)
            )
        if err: st.error(err)
        else: st.markdown(ans)

with t[7]:
    report=["# AI Data Analyst Report","",f"File: {file.name}",""]+["- "+x for x in automatic_insights(df)]
    text="\n".join(report)
    st.text_area("Report Preview",text,height=250)
    st.download_button("⬇️ Download Report",text,"AI_Data_Analysis_Report.txt")
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        df.to_excel(w,index=False,sheet_name="Cleaned Data")
        profile(df).to_excel(w,index=False,sheet_name="Profile")
        stats(df).to_excel(w,index=False,sheet_name="Statistics")
        outliers(df).to_excel(w,index=False,sheet_name="Outliers")
        correlations(df).to_excel(w,index=False,sheet_name="Correlations")
        category_patterns(df).to_excel(w,index=False,sheet_name="Categories")
    buf.seek(0)
    st.download_button("⬇️ Download Analysis Excel",buf,"AI_Data_Analyst_Results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
