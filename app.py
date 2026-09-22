
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
    return x.reset_index().rename(columns={"index":"column"})

def outliers(df):
    rows=[]
    for c in df.select_dtypes(include=np.number):
        s=df[c].dropna()
        if len(s)<4: continue
        q1,q3=s.quantile([.25,.75]); iqr=q3-q1
        n=0 if iqr==0 else int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum())
        rows.append({"column":c,"outliers":n,"outlier_pct":round(n/len(s)*100,2)})
    return pd.DataFrame(rows)

def correlations(df):
    n=df.select_dtypes(include=np.number)
    if n.shape[1]<2: return pd.DataFrame()
    c=n.corr()
    rows=[]
    for i,a in enumerate(c.columns):
        for b in c.columns[i+1:]:
            if pd.notna(c.loc[a,b]):
                rows.append({"column_1":a,"column_2":b,"correlation":round(float(c.loc[a,b]),3)})
    return pd.DataFrame(rows).sort_values("correlation",key=lambda x:x.abs(),ascending=False)

def insights(df):
    x=[
        f"Dataset has {len(df):,} rows and {len(df.columns):,} columns.",
        f"Missing cells: {int(df.isna().sum().sum()):,}.",
        f"Duplicate rows: {int(df.duplicated().sum()):,}."
    ]
    s=stats(df)
    if not s.empty:
        r=s.sort_values("mean",ascending=False).iloc[0]
        x.append(f"Highest numeric mean is {r['column']} at {r['mean']:,.2f}.")
    o=outliers(df)
    if not o.empty and o["outliers"].max()>0:
        r=o.sort_values("outliers",ascending=False).iloc[0]
        x.append(f"Most outliers are in {r['column']}: {int(r['outliers'])} ({r['outlier_pct']:.2f}%).")
    c=correlations(df)
    if not c.empty:
        r=c.iloc[0]
        x.append(f"Strongest numeric relationship: {r['column_1']} vs {r['column_2']} (r={r['correlation']}).")
    for col in df.select_dtypes(exclude=np.number).columns[:5]:
        v=df[col].value_counts(dropna=True)
        if not v.empty: x.append(f"Most common {col}: {v.index[0]} ({int(v.iloc[0]):,} rows).")
    return x

def ai_answer(question,df):
    from openai import OpenAI
    key=st.secrets.get("OPENAI_API_KEY","").strip()
    if not key: return None,"OPENAI_API_KEY is missing from Streamlit Secrets."
    context={
        "shape":[len(df),len(df.columns)],
        "columns":list(map(str,df.columns)),
        "profile":profile(df).to_dict("records"),
        "statistics":stats(df).round(4).to_dict("records"),
        "outliers":outliers(df).to_dict("records"),
        "correlations":correlations(df).head(20).to_dict("records"),
        "automatic_insights":insights(df),
        "sample_rows":df.head(30).to_dict("records")
    }
    prompt=("You are an advanced data analyst. Answer only from the supplied dataset context. "
            "Never invent numbers. Give practical findings and exact numbers when available.\n\n"
            "QUESTION:\n"+question+"\n\nDATA:\n"+json.dumps(context,default=str))
    try:
        r=OpenAI(api_key=key).responses.create(model="gpt-5.6-luna",input=prompt)
        return r.output_text,None
    except Exception as e:
        return None,str(e)

st.title("🤖 AI Data Analyst")
st.caption("Upload → clean → analyze → ask AI → export")

file=st.file_uploader("Upload Excel or CSV",type=["csv","xlsx","xls"])
if not file:
    st.info("Upload a dataset to begin.")
    st.stop()

try:
    df=pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    df=clean(df)
except Exception as e:
    st.error(f"Could not read file: {e}"); st.stop()

st.success(f"Loaded: {file.name}")
t=st.tabs(["Overview","Data Quality","Statistics","Charts","Insights","Ask AI","Report"])

with t[0]:
    a,b,c,d=st.columns(4)
    a.metric("Rows",f"{len(df):,}"); b.metric("Columns",f"{len(df.columns):,}")
    c.metric("Missing",f"{int(df.isna().sum().sum()):,}"); d.metric("Duplicates",f"{int(df.duplicated().sum()):,}")
    st.dataframe(df.head(100),use_container_width=True)
    st.dataframe(profile(df),use_container_width=True)

with t[1]:
    st.subheader("Missing values")
    m=pd.DataFrame({"column":df.columns,"missing":[int(df[c].isna().sum()) for c in df.columns]})
    m["missing_pct"]=(m["missing"]/max(len(df),1)*100).round(2)
    st.dataframe(m.sort_values("missing",ascending=False),use_container_width=True)
    st.subheader("Outliers")
    st.dataframe(outliers(df),use_container_width=True)

with t[2]:
    st.subheader("Numeric statistics"); st.dataframe(stats(df),use_container_width=True)
    st.subheader("Correlations"); st.dataframe(correlations(df).head(20),use_container_width=True)

with t[3]:
    nums=list(df.select_dtypes(include=np.number).columns)
    if nums:
        col=st.selectbox("Numeric column",nums)
        fig,ax=plt.subplots(); ax.hist(df[col].dropna(),bins=30); ax.set_title(col); st.pyplot(fig); plt.close(fig)
    cats=list(df.select_dtypes(exclude=np.number).columns)
    if cats:
        col=st.selectbox("Category column",cats)
        vc=df[col].value_counts().head(15)
        fig,ax=plt.subplots(); vc.plot(kind="bar",ax=ax); ax.set_title(col); st.pyplot(fig); plt.close(fig)

with t[4]:
    for i,x in enumerate(insights(df),1): st.write(f"**{i}.** {x}")

with t[5]:
    q=st.text_area("Your question",placeholder="Give me 5 important insights with numbers.")
    if st.button("🤖 Analyze with AI",type="primary"):
        if not q.strip(): st.warning("Write a question first.")
        else:
            with st.spinner("Analyzing..."):
                ans,err=ai_answer(q,df)
            st.error(err) if err else st.markdown(ans)
    if st.button("✨ Generate AI Insights"):
        with st.spinner("Generating..."):
            ans,err=ai_answer("Give 8 important insights, including data quality, outliers, relationships and category patterns. Use exact numbers.",df)
        st.error(err) if err else st.markdown(ans)

with t[6]:
    text="# AI Data Analyst Report\n\n"+"\n".join("- "+x for x in insights(df))
    st.text_area("Report",text,height=250)
    st.download_button("Download Report",text,"analysis_report.txt")
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        df.to_excel(w,index=False,sheet_name="Cleaned Data")
        profile(df).to_excel(w,index=False,sheet_name="Profile")
        stats(df).to_excel(w,index=False,sheet_name="Statistics")
        outliers(df).to_excel(w,index=False,sheet_name="Outliers")
        correlations(df).to_excel(w,index=False,sheet_name="Correlations")
    buf.seek(0)
    st.download_button("Download Analysis Excel",buf,"AI_Data_Analyst_Results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
