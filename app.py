
import io, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Data Analyst V5", page_icon="📈", layout="wide")

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
    x["median"]=n.median(); x["missing"]=n.isna().sum(); x["range"]=n.max()-n.min()
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
    c=n.corr(); rows=[]
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

# ---------- V5: Smart Trend & Time-Series ----------
def detect_date_columns(df):
    """Return likely date columns, scored by successful parsing and date-like column names."""
    candidates=[]
    date_words=("date","time","day","month","year","timestamp","created","updated","order","invoice","transaction")
    for c in df.columns:
        s=df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            parsed=s
        else:
            if pd.api.types.is_numeric_dtype(s):
                continue
            parsed=pd.to_datetime(s, errors="coerce")
        valid=parsed.notna().mean() if len(parsed) else 0
        name_score=any(w in str(c).lower() for w in date_words)
        if valid >= 0.70 or (name_score and valid >= 0.40):
            candidates.append((c, float(valid), name_score))
    candidates.sort(key=lambda x:(x[2],x[1]), reverse=True)
    return [x[0] for x in candidates]

def prepare_time_data(df, date_col, value_col, freq):
    x=df[[date_col,value_col]].copy()
    x[date_col]=pd.to_datetime(x[date_col], errors="coerce")
    x[value_col]=pd.to_numeric(x[value_col], errors="coerce")
    x=x.dropna()
    if x.empty: return pd.DataFrame()
    x=x.sort_values(date_col)
    rule={"Daily":"D","Weekly":"W","Monthly":"ME","Yearly":"YE"}[freq]
    # For compatibility with older pandas, fall back from ME/YE.
    try:
        g=x.set_index(date_col)[value_col].resample(rule).sum().dropna()
    except ValueError:
        fallback={"ME":"M","YE":"Y"}[rule] if rule in ("ME","YE") else rule
        g=x.set_index(date_col)[value_col].resample(fallback).sum().dropna()
    out=g.reset_index(name="value")
    out["period"]=out[date_col].dt.strftime(
        "%Y-%m-%d" if freq=="Daily" else ("%Y-%m" if freq=="Monthly" else "%Y")
    )
    out["change"]=out["value"].diff()
    out["growth_pct"]=out["value"].pct_change().replace([np.inf,-np.inf],np.nan)*100
    return out

def trend_summary(ts, value_col):
    if ts.empty: return []
    first=float(ts["value"].iloc[0]); last=float(ts["value"].iloc[-1])
    change=last-first
    pct=(change/first*100) if first!=0 else np.nan
    peak=ts.loc[ts["value"].idxmax()]
    low=ts.loc[ts["value"].idxmin()]
    direction="increased" if change>0 else ("decreased" if change<0 else "remained stable")
    return [
        f"{value_col}: {direction} from {first:,.2f} to {last:,.2f} ({pct:+.2f}% overall change)." if pd.notna(pct)
        else f"{value_col}: changed from {first:,.2f} to {last:,.2f}; percentage change is unavailable because the first value is zero.",
        f"Highest period: {peak['period']} ({peak['value']:,.2f}).",
        f"Lowest period: {low['period']} ({low['value']:,.2f})."
    ]

def linear_forecast(ts, periods=3):
    if len(ts)<3: return pd.DataFrame()
    y=ts["value"].astype(float).to_numpy()
    x=np.arange(len(y),dtype=float)
    slope,intercept=np.polyfit(x,y,1)
    future_x=np.arange(len(y),len(y)+periods,dtype=float)
    pred=intercept+slope*future_x
    last_date=pd.to_datetime(ts.iloc[-1]["period"] + ("-01" if len(str(ts.iloc[-1]["period"]))==7 else ""), errors="coerce")
    if pd.isna(last_date):
        last_date=pd.to_datetime(ts.iloc[-1].iloc[0])
    # Forecast labels based on the selected period spacing.
    date_col=ts.columns[0]
    last_dt=pd.to_datetime(ts.iloc[-1][date_col])
    if len(ts)>1:
        step=pd.to_timedelta(np.median(np.diff(pd.to_datetime(ts[date_col]).astype("int64"))),unit="ns")
    else:
        step=pd.Timedelta(days=30)
    dates=[last_dt + step*(i+1) for i in range(periods)]
    return pd.DataFrame({"forecast_date":dates,"forecast":pred})

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
        r=o.iloc[0]; x.append(f"Most outliers: {r['column']} ({int(r['outliers'])}, {r['outlier_pct']:.2f}%).")
    c=correlations(df)
    if not c.empty:
        r=c.iloc[0]; x.append(f"Strongest numeric relationship: {r['column_1']} vs {r['column_2']} (r={r['correlation']}).")
    for col in df.select_dtypes(exclude=np.number).columns[:6]:
        vc=df[col].value_counts(dropna=True)
        if not vc.empty: x.append(f"Most common {col}: {vc.index[0]} ({int(vc.iloc[0]):,} rows).")
    return x

def business_analysis(df):
    nums=df.select_dtypes(include=np.number)
    return {
        "kpis":{"rows":len(df),"columns":len(df.columns),"missing_cells":int(df.isna().sum().sum()),
                "duplicate_rows":int(df.duplicated().sum()),"numeric_columns":len(nums.columns),
                "categorical_columns":len(df.select_dtypes(exclude=np.number).columns)},
        "numeric_summary":stats(df).to_dict("records"),
        "outliers":outliers(df).to_dict("records"),
        "correlations":correlations(df).head(20).to_dict("records"),
        "category_patterns":category_patterns(df).to_dict("records"),
        "automatic_insights":automatic_insights(df)
    }

def ask_ai(question, context):
    try:
        from openai import OpenAI
        key=st.secrets.get("OPENAI_API_KEY","").strip()
        if not key: return None,"OPENAI_API_KEY is missing from Streamlit Secrets."
        prompt=f"""You are an advanced data analyst.
Use ONLY the supplied computed dataset context. Never invent numbers.
Explain findings clearly and distinguish facts from recommendations.

USER REQUEST:
{question}

COMPUTED DATASET CONTEXT:
{json.dumps(context,default=str)}
"""
        r=OpenAI(api_key=key).responses.create(model="gpt-5.6-luna",input=prompt)
        return r.output_text,None
    except Exception as e: return None,str(e)

# ---------- App ----------
st.title("🤖 AI Data Analyst V5")
st.caption("Upload → analyze → smart trends → forecasting → AI insights → export")

file=st.file_uploader("Upload Excel or CSV",type=["csv","xlsx","xls"])
if not file:
    st.info("Upload a dataset to begin."); st.stop()

try:
    df=pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    df=clean(df)
except Exception as e:
    st.error(f"Could not read file: {e}"); st.stop()

st.success(f"Loaded: {file.name}")
t=st.tabs(["Overview","Data Quality","Statistics","Charts","Insights","🎯 Business Analysis","📈 V5 Trends","Ask AI","Report"])

with t[0]:
    a,b,c,d=st.columns(4)
    a.metric("Rows",f"{len(df):,}"); b.metric("Columns",f"{len(df.columns):,}")
    c.metric("Missing",f"{int(df.isna().sum().sum()):,}"); d.metric("Duplicates",f"{int(df.duplicated().sum()):,}")
    st.dataframe(df.head(100),use_container_width=True); st.dataframe(profile(df),use_container_width=True)

with t[1]:
    m=pd.DataFrame({"column":df.columns,"missing":[int(df[c].isna().sum()) for c in df.columns]})
    m["missing_pct"]=(m["missing"]/max(len(df),1)*100).round(2)
    st.subheader("Missing Values"); st.dataframe(m.sort_values("missing",ascending=False),use_container_width=True)
    st.subheader("Outliers (IQR)"); st.dataframe(outliers(df),use_container_width=True)

with t[2]:
    st.subheader("Numeric Statistics"); st.dataframe(stats(df),use_container_width=True)
    st.subheader("Correlations"); st.dataframe(correlations(df).head(20),use_container_width=True)

with t[3]:
    nums=list(df.select_dtypes(include=np.number).columns)
    if nums:
        col=st.selectbox("Numeric column",nums,key="hist_col")
        fig,ax=plt.subplots(); ax.hist(df[col].dropna(),bins=30); ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col); ax.set_ylabel("Frequency"); st.pyplot(fig); plt.close(fig)
    cats=list(df.select_dtypes(exclude=np.number).columns)
    if cats:
        col=st.selectbox("Categorical column",cats,key="cat_col"); vc=df[col].value_counts().head(15)
        fig,ax=plt.subplots(); vc.plot(kind="bar",ax=ax); ax.set_title(f"Top categories: {col}")
        ax.set_ylabel("Count"); st.pyplot(fig); plt.close(fig)

with t[4]:
    st.subheader("Automatic Insights")
    for i,x in enumerate(automatic_insights(df),1): st.write(f"**{i}.** {x}")

with t[5]:
    st.subheader("🎯 Automatic Business Analysis")
    ba=business_analysis(df); k=ba["kpis"]
    a,b,c,d,e,f=st.columns(6)
    a.metric("Rows",f"{k['rows']:,}"); b.metric("Columns",f"{k['columns']:,}"); c.metric("Missing",f"{k['missing_cells']:,}")
    d.metric("Duplicates",f"{k['duplicate_rows']:,}"); e.metric("Numeric",k["numeric_columns"]); f.metric("Categorical",k["categorical_columns"])
    st.markdown("### Key Findings")
    for i,x in enumerate(ba["automatic_insights"],1): st.write(f"**{i}.** {x}")
    st.markdown("### Outlier Signals"); st.dataframe(outliers(df),use_container_width=True)
    st.markdown("### Strong Relationships"); st.dataframe(correlations(df).head(10),use_container_width=True)
    st.markdown("### Top Category Patterns"); st.dataframe(category_patterns(df).head(30),use_container_width=True)
    if st.button("🤖 Generate Business Recommendations",type="primary"):
        with st.spinner("Generating recommendations..."):
            ans,err=ask_ai("Based on these computed results, give 6 practical recommendations. Cite exact evidence/numbers and do not invent context.",ba)
        if err: st.error(err)
        else: st.markdown(ans)

with t[6]:
    st.subheader("📈 Smart Trend & Time-Series Analysis")
    date_candidates=detect_date_columns(df)
    if not date_candidates:
        st.warning("No reliable date column was detected. Make sure your dataset contains a Date/Time column.")
    else:
        st.success("Detected date column(s): " + ", ".join(date_candidates))
        date_col=st.selectbox("Date column",date_candidates)
        numeric_cols=list(df.select_dtypes(include=np.number).columns)
        if not numeric_cols:
            st.warning("A numeric column is required for trend analysis.")
        else:
            value_col=st.selectbox("Measure / value column",numeric_cols)
            freq=st.selectbox("Trend period",["Daily","Weekly","Monthly","Yearly"],index=2)
            ts=prepare_time_data(df,date_col,value_col,freq)
            if ts.empty:
                st.warning("Not enough valid date/value data after conversion.")
            else:
                st.markdown("### Trend Summary")
                for x in trend_summary(ts,value_col): st.write("• "+x)
                a,b,c=st.columns(3)
                a.metric("Periods",f"{len(ts):,}")
                a.metric("Highest",f"{ts['value'].max():,.2f}")
                b.metric("Lowest",f"{ts['value'].min():,.2f}")
                c.metric("Latest",f"{ts['value'].iloc[-1]:,.2f}")
                st.markdown("### Trend Table")
                st.dataframe(ts,use_container_width=True)

                fig,ax=plt.subplots()
                ax.plot(pd.to_datetime(ts[date_col]),ts["value"],marker="o")
                ax.set_title(f"{freq} Trend — {value_col}")
                ax.set_xlabel("Date"); ax.set_ylabel(value_col); ax.grid(alpha=.25)
                st.pyplot(fig); plt.close(fig)

                st.markdown("### 📊 Growth / Decline")
                gd=ts[[date_col,"period","value","change","growth_pct"]].copy()
                gd["growth_pct"]=gd["growth_pct"].round(2)
                st.dataframe(gd,use_container_width=True)

                st.markdown("### 🔮 Forecast")
                if len(ts)<3:
                    st.info("At least 3 time periods are required for the basic forecast.")
                else:
                    horizon=st.slider("Forecast periods",1,12,3)
                    fc=linear_forecast(ts,horizon)
                    st.dataframe(fc,use_container_width=True)
                    fig,ax=plt.subplots()
                    ax.plot(pd.to_datetime(ts[date_col]),ts["value"],marker="o",label="Historical")
                    ax.plot(fc["forecast_date"],fc["forecast"],marker="o",linestyle="--",label="Forecast")
                    ax.set_title(f"{freq} Trend + Forecast — {value_col}")
                    ax.set_xlabel("Date"); ax.set_ylabel(value_col); ax.legend(); ax.grid(alpha=.25)
                    st.pyplot(fig); plt.close(fig)

                if st.button("🤖 Explain This Trend with AI"):
                    context={"date_column":date_col,"value_column":value_col,"frequency":freq,
                             "trend_summary":trend_summary(ts,value_col),
                             "trend_table":ts.to_dict("records")}
                    if len(ts)>=3: context["forecast"]=linear_forecast(ts,3).to_dict("records")
                    with st.spinner("Generating trend explanation..."):
                        ans,err=ask_ai("Explain the trend in simple language. Mention overall direction, highest/lowest periods, growth/decline and forecast if supplied. Use only the computed data.",context)
                    if err: st.error(err)
                    else: st.markdown(ans)

with t[7]:
    q=st.text_area("Ask your AI Data Analyst",placeholder="Which findings are most important and why?",key="ask_q")
    if st.button("🤖 Analyze with AI",type="primary",key="ask_btn"):
        if not q.strip(): st.warning("Write a question first.")
        else:
            with st.spinner("Analyzing..."):
                ans,err=ask_ai(q,business_analysis(df))
            if err: st.error(err)
            else: st.markdown(ans)
    if st.button("✨ Generate AI Insights",key="insight_btn"):
        with st.spinner("Generating..."):
            ans,err=ask_ai("Give 8 important insights from this dataset. Include data quality, outliers, relationships and category patterns with exact numbers.",business_analysis(df))
        if err: st.error(err)
        else: st.markdown(ans)

with t[8]:
    report=["# AI Data Analyst V5 Report","",f"File: {file.name}",""]+["- "+x for x in automatic_insights(df)]
    date_candidates=detect_date_columns(df)
    if date_candidates and len(df.select_dtypes(include=np.number).columns):
        report += ["","## Detected Time-Series","- Date column: "+date_candidates[0]]
    text="\n".join(report)
    st.text_area("Report Preview",text,height=250)
    st.download_button("⬇️ Download Report",text,"AI_Data_Analyst_V5_Report.txt")
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        df.to_excel(w,index=False,sheet_name="Cleaned Data")
        profile(df).to_excel(w,index=False,sheet_name="Profile")
        stats(df).to_excel(w,index=False,sheet_name="Statistics")
        outliers(df).to_excel(w,index=False,sheet_name="Outliers")
        correlations(df).to_excel(w,index=False,sheet_name="Correlations")
        category_patterns(df).to_excel(w,index=False,sheet_name="Categories")
        if date_candidates and len(df.select_dtypes(include=np.number).columns):
            tc=prepare_time_data(df,date_candidates[0],df.select_dtypes(include=np.number).columns[0],"Monthly")
            if not tc.empty: tc.to_excel(w,index=False,sheet_name="Monthly Trend")
    buf.seek(0)
    st.download_button("⬇️ Download Analysis Excel",buf,"AI_Data_Analyst_V5_Results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
