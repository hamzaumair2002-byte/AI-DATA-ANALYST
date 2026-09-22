
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simple AI Data Analyst", page_icon="📊", layout="wide")

# -------------------- CLEANING --------------------
MISSING_WORDS = ["", " ", "na", "n/a", "nan", "null", "none", "-", "--", "missing"]

def clean_data(df):
    df = df.copy()
    original_rows, original_cols = df.shape

    # Clean column names
    df.columns = (
        pd.Index(df.columns).astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Remove completely blank rows/columns
    blank_rows = int(df.isna().all(axis=1).sum())
    blank_cols = int(df.isna().all(axis=0).sum())
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Clean text and standard missing values
    for c in df.columns:
        if df[c].dtype == "object":
            s = df[c].astype("string").str.strip()
            lower = s.str.lower()
            s = s.mask(lower.isin(MISSING_WORDS))
            df[c] = s

    # Remove exact duplicate rows
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # Try numeric conversion only when most non-missing values look numeric
    for c in df.columns:
        if df[c].dtype == "object" or str(df[c].dtype) == "string":
            s = df[c]
            nonnull = s.dropna()
            if len(nonnull) >= 3:
                converted = pd.to_numeric(
                    s.astype("string").str.replace(",", "", regex=False).str.replace("%", "", regex=False),
                    errors="coerce"
                )
                success = converted.notna().sum() / max(nonnull.notna().sum(), 1)
                if success >= 0.85:
                    df[c] = converted

    return df, {
        "original_rows": original_rows,
        "original_columns": original_cols,
        "blank_rows_removed": blank_rows,
        "blank_columns_removed": blank_cols,
        "duplicates_removed": duplicates,
        "final_rows": len(df),
        "final_columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum())
    }

def quality_table(df):
    out = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Missing %": [(df[c].isna().mean()*100).round(2) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })
    return out

# -------------------- PIVOTS --------------------
def numeric_columns(df):
    return list(df.select_dtypes(include=np.number).columns)

def categorical_columns(df):
    return list(df.select_dtypes(exclude=np.number).columns)

def make_pivot(df, category, value, agg):
    x = df[[category, value]].copy()
    x[value] = pd.to_numeric(x[value], errors="coerce")
    x = x.dropna(subset=[category, value])
    if x.empty:
        return pd.DataFrame()
    if agg == "Sum":
        p = x.groupby(category, as_index=False)[value].sum()
    elif agg == "Average":
        p = x.groupby(category, as_index=False)[value].mean()
    elif agg == "Count":
        p = x.groupby(category, as_index=False)[value].count()
    elif agg == "Maximum":
        p = x.groupby(category, as_index=False)[value].max()
    else:
        p = x.groupby(category, as_index=False)[value].min()
    return p.sort_values(value, ascending=False)

def date_columns(df):
    result = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            result.append(c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        parsed = pd.to_datetime(df[c], errors="coerce")
        if len(df) and parsed.notna().mean() >= 0.75:
            result.append(c)
    return result

def time_pivot(df, date_col, value_col, period):
    x = df[[date_col, value_col]].copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x[value_col] = pd.to_numeric(x[value_col], errors="coerce")
    x = x.dropna()
    if x.empty:
        return pd.DataFrame()
    rule = {"Daily":"D", "Weekly":"W", "Monthly":"ME", "Yearly":"YE"}[period]
    try:
        y = x.set_index(date_col)[value_col].resample(rule).sum().dropna()
    except ValueError:
        y = x.set_index(date_col)[value_col].resample({"ME":"M","YE":"Y"}.get(rule, rule)).sum().dropna()
    return y.reset_index(name=value_col)

# -------------------- UI --------------------
st.title("📊 Simple AI Data Analyst")
st.caption("A simple workflow: 1) Clean data → 2) Create important Pivot Tables → 3) Make Charts → 4) Understand the results")

file = st.file_uploader("📁 Step 1 — Upload your Excel or CSV file", type=["csv", "xlsx", "xls"])

if not file:
    st.info("Upload your dataset to start. You will see the analysis in simple steps.")
    st.stop()

try:
    raw = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

cleaned, cleaning_info = clean_data(raw)

tabs = st.tabs([
    "🧹 1. Clean Data",
    "📋 2. Pivot Tables",
    "📊 3. Charts",
    "📅 4. Time Trends",
    "💡 5. Simple Insights",
    "⬇️ 6. Export"
])

# -------------------- CLEAN --------------------
with tabs[0]:
    st.header("🧹 Step 1 — Data Cleaning")
    st.write("**Goal:** make the dataset ready for analysis. The original file is never changed.")

    a,b,c,d = st.columns(4)
    a.metric("Original rows", f"{cleaning_info['original_rows']:,}")
    b.metric("Duplicates removed", f"{cleaning_info['duplicates_removed']:,}")
    c.metric("Blank rows removed", f"{cleaning_info['blank_rows_removed']:,}")
    d.metric("Missing cells left", f"{cleaning_info['missing_cells']:,}")

    st.subheader("What was cleaned?")
    st.write("✅ Column names were trimmed")
    st.write("✅ Completely blank rows/columns removed")
    st.write("✅ Extra spaces removed from text")
    st.write("✅ Common missing-value labels (NA, N/A, null, -, etc.) converted to blanks")
    st.write("✅ Exact duplicate rows removed")
    st.write("✅ Numeric-looking columns converted to numbers")

    st.subheader("Cleaned data preview")
    st.dataframe(cleaned.head(100), use_container_width=True)

    st.subheader("Data quality after cleaning")
    st.dataframe(quality_table(cleaned), use_container_width=True)

    missing = quality_table(cleaned)
    missing = missing[missing["Missing"] > 0]
    if not missing.empty:
        st.warning("Some values are still missing. They are NOT automatically deleted because sometimes missing values are meaningful.")
        st.dataframe(missing, use_container_width=True)
    else:
        st.success("Excellent — no missing cells remain.")

# -------------------- PIVOTS --------------------
with tabs[1]:
    st.header("📋 Step 2 — Important Pivot Tables")
    st.write("**What is a Pivot Table?** It summarizes many rows into a simple answer such as “sales by product” or “quantity by city”.")

    cats = categorical_columns(cleaned)
    nums = numeric_columns(cleaned)

    if not cats or not nums:
        st.warning("A useful pivot normally needs at least one category column and one numeric column.")
    else:
        st.subheader("Pivot 1 — Summary by Category")
        c1,c2,c3 = st.columns(3)
        cat = c1.selectbox("Group by", cats)
        val = c2.selectbox("Calculate", nums)
        agg = c3.selectbox("Calculation", ["Sum", "Average", "Count", "Maximum", "Minimum"])
        p1 = make_pivot(cleaned, cat, val, agg)
        if not p1.empty:
            st.dataframe(p1, use_container_width=True)
            st.caption(f"Meaning: this table shows {agg.lower()} of **{val}** for every **{cat}**.")

        st.subheader("Pivot 2 — Top 10")
        top = p1.head(10) if not p1.empty else pd.DataFrame()
        if not top.empty:
            st.dataframe(top, use_container_width=True)
            st.caption("This quickly shows the biggest categories.")

        st.subheader("Pivot 3 — Category Count")
        counts = cleaned[cat].value_counts(dropna=True).head(15).reset_index()
        counts.columns = [cat, "Count"]
        st.dataframe(counts, use_container_width=True)
        st.caption(f"Meaning: how many records belong to each **{cat}**.")

# -------------------- CHARTS --------------------
with tabs[2]:
    st.header("📊 Step 3 — Charts")
    st.write("Charts are created from the same summaries above, so the chart is easy to understand.")

    cats = categorical_columns(cleaned)
    nums = numeric_columns(cleaned)

    if cats and nums:
        cat = st.selectbox("Choose category", cats, key="chart_cat")
        val = st.selectbox("Choose numeric value", nums, key="chart_val")
        agg = st.selectbox("How to calculate", ["Sum", "Average", "Count"], key="chart_agg")
        p = make_pivot(cleaned, cat, val, agg).head(15)

        if not p.empty:
            fig, ax = plt.subplots(figsize=(10,5))
            ax.bar(p[cat].astype(str), p[val])
            ax.set_title(f"{agg} of {val} by {cat}")
            ax.set_xlabel(cat)
            ax.set_ylabel(f"{agg} of {val}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.info("How to read it: taller bars mean a larger value. Compare the categories from left to right.")

# -------------------- TIME --------------------
with tabs[3]:
    st.header("📅 Step 4 — Time Trends")
    st.write("If your data has a date column, this section shows how a value changes over time.")

    dates = date_columns(cleaned)
    nums = numeric_columns(cleaned)

    if not dates:
        st.info("No clear date column was detected. You can still use the Pivot Tables and Charts.")
    elif not nums:
        st.warning("A numeric column is needed for a time trend.")
    else:
        c1,c2 = st.columns(2)
        dc = c1.selectbox("Date column", dates)
        vc = c2.selectbox("Value column", nums)
        period = st.selectbox("Time period", ["Monthly","Weekly","Daily","Yearly"], index=0)
        trend = time_pivot(cleaned, dc, vc, period)

        if trend.empty:
            st.warning("There is not enough valid date/value data.")
        else:
            st.dataframe(trend, use_container_width=True)
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(trend[dc], trend[vc], marker="o")
            ax.set_title(f"{vc} trend — {period}")
            ax.set_xlabel("Date")
            ax.set_ylabel(vc)
            ax.grid(alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            first = float(trend[vc].iloc[0])
            last = float(trend[vc].iloc[-1])
            change = last-first
            pct = change/first*100 if first != 0 else np.nan
            if change > 0:
                st.success(f"Simple explanation: the value increased from {first:,.2f} to {last:,.2f}.")
            elif change < 0:
                st.warning(f"Simple explanation: the value decreased from {first:,.2f} to {last:,.2f}.")
            else:
                st.info("Simple explanation: the first and last period have the same value.")
            if np.isfinite(pct):
                st.write(f"Overall change: **{pct:+.2f}%**")
            st.write(f"Highest period: **{trend.loc[trend[vc].idxmax(), dc]}**")
            st.write(f"Lowest period: **{trend.loc[trend[vc].idxmin(), dc]}**")

# -------------------- INSIGHTS --------------------
with tabs[4]:
    st.header("💡 Step 5 — Simple Insights")
    st.write("These are simple, data-based observations — no complicated statistics required.")

    nums = numeric_columns(cleaned)
    cats = categorical_columns(cleaned)

    st.subheader("Dataset overview")
    st.write(f"• **{len(cleaned):,}** rows and **{len(cleaned.columns):,}** columns")
    st.write(f"• **{len(nums)}** numeric columns and **{len(cats)}** category/text columns")
    st.write(f"• **{int(cleaned.isna().sum().sum()):,}** missing cells after cleaning")

    if cats and nums:
        cat = cats[0]
        val = nums[0]
        p = make_pivot(cleaned, cat, val, "Sum")
        if not p.empty:
            high = p.iloc[0]
            low = p.iloc[-1]
            st.subheader("Important findings")
            st.write(f"🔝 Highest **{val}** category: **{high[cat]}** ({high[val]:,.2f})")
            st.write(f"🔻 Lowest **{val}** category: **{low[cat]}** ({low[val]:,.2f})")
            st.write(f"🏆 Total **{val}: {p[val].sum():,.2f}**")

# -------------------- EXPORT --------------------
with tabs[5]:
    st.header("⬇️ Step 6 — Export")
    st.write("Download the cleaned data and analysis tables.")

    cleaned_buf = io.BytesIO()
    with pd.ExcelWriter(cleaned_buf, engine="openpyxl") as writer:
        cleaned.to_excel(writer, index=False, sheet_name="Cleaned Data")
        quality_table(cleaned).to_excel(writer, index=False, sheet_name="Data Quality")

        cats = categorical_columns(cleaned)
        nums = numeric_columns(cleaned)
        if cats and nums:
            p = make_pivot(cleaned, cats[0], nums[0], "Sum")
            p.to_excel(writer, index=False, sheet_name="Pivot Summary")

        dates = date_columns(cleaned)
        if dates and nums:
            t = time_pivot(cleaned, dates[0], nums[0], "Monthly")
            t.to_excel(writer, index=False, sheet_name="Monthly Trend")

    cleaned_buf.seek(0)
    st.download_button(
        "⬇️ Download Complete Analysis Excel",
        cleaned_buf,
        "Simple_AI_Data_Analyst_Analysis.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    csv = cleaned.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Cleaned CSV", csv, "Cleaned_Data.csv", "text/csv")

st.divider()
st.caption("Simple AI Data Analyst — designed for beginners: Clean → Pivot → Chart → Understand → Export")
