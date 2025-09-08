import datetime as dt
from io import BytesIO
from typing import List, Optional

import pandas as pd
import streamlit as st
import plotly.express as px

# Optional import: we degrade gracefully if yfinance is missing at runtime.
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

st.set_page_config(page_title="Data Viz Dashboard", page_icon="📈", layout="wide")

@st.cache_data(show_spinner=False)
def fetch_prices(tickers: List[str], start: dt.date, end: dt.date, interval: str) -> pd.DataFrame:
    """
    Fetch OHLCV data with yfinance (multi-index columns), return a tidy DataFrame:
    index: Datetime, columns: ['ticker', 'Open','High','Low','Close','Adj Close','Volume'].
    """
    if yf is None:
        raise RuntimeError("yfinance not installed on server. Use CSV upload or add yfinance.")

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end + dt.timedelta(days=1),  # include end date
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    frames = []
    for t in tickers:
        df = data[t].copy() if len(tickers) > 1 else data.copy()
        df["ticker"] = t
        frames.append(df.reset_index().rename(columns={"index": "Date"}))
    out = pd.concat(frames, ignore_index=True)
    out.rename(columns={"Date": "date", "Adj Close": "AdjClose", "Close": "Close"}, inplace=True)
    out = out.sort_values(["ticker", "date"]).dropna(subset=["Close"])
    return out

def read_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Try to standardize likely column names
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = None
    for k in ["date", "timestamp", "time"]:
        if k in cols:
            date_col = cols[k]
            break
    if date_col is None:
        raise ValueError("CSV must include a 'date' (or timestamp/time) column.")
    df.rename(columns={date_col: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    if "ticker" not in df.columns:
        df["ticker"] = "DATA"
    # Normalize typical price col names
    for candidate in [("adj close", "AdjClose"), ("adjclose", "AdjClose"), ("close", "Close")]:
        lower = candidate[0]
        if lower in cols:
            df.rename(columns={cols[lower]: candidate[1]}, inplace=True)
    df = df.sort_values(["ticker", "date"])
    return df

def to_csv_download(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()

st.sidebar.title("📊 Controls")

# Data source
source = st.sidebar.selectbox("Data source", ["Fetch with yfinance", "Upload CSV"])

# Date range
today = dt.date.today()
default_start = today - dt.timedelta(days=365)
start_date, end_date = st.sidebar.date_input(
    "Date range", value=(default_start, today), min_value=today - dt.timedelta(days=3650), max_value=today
)

# Interval / resampling
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"])
resample_rule = st.sidebar.selectbox("Resample", ["None", "W", "M", "Q"])

# Columns & options
price_col = st.sidebar.selectbox("Price column", ["AdjClose", "Close"])
norm = st.sidebar.checkbox("Normalize to 100 at start", value=True)
ma = st.sidebar.multiselect("Moving averages (days)", [7, 20, 50, 100, 200], default=[20, 50])

# Data load
df: Optional[pd.DataFrame] = None
error_placeholder = st.empty()

if source == "Fetch with yfinance":
    tickers_text = st.sidebar.text_input("Tickers (comma-separated)", value="AAPL,MSFT,GOOGL")
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    fetch_btn = st.sidebar.button("Fetch data")
    if fetch_btn:
        try:
            with st.spinner("Fetching prices..."):
                df = fetch_prices(tickers, start_date, end_date, interval)
        except Exception as e:
            error_placeholder.error(f"Could not fetch data: {e}")
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = read_csv(uploaded)
            # Filter to chosen range
            df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]
        except Exception as e:
            error_placeholder.error(f"Invalid CSV: {e}")

st.title("Data Visualization Dashboard")
st.caption("Interactive exploration with Streamlit + Plotly. Use the controls in the sidebar.")

if df is None:
    st.info("Choose a data source on the left (fetch tickers with yfinance or upload your own CSV).")
    st.stop()

# Resample per ticker if requested
if resample_rule != "None":
    df = (df.set_index("date")
            .groupby("ticker")
            .resample(resample_rule)[price_col]
            .last()
            .reset_index())

# Compute normalization & moving averages per ticker
df_plot = df.copy()
df_plot = df_plot.dropna(subset=[price_col])
df_plot["value"] = df_plot[price_col].astype(float)

if norm:
    df_plot["norm_base"] = df_plot.groupby("ticker")["value"].transform("first")
    df_plot["value"] = 100.0 * df_plot["value"] / df_plot["norm_base"]
    y_label = f"Normalized {price_col} (start=100)"
else:
    y_label = price_col

for window in ma:
    df_plot[f"MA{window}"] = (
        df_plot.groupby("ticker")["value"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    )

# KPI row
latest = (df_plot.sort_values("date").groupby("ticker").tail(1))[["ticker", "date", "value"]]
first = (df_plot.sort_values("date").groupby("ticker").head(1))[["ticker", "value"]].rename(columns={"value": "first_val"})
kpi = latest.merge(first, on="ticker")
kpi["return_%"] = 100.0 * (kpi["value"] / kpi["first_val"] - 1.0)

c1, c2, c3 = st.columns(3)
c1.metric("Tickers", ", ".join(kpi["ticker"].tolist()))
c2.metric("Range", f"{start_date} → {end_date}")
c3.metric("Median Return", f"{kpi['return_%'].median():.2f}%")

# Line chart
fig = px.line(
    df_plot,
    x="date",
    y="value",
    color="ticker",
    title="Price over time",
    labels={"value": y_label, "date": "Date", "ticker": "Ticker"},
)
st.plotly_chart(fig, use_container_width=True)

# Optional: overlay MAs (per ticker as separate traces)
for window in ma:
    ma_df = df_plot.dropna(subset=[f"MA{window}"])
    if not ma_df.empty:
        fig_ma = px.line(
            ma_df, x="date", y=f"MA{window}", color="ticker",
            title=f"Moving Average (window={window})", labels={"date": "Date", f"MA{window}": f"MA{window}"}
        )
        st.plotly_chart(fig_ma, use_container_width=True)

# Bar: returns by ticker
ret = kpi.sort_values("return_%", ascending=False)
bar = px.bar(ret, x="ticker", y="return_%", title="Return by ticker (%)", labels={"return_%": "Return (%)"})
st.plotly_chart(bar, use_container_width=True)

# Data preview + download
with st.expander("Preview data"):
    st.dataframe(df.head(50), use_container_width=True)
st.download_button(
    "Download filtered data as CSV",
    data=to_csv_download(df),
    file_name="filtered_data.csv",
    mime="text/csv",
)
st.caption("Tip: Upload your own CSV with columns like date, ticker, Close/AdjClose.")
