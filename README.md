# Streamlit Data Viz Dashboard 📈
[![Dashboard Smoke](https://github.com/GerardoMacedo/viz-dashboard/actions/workflows/smoke.yml/badge.svg)](https://github.com/GerardoMacedo/viz-dashboard/actions/workflows/smoke.yml)


Interactive dashboard for exploring time-series data. Fetch stock prices via `yfinance` **or** upload your own CSV, then filter, resample, plot moving averages, and download the filtered data.

## Features
- Sidebar controls: date range, interval, resample (W/M/Q), normalize, moving averages
- Line & bar charts with Plotly
- CSV upload support + CSV download of filtered data
- Caching for fast reloads

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
