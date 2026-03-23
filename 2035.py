import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

try:
    import kaleido  # noqa: F401
    KALEIDO_AVAILABLE = True
except Exception:
    KALEIDO_AVAILABLE = False

APP_TITLE = "Penzija"
CONFIG_FILE = Path("saved_configs.json")

st.set_page_config(page_title=APP_TITLE, layout="wide")


def load_saved_configs() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}



def save_config(name: str, config: dict) -> None:
    configs = load_saved_configs()
    configs[name] = config
    CONFIG_FILE.write_text(json.dumps(configs, indent=2), encoding="utf-8")



def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()



def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))



def bollinger_bands(series: pd.Series, window: int = 20, std_dev: float = 2.0):
    mid = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return mid, upper, lower



def financial_ratios(info: dict) -> pd.DataFrame:
    ratio_map = {
        "Market Cap": info.get("marketCap"),
        "Trailing P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "Price to Book": info.get("priceToBook"),
        "Dividend Yield": info.get("dividendYield"),
        "Profit Margin": info.get("profitMargins"),
        "Operating Margin": info.get("operatingMargins"),
        "Return on Equity": info.get("returnOnEquity"),
        "Return on Assets": info.get("returnOnAssets"),
        "Current Ratio": info.get("currentRatio"),
        "Debt to Equity": info.get("debtToEquity"),
        "Beta": info.get("beta"),
    }
    return pd.DataFrame(
        {"Metric": list(ratio_map.keys()), "Value": list(ratio_map.values())}
    )



def normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Weight" not in df.columns:
        df["Weight"] = 1.0
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0)
    total = df["Weight"].sum()
    if total > 0:
        df["Normalized Weight"] = df["Weight"] / total
    else:
        df["Normalized Weight"] = 1 / max(len(df), 1)
    return df



def build_price_figure(
    data: pd.DataFrame,
    ticker: str,
    show_sma_20: bool,
    show_sma_50: bool,
    show_bb: bool,
    rsi_period: int,
    bb_window: int,
    bb_std: float,
    chart_height: int,
) -> go.Figure:
    data = data.copy()
    data["RSI"] = rsi(data["Close"], rsi_period)
    data["SMA20"] = sma(data["Close"], 20)
    data["SMA50"] = sma(data["Close"], 50)
    bb_mid, bb_upper, bb_lower = bollinger_bands(data["Close"], bb_window, bb_std)
    data["BB_Mid"] = bb_mid
    data["BB_Upper"] = bb_upper
    data["BB_Lower"] = bb_lower

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=(f"{ticker} Price", "RSI"),
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Candlestick",
        ),
        row=1,
        col=1,
    )

    if show_sma_20:
        fig.add_trace(
            go.Scatter(x=data.index, y=data["SMA20"], mode="lines", name="SMA 20"),
            row=1,
            col=1,
        )

    if show_sma_50:
        fig.add_trace(
            go.Scatter(x=data.index, y=data["SMA50"], mode="lines", name="SMA 50"),
            row=1,
            col=1,
        )

    if show_bb:
        fig.add_trace(
            go.Scatter(x=data.index, y=data["BB_Upper"], mode="lines", name="BB Upper"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data["BB_Mid"], mode="lines", name="BB Mid"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data["BB_Lower"], mode="lines", name="BB Lower"),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI"),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", row=2, col=1)

    fig.update_layout(
        height=chart_height,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig



def build_correlation_heatmap(adj_close: pd.DataFrame) -> go.Figure:
    corr = adj_close.corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(template="plotly_white", height=600, title="Stock Correlation Heatmap")
    return fig



def download_plot_html(fig: go.Figure, filename: str):
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    st.download_button(
        label="Download Chart as HTML",
        data=html,
        file_name=filename,
        mime="text/html",
    )



def download_plot_png(fig: go.Figure, filename: str):
    if KALEIDO_AVAILABLE:
        png_bytes = fig.to_image(format="png")
        st.download_button(
            label="Download Chart as PNG",
            data=png_bytes,
            file_name=filename,
            mime="image/png",
        )
    else:
        st.info("PNG export requires the 'kaleido' package. Install it with: pip install kaleido")



def read_portfolio_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


st.title(APP_TITLE)
st.caption("Analyze stocks, portfolios, technical indicators, and correlations with interactive charts.")

with st.sidebar:
    st.header("Configuration")
    ticker_input = st.text_input("Ticker(s)", value="AAPL,MSFT,GOOGL")
    ticker_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    period = st.selectbox(
        "Timeframe",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=3,
    )
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    st.subheader("Indicators")
    show_sma_20 = st.checkbox("Show SMA 20", value=True)
    show_sma_50 = st.checkbox("Show SMA 50", value=True)
    show_bb = st.checkbox("Show Bollinger Bands", value=True)
    rsi_period = st.slider("RSI Period", min_value=5, max_value=30, value=14)
    bb_window = st.slider("BB Window", min_value=10, max_value=50, value=20)
    bb_std = st.slider("BB Std Dev", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    chart_height = st.slider("Chart Height", min_value=500, max_value=1200, value=850, step=50)

    st.subheader("Saved Views")
    configs = load_saved_configs()
    selected_config = st.selectbox("Load saved configuration", ["None"] + list(configs.keys()))
    if selected_config != "None":
        loaded = configs[selected_config]
        st.write("Loaded config preview:")
        st.json(loaded)

    current_config = {
        "tickers": ticker_list,
        "period": period,
        "interval": interval,
        "show_sma_20": show_sma_20,
        "show_sma_50": show_sma_50,
        "show_bb": show_bb,
        "rsi_period": rsi_period,
        "bb_window": bb_window,
        "bb_std": bb_std,
        "chart_height": chart_height,
    }

    config_name = st.text_input("Save current config as", value="default_view")
    if st.button("Save configuration"):
        save_config(config_name, current_config)
        st.success(f"Configuration '{config_name}' saved.")

    st.download_button(
        label="Download current config JSON",
        data=json.dumps(current_config, indent=2),
        file_name="stock_visualizer_config.json",
        mime="application/json",
    )

    st.markdown("---")
    st.write("Shareable query string")
    st.code("?tickers=" + ",".join(ticker_list) + f"&period={period}&interval={interval}")

if not ticker_list:
    st.warning("Please enter at least one ticker.")
    st.stop()

with st.spinner("Fetching market data..."):
    raw = yf.download(
        tickers=ticker_list,
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

if raw.empty:
    st.error("No data returned. Check the ticker symbol(s) and try again.")
    st.stop()

if len(ticker_list) == 1:
    ticker = ticker_list[0]
    stock_df = raw.copy()
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(-1)

    fig = build_price_figure(
        stock_df,
        ticker,
        show_sma_20,
        show_sma_50,
        show_bb,
        rsi_period,
        bb_window,
        bb_std,
        chart_height,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    close = stock_df["Close"].dropna()
    latest = float(close.iloc[-1])
    start = float(close.iloc[0])
    pct_change = ((latest - start) / start) * 100 if start else 0
    daily_change = float(stock_df["Close"].iloc[-1] - stock_df["Close"].iloc[-2]) if len(stock_df) > 1 else 0
    volume = int(stock_df["Volume"].dropna().iloc[-1]) if "Volume" in stock_df.columns else 0

    c1.metric("Latest Close", f"{latest:,.2f}")
    c2.metric("Period Return", f"{pct_change:,.2f}%")
    c3.metric("Daily Change", f"{daily_change:,.2f}")
    c4.metric("Volume", f"{volume:,}")

    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info if ticker_obj else {}
    ratios_df = financial_ratios(info)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Financial Ratios")
        st.dataframe(ratios_df, use_container_width=True)
    with right:
        st.subheader("Company Snapshot")
        st.write(f"**Name:** {info.get('longName', ticker)}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Website:** {info.get('website', 'N/A')}")
        st.write(info.get("longBusinessSummary", "No summary available."))

    st.subheader("Export Chart")
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        download_plot_html(fig, f"{ticker.lower()}_chart.html")
    with export_col2:
        download_plot_png(fig, f"{ticker.lower()}_chart.png")

else:
    st.subheader("Multi-Stock Comparison")

    adj_close = pd.DataFrame()
    for ticker in ticker_list:
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                adj_close[ticker] = raw[ticker]["Close"]
            except Exception:
                continue

    adj_close = adj_close.dropna(how="all")

    if adj_close.empty:
        st.error("Could not build comparison data for the selected tickers.")
        st.stop()

    normalized = adj_close / adj_close.iloc[0] * 100
    comparison_fig = go.Figure()
    for col in normalized.columns:
        comparison_fig.add_trace(
            go.Scatter(x=normalized.index, y=normalized[col], mode="lines", name=col)
        )
    comparison_fig.update_layout(
        title="Normalized Performance Comparison",
        template="plotly_white",
        height=550,
        xaxis_title="Date",
        yaxis_title="Indexed Price (Base = 100)",
    )
    st.plotly_chart(comparison_fig, use_container_width=True)

    corr_fig = build_correlation_heatmap(adj_close)
    st.plotly_chart(corr_fig, use_container_width=True)

    returns = adj_close.pct_change().dropna()
    annualized_return = returns.mean() * 252
    annualized_vol = returns.std() * np.sqrt(252)

    summary_df = pd.DataFrame(
        {
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_vol,
        }
    ).reset_index().rename(columns={"index": "Ticker"})

    st.subheader("Return and Risk Summary")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Export Charts")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        download_plot_html(comparison_fig, "comparison_chart.html")
    with col_b:
        download_plot_png(comparison_fig, "comparison_chart.png")
    with col_c:
        download_plot_html(corr_fig, "correlation_heatmap.html")
    with col_d:
        download_plot_png(corr_fig, "correlation_heatmap.png")

st.markdown("---")
st.header("Portfolio Tracker")
st.write("Upload a CSV or Excel file with at least a 'Ticker' column. Optional columns: 'Shares', 'CostBasis', 'Weight'.")

portfolio_file = st.file_uploader("Upload portfolio file", type=["csv", "xlsx", "xls"])
if portfolio_file is not None:
    try:
        portfolio_df = read_portfolio_file(portfolio_file)
        portfolio_df.columns = [str(c).strip() for c in portfolio_df.columns]
        if "Ticker" not in portfolio_df.columns:
            st.error("Portfolio file must include a 'Ticker' column.")
        else:
            portfolio_df["Ticker"] = portfolio_df["Ticker"].astype(str).str.upper().str.strip()
            portfolio_df = normalize_weights(portfolio_df)
            st.dataframe(portfolio_df, use_container_width=True)

            portfolio_tickers = portfolio_df["Ticker"].dropna().unique().tolist()
            p_raw = yf.download(
                tickers=portfolio_tickers,
                period=period,
                interval=interval,
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=True,
            )

            latest_prices = {}
            for t in portfolio_tickers:
                try:
                    latest_prices[t] = float(p_raw[t]["Close"].dropna().iloc[-1])
                except Exception:
                    latest_prices[t] = np.nan

            portfolio_df["Latest Price"] = portfolio_df["Ticker"].map(latest_prices)
            portfolio_df["Shares"] = pd.to_numeric(portfolio_df.get("Shares", 0), errors="coerce").fillna(0)
            portfolio_df["CostBasis"] = pd.to_numeric(portfolio_df.get("CostBasis", 0), errors="coerce").fillna(0)
            portfolio_df["Market Value"] = portfolio_df["Shares"] * portfolio_df["Latest Price"]
            portfolio_df["Cost Value"] = portfolio_df["Shares"] * portfolio_df["CostBasis"]
            portfolio_df["PnL"] = portfolio_df["Market Value"] - portfolio_df["Cost Value"]

            total_value = portfolio_df["Market Value"].sum()
            total_cost = portfolio_df["Cost Value"].sum()
            total_pnl = portfolio_df["PnL"].sum()

            p1, p2, p3 = st.columns(3)
            p1.metric("Portfolio Value", f"{total_value:,.2f}")
            p2.metric("Portfolio Cost", f"{total_cost:,.2f}")
            p3.metric("Portfolio P&L", f"{total_pnl:,.2f}")

            st.subheader("Portfolio Holdings")
            st.dataframe(portfolio_df, use_container_width=True)

            holdings_fig = go.Figure(
                data=[
                    go.Pie(
                        labels=portfolio_df["Ticker"],
                        values=portfolio_df["Market Value"].replace(0, np.nan).fillna(0.0001),
                        hole=0.4,
                    )
                ]
            )
            holdings_fig.update_layout(template="plotly_white", title="Portfolio Allocation by Market Value")
            st.plotly_chart(holdings_fig, use_container_width=True)

            download_plot_html(holdings_fig, "portfolio_allocation.html")
            download_plot_png(holdings_fig, "portfolio_allocation.png")
    except Exception as exc:
        st.error(f"Could not process portfolio file: {exc}")

st.markdown("---")
st.subheader("Notes")
st.markdown(
    """
- For PNG export, install `kaleido`.
- Real-time intraday quality depends on Yahoo Finance availability.
- To deploy, use Streamlit Community Cloud, Docker, or any VM with Python.
- GitHub push must be done from your machine or IDE after creating a repository.
"""
)
