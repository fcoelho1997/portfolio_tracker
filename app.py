import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Login")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if password == "2972":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()

check_password()


PORTFOLIO_FILE = "portfolio.csv"
RF_ANNUAL = 0.045
ALPHA_SINCE_DATE = datetime(2026, 2, 1).date()

# Custom display order for tables and charts
TICKER_ORDER = ["VOO", "DIA", "VINP", "NU", "BRK-B", "MSA", "DUK"]

# Sector classification
SECTOR_MAP = {
    "VOO":   "Broad US Equity (ETF)",
    "DIA":   "Broad US Equity (ETF)",
    "SPY":   "Broad US Equity (ETF)",
    "VINP":  "Financial Services",
    "NU":    "Financial Services",
    "BRK-B": "Diversified Equity (US)",
    "MSA":   "Industrials",
    "DUK":   "Utilities",
}

# Geography classification
GEOGRAPHY_MAP = {
    "VOO":   "United States",
    "DIA":   "United States",
    "SPY":   "United States",
    "VINP":  "Brazil",
    "NU":    "Brazil",
    "BRK-B": "United States",
    "MSA":   "United States",
    "DUK":   "United States",
}

# Manually sourced 5Y betas (yfinance returns None for several of these holdings)
# Sources: stockanalysis.com, macroaxis.com (May 2026)
MANUAL_BETAS = {
    "VOO":   1.00,
    "DIA":   0.95,
    "SPY":   1.00,
    "VINP":  0.09,
    "NU":    1.01,
    "BRK-B": 0.62,
    "MSA":   1.01,
    "DUK":   0.40,
}


def sort_tickers(tickers):
    """Sort tickers per TICKER_ORDER, unknown tickers alphabetically at the end."""
    in_order = [t for t in TICKER_ORDER if t in tickers]
    others   = sorted([t for t in tickers if t not in TICKER_ORDER])
    return in_order + others


def classify_sector(ticker, info=None):
    if ticker in SECTOR_MAP:
        return SECTOR_MAP[ticker]
    if info:
        return info.get("sector") or "Unknown"
    return "Unknown"


def classify_geography(ticker, info=None):
    if ticker in GEOGRAPHY_MAP:
        return GEOGRAPHY_MAP[ticker]
    if info:
        return info.get("country") or "Unknown"
    return "Unknown"


def get_manual_beta(ticker):
    return MANUAL_BETAS.get(ticker)


# ── Formatters ────────────────────────────────────────────────────────────────

def fmt_usd(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"${x:,.2f}"

def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.2f}%"

def fmt_pct_sign(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.2f}%"

def fmt_mcap(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if x >= 1e12:
        return f"${x/1e12:.2f}T"
    if x >= 1e9:
        return f"${x/1e9:.2f}B"
    if x >= 1e6:
        return f"${x/1e6:.2f}M"
    return f"${x:,.0f}"

def fmt_num(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{d}f}"

def company_label(ticker, info):
    name = info.get("longName") or info.get("shortName") or ticker
    return f"{name} ({ticker})"


# ── Persistence ───────────────────────────────────────────────────────────────

def _sb():
    try:
        url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
        if url and key:
            from supabase import create_client
            return create_client(url, key)
    except Exception:
        pass
    return None


def load_portfolio():
    client = _sb()
    if client:
        result = client.table("trades").select("*").order("date").execute()
        if result.data:
            df = pd.DataFrame(result.data)
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            return df[["id", "ticker", "date", "quantity", "price_paid"]]
        return pd.DataFrame(columns=["id", "ticker", "date", "quantity", "price_paid"])
    if os.path.exists(PORTFOLIO_FILE):
        df = pd.read_csv(PORTFOLIO_FILE, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if "id" not in df.columns:
            df.insert(0, "id", range(len(df)))
        return df
    return pd.DataFrame(columns=["id", "ticker", "date", "quantity", "price_paid"])


def add_trade(ticker, trade_date, quantity, price_paid):
    client = _sb()
    if client:
        client.table("trades").insert({
            "ticker": ticker, "date": str(trade_date),
            "quantity": float(quantity), "price_paid": float(price_paid),
        }).execute()
    else:
        df = load_portfolio()
        new_id = int(df["id"].max() + 1) if not df.empty else 0
        df = pd.concat([df, pd.DataFrame({
            "id": [new_id], "ticker": [ticker],
            "date": [pd.Timestamp(trade_date)],
            "quantity": [float(quantity)], "price_paid": [float(price_paid)],
        })], ignore_index=True)
        df.to_csv(PORTFOLIO_FILE, index=False)


def delete_trade(row_id):
    client = _sb()
    if client:
        client.table("trades").delete().eq("id", int(row_id)).execute()
    else:
        df = load_portfolio()
        df = df[df["id"] != row_id].reset_index(drop=True)
        df.to_csv(PORTFOLIO_FILE, index=False)


def compute_net_positions(portfolio):
    result = {}
    for ticker, g in portfolio.groupby("ticker"):
        buys  = g[g["quantity"] > 0]
        sells = g[g["quantity"] < 0]

        net_qty = float(g["quantity"].sum())

        total_buy_qty  = float(buys["quantity"].sum()) if not buys.empty else 0.0
        total_buy_cost = float((buys["quantity"] * buys["price_paid"]).sum()) if not buys.empty else 0.0
        avg_cost = total_buy_cost / total_buy_qty if total_buy_qty > 0 else 0.0

        realized_pnl = 0.0
        if not sells.empty:
            for _, sell_row in sells.iterrows():
                shares_sold = abs(float(sell_row["quantity"]))
                sale_price  = float(sell_row["price_paid"])
                realized_pnl += (sale_price - avg_cost) * shares_sold

        result[ticker] = {
            "net_qty":      net_qty,
            "avg_cost":     avg_cost,
            "realized_pnl": realized_pnl,
        }
    return result


# ── Fetching ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=0)
def fetch_prices(ticker, start, end):
    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw.empty:
            return pd.Series(dtype=float, name=ticker)
        close = raw["Close"][ticker] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if getattr(close.index, "tz", None) is not None:
            close.index = close.index.tz_localize(None)
        return close.rename(ticker)
    except Exception:
        return pd.Series(dtype=float, name=ticker)


@st.cache_data(ttl=0)
def get_price_history(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    try:
        data = yf.download(list(tickers), start=start, end=end, auto_adjust=True,
                           progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0])
        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_localize(None)
        return data
    except Exception:
        return pd.DataFrame()


def current_price(ticker):
    s = fetch_prices(ticker, str(date.today() - timedelta(days=7)), str(date.today() + timedelta(days=1)))
    return float(s.iloc[-1]) if not s.empty else None


@st.cache_data(ttl=3600)
def get_ticker_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


@st.cache_data(ttl=300)
def get_dividends(ticker):
    try:
        divs = yf.Ticker(ticker).dividends
        if not divs.empty and divs.index.tz is not None:
            divs.index = divs.index.tz_localize(None)
        return divs
    except Exception:
        return pd.Series(dtype=float)


def period_return(history_series, period_days):
    """Total price return over the last `period_days`. Returns % (e.g. 12.5 = +12.5%)."""
    if history_series is None or history_series.empty:
        return None
    start_ts = pd.Timestamp(date.today() - timedelta(days=period_days))
    slice_ = history_series[history_series.index >= start_ts]
    if slice_.empty or len(slice_) < 2:
        return None
    try:
        return (float(slice_.iloc[-1]) / float(slice_.iloc[0]) - 1) * 100
    except Exception:
        return None


# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Portfolio Tracker", page_icon="📈", layout="wide")
st.title("Stock Portfolio Tracker")

with st.sidebar:
    st.header("Record a Trade")
    with st.form("add_trade"):
        ticker_input  = st.text_input("Ticker Symbol", placeholder="AAPL")
        trade_type    = st.radio("Trade Type", ["Buy", "Sell"], horizontal=True)
        trade_date    = st.date_input("Trade Date", max_value=date.today())
        quantity_input = st.number_input("Shares", min_value=0.001, step=1.0, format="%.3f")
        price_input   = st.number_input(
            "Price per Share ($)",
            min_value=0.01, step=0.01, format="%.2f",
            help="Enter the price paid (buy) or price received (sell)."
        )
        submitted = st.form_submit_button("Submit Trade", use_container_width=True)

    if submitted:
        sym = ticker_input.strip().upper()
        if not sym:
            st.error("Ticker symbol is required.")
        elif current_price(sym) is None:
            st.error(f"Could not find '{sym}'.")
        else:
            _current_portfolio = load_portfolio()
            _net_pos = compute_net_positions(_current_portfolio)
            _net_qty_ticker = _net_pos.get(sym, {}).get("net_qty", 0.0)

            if trade_type == "Sell":
                if quantity_input > _net_qty_ticker:
                    st.warning(
                        f"You are trying to sell {quantity_input:.3f} shares of {sym}, "
                        f"but your net position is only {_net_qty_ticker:.3f} shares."
                    )
                else:
                    add_trade(sym, trade_date, -float(quantity_input), float(price_input))
                    st.success(f"Sold {quantity_input:.3f} x {sym} @ ${price_input:.2f}")
                    st.rerun()
            else:
                add_trade(sym, trade_date, float(quantity_input), float(price_input))
                st.success(f"Bought {quantity_input:.3f} x {sym} @ ${price_input:.2f}")
                st.rerun()

# ── Load ALL data upfront ─────────────────────────────────────────────────────

portfolio = load_portfolio()

if portfolio.empty:
    st.info("No trades yet. Use the sidebar to add your first trade.")
    st.stop()

net_positions = compute_net_positions(portfolio)
active_tickers = sort_tickers([t for t, v in net_positions.items() if v["net_qty"] > 0])
all_tickers    = portfolio["ticker"].unique().tolist()

min_date = portfolio["date"].min().date()
today    = date.today()
end_str  = str(today + timedelta(days=1))
five_yr_start = today - timedelta(days=5 * 365)

with st.spinner("Loading market data, please wait..."):
    prices   = {t: current_price(t)  for t in active_tickers}
    infos    = {t: get_ticker_info(t) for t in active_tickers}
    history  = {t: fetch_prices(t, str(min_date), end_str) for t in all_tickers + ["SPY"]}
    div_data = {t: get_dividends(t)   for t in active_tickers}

    all_needed_tickers = tuple(sorted(set(active_tickers + ["SPY"])))
    bulk_history = get_price_history(all_needed_tickers, str(min_date), end_str)

    # 5Y price history for each holding (used for 1Y and 5Y return calculations)
    hist_5y  = {}
    for t in active_tickers:
        hist_5y[t]  = fetch_prices(t, str(five_yr_start), end_str)

# ── Dividends & yield-on-cost helpers ─────────────────────────────────────────

def ttm_dividends_per_share(ticker):
    divs = div_data.get(ticker, pd.Series(dtype=float))
    if divs.empty:
        return 0.0
    cutoff = pd.Timestamp(today - timedelta(days=365))
    return float(divs[divs.index >= cutoff].sum())


# ── Enrich rows (active holdings only) ───────────────────────────────────────

div_by_ticker = {t: 0.0 for t in active_tickers}
for t in active_tickers:
    divs = div_data.get(t, pd.Series(dtype=float))
    if divs.empty:
        continue
    for div_date, dps in divs.items():
        shares_at_date = float(portfolio[
            (portfolio["ticker"] == t) &
            (pd.to_datetime(portfolio["date"]).dt.tz_localize(None) <= div_date)
        ]["quantity"].sum())
        if shares_at_date > 0:
            div_by_ticker[t] += dps * shares_at_date

holding_rows = []
for ticker in active_tickers:
    pos     = net_positions[ticker]
    net_qty = pos["net_qty"]
    avg_cost = pos["avg_cost"]
    cur_price = prices.get(ticker)
    cur_val   = net_qty * cur_price if cur_price is not None else None
    cost_bas  = net_qty * avg_cost
    gain_loss = (cur_val - cost_bas) if cur_val is not None else None
    ret_pct   = (gain_loss / cost_bas * 100) if (gain_loss is not None and cost_bas > 0) else None

    divs_received = div_by_ticker.get(ticker, 0.0)
    ttm_dps       = ttm_dividends_per_share(ticker)
    yoc_pct       = (ttm_dps / avg_cost * 100) if avg_cost > 0 else None
    total_pnl     = (gain_loss + divs_received) if gain_loss is not None else None
    total_ret_pct = (total_pnl / cost_bas * 100) if (total_pnl is not None and cost_bas > 0) else None

    # Per-ticker 1Y and 5Y returns from yfinance price history
    h5 = hist_5y.get(ticker, pd.Series(dtype=float))
    ret_1y = period_return(h5, 365)
    ret_5y = None
    if not h5.empty and len(h5) > 10:
        try:
            ret_5y = (float(h5.iloc[-1]) / float(h5.iloc[0]) - 1) * 100
        except Exception:
            ret_5y = None

    beta = get_manual_beta(ticker)

    ticker_buys = portfolio[(portfolio["ticker"] == ticker) & (portfolio["quantity"] > 0)]
    first_buy_date = ticker_buys["date"].min() if not ticker_buys.empty else portfolio[portfolio["ticker"] == ticker]["date"].min()
    holding_days = (pd.Timestamp(today) - pd.Timestamp(first_buy_date)).days

    holding_rows.append({
        "ticker":          ticker,
        "company":         company_label(ticker, infos.get(ticker, {})),
        "net_qty":         net_qty,
        "avg_cost":        avg_cost,
        "current_price":   cur_price,
        "cost_basis":      cost_bas,
        "current_value":   cur_val,
        "gain_loss":       gain_loss,
        "return_pct":      ret_pct,
        "dividends":       divs_received,
        "total_pnl":       total_pnl,
        "total_return_pct": total_ret_pct,
        "yield_on_cost":   yoc_pct,
        "ttm_dps":         ttm_dps,
        "holding_days":    holding_days,
        "realized_pnl":    pos["realized_pnl"],
        "sector":          classify_sector(ticker, infos.get(ticker, {})),
        "geography":       classify_geography(ticker, infos.get(ticker, {})),
        "beta":            beta,
        "ret_1y":          ret_1y,
        "ret_5y":          ret_5y,
    })

holdings_df = pd.DataFrame(holding_rows) if holding_rows else pd.DataFrame()

# Realized P&L totals
total_realized_pnl = sum(net_positions[t]["realized_pnl"] for t in all_tickers)

# ── Performance Series ────────────────────────────────────────────────────────

date_range = pd.date_range(start=min_date, end=today, freq="B")
port_vals, port_costs = [], []

for d in date_range:
    pv = pc = 0.0
    active_at_d = portfolio[pd.to_datetime(portfolio["date"]).dt.tz_localize(None) <= d]
    net_at_d = {}
    for t, g in active_at_d.groupby("ticker"):
        buys_d  = g[g["quantity"] > 0]
        net_qty_d = float(g["quantity"].sum())
        if net_qty_d <= 0:
            continue
        total_buy_qty_d  = float(buys_d["quantity"].sum()) if not buys_d.empty else 0.0
        total_buy_cost_d = float((buys_d["quantity"] * buys_d["price_paid"]).sum()) if not buys_d.empty else 0.0
        avg_cost_d = total_buy_cost_d / total_buy_qty_d if total_buy_qty_d > 0 else 0.0
        net_at_d[t] = {"net_qty": net_qty_d, "avg_cost": avg_cost_d}

    for t, pos_d in net_at_d.items():
        if t not in history or history[t].empty:
            continue
        avail = history[t][history[t].index <= d]
        if avail.empty:
            continue
        pv += pos_d["net_qty"] * float(avail.iloc[-1])
        pc += pos_d["net_qty"] * pos_d["avg_cost"]
    port_vals.append(pv)
    port_costs.append(pc)

perf = pd.DataFrame({"date": date_range, "port_val": port_vals, "port_cost": port_costs})
perf = perf[perf["port_cost"] > 0].copy().reset_index(drop=True)
perf["port_ret"] = (perf["port_val"] - perf["port_cost"]) / perf["port_cost"] * 100

spy_hist = history.get("SPY", pd.Series(dtype=float))
if not spy_hist.empty:
    spy_from = spy_hist[spy_hist.index >= pd.Timestamp(min_date)]
    spy_base = float(spy_from.iloc[0]) if not spy_from.empty else float(spy_hist.iloc[0])
    perf["spy_ret"] = perf["date"].apply(
        lambda d: (float(spy_hist[spy_hist.index <= d].iloc[-1]) / spy_base - 1) * 100
        if len(spy_hist[spy_hist.index <= d]) > 0 else 0.0
    )
else:
    perf["spy_ret"] = 0.0

if not perf.empty:
    perf["port_ret"] -= perf["port_ret"].iloc[0]

perf["alpha"] = perf["port_ret"] - perf["spy_ret"]

# Drawdown series (used in Performance tab)
dd_series = pd.Series(dtype=float)
if len(perf) >= 5:
    port_daily = perf.set_index("date")["port_val"].pct_change().dropna()
    cum      = (1 + port_daily.fillna(0)).cumprod()
    roll_max = cum.expanding().max()
    dd_series = (cum - roll_max) / roll_max * 100

# ── Portfolio-level weighted metrics for Holdings tab ─────────────────────────

total_port_val = holdings_df["current_value"].sum() if holdings_df["current_value"].notna().any() else 0.0

# Weighted beta (only over holdings with a beta value, renormalized)
beta_num = 0.0
beta_den = 0.0
# Weighted 1Y return
ret1y_num = 0.0
ret1y_den = 0.0
# Weighted 5Y return
ret5y_num = 0.0
ret5y_den = 0.0

for _, hr in holdings_df.iterrows():
    if total_port_val <= 0 or pd.isna(hr["current_value"]):
        continue
    weight = hr["current_value"] / total_port_val
    if hr["beta"] is not None:
        beta_num += hr["beta"] * weight
        beta_den += weight
    if hr["ret_1y"] is not None:
        ret1y_num += hr["ret_1y"] * weight
        ret1y_den += weight
    if hr["ret_5y"] is not None:
        ret5y_num += hr["ret_5y"] * weight
        ret5y_den += weight

portfolio_beta  = (beta_num  / beta_den)  if beta_den  > 0 else None
portfolio_1y    = (ret1y_num / ret1y_den) if ret1y_den > 0 else None
portfolio_5y    = (ret5y_num / ret5y_den) if ret5y_den > 0 else None

# ── Alpha Calculations ────────────────────────────────────────────────────────

def compute_alpha_for_period(start_date, net_pos, history_dict, spy_series):
    start_ts = pd.Timestamp(start_date)

    port_val_start = 0.0
    port_val_end   = 0.0
    for t, pos in net_pos.items():
        if pos["net_qty"] <= 0:
            continue
        h = history_dict.get(t)
        if h is None or h.empty:
            continue
        avail_start = h[h.index <= start_ts]
        avail_end   = h[h.index <= pd.Timestamp(today)]
        if avail_start.empty or avail_end.empty:
            continue
        port_val_start += pos["net_qty"] * float(avail_start.iloc[-1])
        port_val_end   += pos["net_qty"] * float(avail_end.iloc[-1])

    if port_val_start <= 0:
        return None, None, None

    port_ret = (port_val_end / port_val_start - 1) * 100

    if spy_series.empty:
        return port_ret, None, None
    spy_start_slice = spy_series[spy_series.index <= start_ts]
    spy_end_slice   = spy_series[spy_series.index <= pd.Timestamp(today)]
    if spy_start_slice.empty or spy_end_slice.empty:
        return port_ret, None, None

    spy_start_price = float(spy_start_slice.iloc[-1])
    spy_end_price   = float(spy_end_slice.iloc[-1])
    spy_ret = (spy_end_price / spy_start_price - 1) * 100
    alpha   = port_ret - spy_ret
    return port_ret, spy_ret, alpha


alpha_inception_port, alpha_inception_spy, alpha_inception = compute_alpha_for_period(
    min_date, net_positions, history, spy_hist
)
alpha_feb_port, alpha_feb_spy, alpha_feb = compute_alpha_for_period(
    ALPHA_SINCE_DATE, net_positions, history, spy_hist
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_h, tab_p, tab_a, tab_pnl, tab_nw, tab_alpha = st.tabs([
    "Holdings", "Performance", "Allocation",
    "P&L & Dividends", "Net Worth", "Alpha",
])

# ─── Holdings ────────────────────────────────────────────────────────────────

with tab_h:
    if holdings_df.empty:
        st.info("No active holdings.")
    else:
        grouped_rows = []
        for _, hr in holdings_df.iterrows():
            grouped_rows.append({
                "Company":         hr["company"],
                "Shares":          fmt_num(hr["net_qty"]),
                "Avg Cost":        fmt_usd(hr["avg_cost"]),
                "Current Price":   fmt_usd(hr["current_price"]),
                "Cost Basis":      fmt_usd(hr["cost_basis"]),
                "Current Value":   fmt_usd(hr["current_value"]),
                "Unrealized P&L":  fmt_usd(hr["gain_loss"]),
                "Dividends":       fmt_usd(hr["dividends"]),
                "Total P&L":       fmt_usd(hr["total_pnl"]),
                "Return (w/ div)": fmt_pct_sign(hr["total_return_pct"]),
                "Yield on Cost":   fmt_pct(hr["yield_on_cost"]) if hr["yield_on_cost"] is not None else "—",
                "Avg Hold (days)": int(hr["holding_days"]),
            })
        st.dataframe(pd.DataFrame(grouped_rows), use_container_width=True, hide_index=True)

        tc      = holdings_df["cost_basis"].sum()
        tv      = holdings_df["current_value"].sum() if holdings_df["current_value"].notna().any() else 0.0
        tg      = tv - tc
        tdivs   = holdings_df["dividends"].sum()
        ttm_div_dollar = float((holdings_df["ttm_dps"] * holdings_df["net_qty"]).sum())
        port_yoc = (ttm_div_dollar / tc * 100) if tc > 0 else None

        # Row 1: core stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Invested",   fmt_usd(tc))
        c2.metric("Portfolio Value",  fmt_usd(tv))
        c3.metric("Unrealized P&L",   fmt_usd(tg))
        c4.metric("Overall Return",   fmt_pct(tg / tc * 100) if tc > 0 else "—",
                  delta=fmt_pct(tg / tc * 100) if tc > 0 else None)

        # Row 2: dividend & total P&L stats
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Dividends Received", fmt_usd(tdivs))
        d2.metric("P&L (incl. Dividends)",    fmt_usd(tg + tdivs))
        d3.metric("Total Return (w/ div)",
                  fmt_pct_sign((tg + tdivs) / tc * 100) if tc > 0 else "—")
        d4.metric("Portfolio Yield on Cost",
                  fmt_pct(port_yoc) if port_yoc is not None else "—",
                  help="Trailing 12-month dividends ÷ original cost basis.")

        if total_realized_pnl != 0.0:
            st.metric("Realized P&L (all sells)", fmt_usd(total_realized_pnl))

        # ─── Performance & Risk by holding ────────────────────────────────────
        st.divider()
        st.subheader("Performance & Risk by Holding")

        risk_rows = []
        for _, hr in holdings_df.iterrows():
            risk_rows.append({
                "Company":     hr["company"],
                "Weight":      fmt_pct((hr["current_value"] / total_port_val * 100)
                                       if total_port_val > 0 and pd.notna(hr["current_value"])
                                       else None),
                "Beta (5Y)":   fmt_num(hr["beta"]) if hr["beta"] is not None else "—",
                "1Y Return":   fmt_pct_sign(hr["ret_1y"]),
                "5Y Return":   fmt_pct_sign(hr["ret_5y"]),
            })
        st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

        # Whole-portfolio summary
        st.markdown("**Whole Portfolio (weighted by current value)**")
        p1, p2, p3 = st.columns(3)
        p1.metric("Overall Beta",
                  fmt_num(portfolio_beta) if portfolio_beta is not None else "—",
                  help="Weighted average of each holding's 5Y beta, by current value.")
        p2.metric("Weighted 1Y Return", fmt_pct_sign(portfolio_1y))
        p3.metric("Weighted 5Y Return", fmt_pct_sign(portfolio_5y))
        st.caption(
            "Betas are 5Y monthly figures sourced from stockanalysis.com / macroaxis.com (May 2026). "
            "1Y and 5Y returns are price returns from yfinance."
        )

    with st.expander("View / delete individual trades"):
        ind_rows = []
        for _, r in portfolio.iterrows():
            ticker = r["ticker"]
            is_sell = r["quantity"] < 0
            label_action = "SELL" if is_sell else "BUY"
            pos = net_positions.get(ticker, {})
            avg_c = pos.get("avg_cost", 0.0)
            cur_p = prices.get(ticker) or current_price(ticker)
            if is_sell:
                rpl = (float(r["price_paid"]) - avg_c) * abs(float(r["quantity"]))
                ind_rows.append({
                    "Action":        label_action,
                    "Company":       company_label(ticker, infos.get(ticker, {})),
                    "Date":          pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                    "Shares":        fmt_num(abs(r["quantity"])),
                    "Price":         fmt_usd(r["price_paid"]),
                    "Realized P&L":  fmt_usd(rpl),
                    "Current Price": fmt_usd(cur_p),
                    "Gain/Loss":     "—",
                    "Return":        "—",
                })
            else:
                cost_row = float(r["quantity"]) * float(r["price_paid"])
                cur_val_row = float(r["quantity"]) * cur_p if cur_p else None
                gl_row = (cur_val_row - cost_row) if cur_val_row is not None else None
                ret_row = (gl_row / cost_row * 100) if (gl_row is not None and cost_row > 0) else None
                ind_rows.append({
                    "Action":        label_action,
                    "Company":       company_label(ticker, infos.get(ticker, {})),
                    "Date":          pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                    "Shares":        fmt_num(r["quantity"]),
                    "Price":         fmt_usd(r["price_paid"]),
                    "Realized P&L":  "—",
                    "Current Price": fmt_usd(cur_p),
                    "Gain/Loss":     fmt_usd(gl_row),
                    "Return":        fmt_pct_sign(ret_row),
                })
        st.dataframe(pd.DataFrame(ind_rows), use_container_width=True, hide_index=True)

        labels = []
        for _, r in portfolio.iterrows():
            action = "SELL" if r["quantity"] < 0 else "BUY"
            labels.append(
                f"[{action}] {r['ticker']}  {pd.Timestamp(r['date']).strftime('%Y-%m-%d')}  "
                f"{abs(r['quantity']):.3f} shares @ ${r['price_paid']:.2f}"
            )
        sel = st.selectbox("Select trade to remove", range(len(labels)), format_func=lambda i: labels[i])
        if st.button("Delete", type="secondary"):
            delete_trade(portfolio.iloc[sel]["id"])
            st.rerun()

# ─── Performance ─────────────────────────────────────────────────────────────

with tab_p:
    if perf.empty:
        st.warning("Not enough price data.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=perf["date"], y=perf["port_ret"], name="My Portfolio",
                                 line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=perf["date"], y=perf["spy_ret"],  name="S&P 500 (SPY)",
                                 line=dict(color="#ff7f0e", width=2)))
        fig.add_trace(go.Scatter(x=perf["date"], y=perf["alpha"],    name="Alpha",
                                 line=dict(color="#2ca02c", width=2, dash="dot")))
        fig.add_hline(y=0, line_color="gray", line_dash="dash", opacity=0.4)
        fig.update_layout(xaxis_title="Date", yaxis_title="Return (%)",
                          hovermode="x unified", height=420,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        last = perf.iloc[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Portfolio Return", fmt_pct(last["port_ret"]))
        m2.metric("S&P 500 Return",   fmt_pct(last["spy_ret"]))
        m3.metric("Alpha",            fmt_pct_sign(last["alpha"]), delta=fmt_pct_sign(last["alpha"]))
        st.caption(
            f"Portfolio normalized to 0% on {min_date}. "
            f"S&P 500 shows SPY price return from same date."
        )

        st.divider()
        st.subheader("Individual Holdings")
        cols = st.columns(2)
        for i, t in enumerate(active_tickers):
            ticker_trades = portfolio[portfolio["ticker"] == t]
            first_buy = pd.Timestamp(ticker_trades[ticker_trades["quantity"] > 0]["date"].min())
            hist = history.get(t)
            if hist is None or hist.empty:
                continue
            hist_from = hist[hist.index >= first_buy]
            if hist_from.empty:
                continue
            base    = float(hist_from.iloc[0])
            ret_pct_series = (hist_from / base - 1) * 100
            label   = company_label(t, infos.get(t, {}))
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=ret_pct_series.index, y=ret_pct_series.values,
                                       mode="lines", line=dict(width=2),
                                       fill="tozeroy",
                                       fillcolor="rgba(31,119,180,0.1)" if ret_pct_series.iloc[-1] >= 0 else "rgba(214,39,40,0.1)",
                                       line_color="#1f77b4" if ret_pct_series.iloc[-1] >= 0 else "#d62728"))
            fig_s.add_hline(y=0, line_color="gray", line_dash="dash", opacity=0.4)
            fig_s.update_layout(title=label, xaxis_title="", yaxis_title="Return (%)",
                                 height=280, margin=dict(t=40, b=20),
                                 showlegend=False)
            with cols[i % 2]:
                st.plotly_chart(fig_s, use_container_width=True)

        # Drawdown chart moved to bottom of Performance tab
        if not dd_series.empty:
            st.divider()
            st.subheader("Drawdown from Peak")
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=perf["date"].iloc[1:], y=dd_series.values,
                fill="tozeroy", line=dict(color="#d62728", width=1.5), name="Drawdown",
            ))
            fig_dd.update_layout(xaxis_title="Date", yaxis_title="Drawdown (%)",
                                 height=320, yaxis_ticksuffix="%")
            st.plotly_chart(fig_dd, use_container_width=True)
            st.caption("Peak-to-trough decline of total portfolio value over time.")

# ─── Allocation ───────────────────────────────────────────────────────────────

with tab_a:
    if holdings_df.empty:
        st.info("No active holdings.")
    else:
        alloc_rows = []
        total_cost = holdings_df["cost_basis"].sum()
        total_val  = holdings_df["current_value"].sum()

        for _, hr in holdings_df.iterrows():
            cost = hr["cost_basis"]
            val  = hr["current_value"] if pd.notna(hr["current_value"]) else 0.0
            alloc_rows.append({
                "ticker":        hr["ticker"],
                "company":       hr["company"],
                "sector":        hr["sector"],
                "geography":     hr["geography"],
                "cost_basis":    cost,
                "current_value": val,
                "cost_weight":   cost / total_cost * 100 if total_cost > 0 else 0,
                "val_weight":    val  / total_val  * 100 if total_val  > 0 else 0,
            })
        alloc_df = pd.DataFrame(alloc_rows)

        # ─── Per-ticker pies ──────────────────────────────────────────────────
        st.subheader("By Holding")
        col1, col2 = st.columns(2)
        with col1:
            fig_pie1 = px.pie(alloc_df, values="current_value", names="ticker",
                              title="By Current Value")
            fig_pie1.update_traces(textinfo="label+percent")
            st.plotly_chart(fig_pie1, use_container_width=True)
        with col2:
            fig_pie2 = px.pie(alloc_df, values="cost_basis", names="ticker",
                              title="By Cost Basis")
            fig_pie2.update_traces(textinfo="label+percent")
            st.plotly_chart(fig_pie2, use_container_width=True)

        tbl_alloc = []
        for _, r in alloc_df.iterrows():
            tbl_alloc.append({
                "Company":       r["company"],
                "Sector":        r["sector"],
                "Geography":     r["geography"],
                "Current Value": fmt_usd(r["current_value"]),
                "Value Weight":  fmt_pct(r["val_weight"]),
                "Cost Basis":    fmt_usd(r["cost_basis"]),
                "Cost Weight":   fmt_pct(r["cost_weight"]),
            })
        st.dataframe(pd.DataFrame(tbl_alloc), use_container_width=True, hide_index=True)

        # ─── By Sector ────────────────────────────────────────────────────────
        st.divider()
        st.subheader("By Sector")
        sector_agg = alloc_df.groupby("sector", as_index=False).agg(
            current_value=("current_value", "sum"),
            cost_basis=("cost_basis", "sum"),
        )
        sector_agg["val_weight"]  = sector_agg["current_value"] / total_val  * 100 if total_val  > 0 else 0
        sector_agg["cost_weight"] = sector_agg["cost_basis"]    / total_cost * 100 if total_cost > 0 else 0

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_sec1 = px.pie(sector_agg, values="current_value", names="sector",
                              title="By Current Value")
            fig_sec1.update_traces(textinfo="label+percent")
            st.plotly_chart(fig_sec1, use_container_width=True)
        with col_s2:
            fig_sec2 = px.bar(sector_agg.sort_values("current_value", ascending=True),
                              x="current_value", y="sector", orientation="h",
                              title="Sector Allocation ($)", text="current_value")
            fig_sec2.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig_sec2.update_layout(yaxis_title="", xaxis_title="Current Value ($)",
                                   xaxis_tickprefix="$", showlegend=False)
            st.plotly_chart(fig_sec2, use_container_width=True)

        sec_tbl = []
        for _, r in sector_agg.iterrows():
            sec_tbl.append({
                "Sector":        r["sector"],
                "Current Value": fmt_usd(r["current_value"]),
                "Value Weight":  fmt_pct(r["val_weight"]),
                "Cost Basis":    fmt_usd(r["cost_basis"]),
                "Cost Weight":   fmt_pct(r["cost_weight"]),
            })
        st.dataframe(pd.DataFrame(sec_tbl), use_container_width=True, hide_index=True)

        # ─── By Geography ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("By Geography")
        geo_agg = alloc_df.groupby("geography", as_index=False).agg(
            current_value=("current_value", "sum"),
            cost_basis=("cost_basis", "sum"),
        )
        geo_agg["val_weight"]  = geo_agg["current_value"] / total_val  * 100 if total_val  > 0 else 0
        geo_agg["cost_weight"] = geo_agg["cost_basis"]    / total_cost * 100 if total_cost > 0 else 0

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_geo1 = px.pie(geo_agg, values="current_value", names="geography",
                              title="By Current Value",
                              color="geography",
                              color_discrete_map={"United States": "#1f77b4", "Brazil": "#2ca02c"})
            fig_geo1.update_traces(textinfo="label+percent")
            st.plotly_chart(fig_geo1, use_container_width=True)
        with col_g2:
            fig_geo2 = px.bar(geo_agg.sort_values("current_value", ascending=True),
                              x="current_value", y="geography", orientation="h",
                              title="Geographic Allocation ($)", text="current_value",
                              color="geography",
                              color_discrete_map={"United States": "#1f77b4", "Brazil": "#2ca02c"})
            fig_geo2.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig_geo2.update_layout(yaxis_title="", xaxis_title="Current Value ($)",
                                   xaxis_tickprefix="$", showlegend=False)
            st.plotly_chart(fig_geo2, use_container_width=True)

        geo_tbl = []
        for _, r in geo_agg.iterrows():
            geo_tbl.append({
                "Geography":     r["geography"],
                "Current Value": fmt_usd(r["current_value"]),
                "Value Weight":  fmt_pct(r["val_weight"]),
                "Cost Basis":    fmt_usd(r["cost_basis"]),
                "Cost Weight":   fmt_pct(r["cost_weight"]),
            })
        st.dataframe(pd.DataFrame(geo_tbl), use_container_width=True, hide_index=True)

# ─── P&L & Dividends ──────────────────────────────────────────────────────────

with tab_pnl:
    if holdings_df.empty:
        st.info("No active holdings.")
    else:
        pnl_rows = []
        for _, hr in holdings_df.iterrows():
            ticker = hr["ticker"]
            cost   = hr["cost_basis"]
            unr    = hr["gain_loss"] if pd.notna(hr["gain_loss"]) else 0.0
            divs   = hr["dividends"]
            pnl_rows.append({
                "Company":        hr["company"],
                "Shares":         fmt_num(hr["net_qty"]),
                "Cost Basis":     fmt_usd(cost),
                "Current Value":  fmt_usd(hr["current_value"]),
                "Unrealized P&L": fmt_usd(unr),
                "Dividends":      fmt_usd(divs),
                "Total P&L":      fmt_usd(unr + divs),
                "Total Return":   fmt_pct_sign((unr + divs) / cost * 100) if cost > 0 else "—",
                "Yield on Cost":  fmt_pct(hr["yield_on_cost"]) if hr["yield_on_cost"] is not None else "—",
            })
        st.dataframe(pd.DataFrame(pnl_rows), use_container_width=True, hide_index=True)

        tu       = holdings_df["dividends"].sum()
        tv_pnl   = holdings_df["current_value"].sum()
        tc_pnl   = holdings_df["cost_basis"].sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unrealized P&L",  fmt_usd(tv_pnl - tc_pnl))
        c2.metric("Realized P&L",    fmt_usd(total_realized_pnl))
        c3.metric("Total Dividends", fmt_usd(tu))
        c4.metric("Total P&L",       fmt_usd(tv_pnl - tc_pnl + total_realized_pnl + tu))

    # Build dividend events list
    all_div_events = []
    for t in active_tickers:
        divs = div_data.get(t, pd.Series(dtype=float))
        if divs.empty:
            continue
        for div_date, dps in divs.items():
            sh = float(portfolio[
                (portfolio["ticker"] == t) &
                (pd.to_datetime(portfolio["date"]).dt.tz_localize(None) <= div_date)
            ]["quantity"].sum())
            if sh > 0:
                all_div_events.append({"date": div_date, "ticker": t, "amount": dps * sh})

    if all_div_events:
        st.subheader("Cumulative Dividends Received by Stock")
        div_df = pd.DataFrame(all_div_events).sort_values("date")
        div_pivot = div_df.pivot_table(
            index="date", columns="ticker", values="amount", aggfunc="sum"
        ).fillna(0).sort_index()
        div_cumul = div_pivot.cumsum()
        ordered_cols = [c for c in sort_tickers(list(div_cumul.columns)) if c in div_cumul.columns]
        div_cumul = div_cumul[ordered_cols]

        fig_cum = go.Figure()
        for t in ordered_cols:
            fig_cum.add_trace(go.Scatter(
                x=div_cumul.index,
                y=div_cumul[t],
                name=company_label(t, infos.get(t, {})),
                mode="lines",
                stackgroup="one",
                line=dict(width=0.5),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra>%{fullData.name}</extra>",
            ))
        fig_cum.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative Dividends ($)",
            yaxis_tickprefix="$",
            yaxis_tickformat=",.2f",
            height=420,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cum, use_container_width=True)
        st.caption("Each stock's contribution stacks on top — total height at any date = total dividends received to that point.")
    else:
        st.info("No dividend payments found since your purchase dates.")

# ─── Net Worth ────────────────────────────────────────────────────────────────

with tab_nw:
    if perf.empty:
        st.warning("Not enough price data.")
    else:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=perf["date"], y=perf["port_val"],  name="Portfolio Value",
                                  fill="tozeroy", line=dict(color="#1f77b4", width=2)))
        fig2.add_trace(go.Scatter(x=perf["date"], y=perf["port_cost"], name="Invested Capital",
                                  line=dict(color="#aaaaaa", width=2, dash="dash")))
        fig2.update_layout(xaxis_title="Date", yaxis_title="Value ($)",
                           hovermode="x unified", height=480,
                           yaxis_tickprefix="$", yaxis_tickformat=",.0f",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig2, use_container_width=True)
        last = perf.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Value",    fmt_usd(last["port_val"]))
        c2.metric("Invested Capital", fmt_usd(last["port_cost"]))
        c3.metric("Unrealized Gain",  fmt_usd(last["port_val"] - last["port_cost"]))

# ─── Alpha ────────────────────────────────────────────────────────────────────

with tab_alpha:
    st.subheader("Alpha vs. S&P 500 (SPY)")
    st.caption(
        "Alpha = Portfolio cumulative return - SPY cumulative return (simple excess return). "
        "Portfolio return uses current net positions valued at historical prices."
    )

    alpha_data = []
    for period_label, p_ret, s_ret, alph in [
        ("Since Inception", alpha_inception_port, alpha_inception_spy, alpha_inception),
        (f"Since Feb 1, 2026", alpha_feb_port, alpha_feb_spy, alpha_feb),
    ]:
        alpha_data.append({
            "Period":            period_label,
            "Portfolio Return":  fmt_pct_sign(p_ret)  if p_ret  is not None else "N/A",
            "S&P 500 Return":    fmt_pct_sign(s_ret)  if s_ret  is not None else "N/A",
            "Alpha":             fmt_pct_sign(alph)   if alph   is not None else "N/A",
        })

    alpha_df = pd.DataFrame(alpha_data)
    st.dataframe(alpha_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Since Inception")
        if alpha_inception is not None:
            a1, a2, a3 = st.columns(3)
            a1.metric("Portfolio", fmt_pct_sign(alpha_inception_port))
            a2.metric("SPY",       fmt_pct_sign(alpha_inception_spy))
            a3.metric("Alpha",     fmt_pct_sign(alpha_inception),
                      delta=fmt_pct_sign(alpha_inception))
        else:
            st.warning("Not enough history to compute since-inception alpha.")

    with col2:
        st.subheader("Since Feb 1, 2026")
        if alpha_feb is not None:
            b1, b2, b3 = st.columns(3)
            b1.metric("Portfolio", fmt_pct_sign(alpha_feb_port))
            b2.metric("SPY",       fmt_pct_sign(alpha_feb_spy))
            b3.metric("Alpha",     fmt_pct_sign(alpha_feb),
                      delta=fmt_pct_sign(alpha_feb))
        else:
            st.warning("Not enough history to compute alpha since Feb 1, 2026.")

    if not perf.empty:
        st.divider()
        st.subheader("Alpha Over Time (vs. Since Inception)")
        fig_alpha = go.Figure()
        fig_alpha.add_trace(go.Scatter(
            x=perf["date"], y=perf["alpha"],
            fill="tozeroy",
            fillcolor="rgba(44,160,44,0.15)",
            line=dict(color="#2ca02c", width=2),
            name="Alpha",
        ))
        fig_alpha.add_hline(y=0, line_color="gray", line_dash="dash", opacity=0.5)
        fig_alpha.update_layout(
            xaxis_title="Date", yaxis_title="Alpha (%)",
            yaxis_ticksuffix="%", height=320,
            hovermode="x unified",
        )
        st.plotly_chart(fig_alpha, use_container_width=True)
