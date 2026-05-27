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
TICKER_ORDER = ["SPY", "DIA", "VINP", "NU", "BRK-B", "MSA", "DUK"]

# Sector classification (ETFs kept as their own category)
SECTOR_MAP = {
    "SPY":   "Broad US Equity (ETF)",
    "DIA":   "Broad US Equity (ETF)",
    "VINP":  "Financial Services",
    "NU":    "Financial Services",
    "BRK-B": "Financial Services",
    "MSA":   "Industrials",
    "DUK":   "Utilities",
}

# Geography classification
GEOGRAPHY_MAP = {
    "SPY":   "United States",
    "DIA":   "United States",
    "VINP":  "Brazil",
    "NU":    "Brazil",
    "BRK-B": "United States",
    "MSA":   "United States",
    "DUK":   "United States",
}


def sort_tickers(tickers):
    """Sort tickers per TICKER_ORDER, unknown tickers alphabetically at the end."""
    in_order = [t for t in TICKER_ORDER if t in tickers]
    others   = sorted([t for t in tickers if t not in TICKER_ORDER])
    return in_order + others


def classify_sector(ticker, info=None):
    """Return sector for ticker — uses SECTOR_MAP first, falls back to yfinance info."""
    if ticker in SECTOR_MAP:
        return SECTOR_MAP[ticker]
    if info:
        return info.get("sector") or "Unknown"
    return "Unknown"


def classify_geography(ticker, info=None):
    """Return geography for ticker — uses GEOGRAPHY_MAP first, falls back to yfinance info."""
    if ticker in GEOGRAPHY_MAP:
        return GEOGRAPHY_MAP[ticker]
    if info:
        return info.get("country") or "Unknown"
    return "Unknown"


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
    """quantity can be negative for sells."""
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
    """
    Compute per-ticker net quantity, avg cost (buys only), realized P&L.
    Returns a dict keyed by ticker with keys: net_qty, avg_cost, realized_pnl.
    Only tickers with net_qty > 0 are active holdings.
    """
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
    """Download multiple tickers at once and return a DataFrame of Close prices."""
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


@st.cache_data(ttl=3600)
def get_ff4_factors(start):
    try:
        import pandas_datareader.data as web
        f3  = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start=start)[0] / 100
        mom = web.DataReader("F-F_Momentum_Factor_daily",       "famafrench", start=start)[0] / 100
        ff4 = f3.join(mom[["Mom"]])
        ff4.index = pd.to_datetime(ff4.index)
        if ff4.index.tz is not None:
            ff4.index = ff4.index.tz_localize(None)
        return ff4
    except Exception:
        return pd.DataFrame()


def run_ff4(returns, ff4):
    if ff4.empty:
        return {}
    common = returns.index.intersection(ff4.index)
    if len(common) < 30:
        return {}
    y  = (returns.loc[common] - ff4.loc[common, "RF"]).values
    Xd = ff4.loc[common, ["Mkt-RF", "SMB", "HML", "Mom"]].values
    Xc = np.column_stack([np.ones(len(Xd)), Xd])
    coef, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None)
    y_hat  = Xc @ coef
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "Alpha (ann. %)": round(coef[0] * 252 * 100, 2),
        "Beta (Market)":  round(coef[1], 3),
        "SMB":            round(coef[2], 3),
        "HML":            round(coef[3], 3),
        "UMD":            round(coef[4], 3),
        "R²":             round(r2, 3),
    }


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
# Apply custom ordering
active_tickers = sort_tickers([t for t, v in net_positions.items() if v["net_qty"] > 0])
all_tickers    = portfolio["ticker"].unique().tolist()

min_date = portfolio["date"].min().date()
today    = date.today()
end_str  = str(today + timedelta(days=1))

with st.spinner("Loading market data, please wait..."):
    prices   = {t: current_price(t)  for t in active_tickers}
    infos    = {t: get_ticker_info(t) for t in active_tickers}
    history  = {t: fetch_prices(t, str(min_date), end_str) for t in all_tickers + ["SPY"]}
    div_data = {t: get_dividends(t)   for t in active_tickers}
    ff4      = get_ff4_factors(str(min_date))

    all_needed_tickers = tuple(sorted(set(active_tickers + ["SPY"])))
    bulk_history = get_price_history(all_needed_tickers, str(min_date), end_str)

    ytd_start     = date(today.year, 1, 1)
    five_yr_start = today - timedelta(days=5 * 365)

    hist_ytd = {}
    hist_5y  = {}
    for t in active_tickers:
        hist_ytd[t] = fetch_prices(t, str(ytd_start - timedelta(days=5)), end_str)
        hist_5y[t]  = fetch_prices(t, str(five_yr_start), end_str)

    hist_ytd["SPY"] = fetch_prices("SPY", str(ytd_start - timedelta(days=5)), end_str)
    hist_5y["SPY"]  = fetch_prices("SPY", str(five_yr_start), end_str)

# ── Dividends & yield-on-cost helpers ─────────────────────────────────────────

def ttm_dividends_per_share(ticker):
    """Trailing 12-month dividends per share (sum of dividend payments)."""
    divs = div_data.get(ticker, pd.Series(dtype=float))
    if divs.empty:
        return 0.0
    cutoff = pd.Timestamp(today - timedelta(days=365))
    return float(divs[divs.index >= cutoff].sum())


# ── Enrich rows (active holdings only) ───────────────────────────────────────

# Dividends actually received by Fernando (based on his holding history)
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

    # Dividends received and yield-on-cost
    divs_received = div_by_ticker.get(ticker, 0.0)
    ttm_dps       = ttm_dividends_per_share(ticker)
    yoc_pct       = (ttm_dps / avg_cost * 100) if avg_cost > 0 else None
    total_pnl     = (gain_loss + divs_received) if gain_loss is not None else None
    total_ret_pct = (total_pnl / cost_bas * 100) if (total_pnl is not None and cost_bas > 0) else None

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

# ── Risk Metrics (rebuilt with weighted individual betas + 1Y total return) ──

risk_vals = {}
dd_series = pd.Series(dtype=float)
ticker_risk = {}  # per-ticker 1Y total return, vol, beta

if len(perf) >= 5:
    port_daily = perf.set_index("date")["port_val"].pct_change().dropna()
    spy_daily  = spy_hist.pct_change().dropna() if not spy_hist.empty else pd.Series(dtype=float)
    rf_d       = RF_ANNUAL / 252

    # Portfolio annual volatility from daily portfolio returns
    ann_vol = port_daily.std() * np.sqrt(252) * 100

    # Per-ticker 1Y total return (price + TTM dividends), vol, and beta
    hist_1y_start = str(today - timedelta(days=365))
    total_port_val = sum(
        (net_positions[t]["net_qty"] * prices[t])
        for t in active_tickers if prices.get(t) is not None
    )

    weighted_ann_ret = 0.0
    weighted_beta_num = 0.0
    weighted_beta_den = 0.0

    for t in active_tickers:
        cur_p = prices.get(t)
        if cur_p is None:
            continue
        h1y = fetch_prices(t, hist_1y_start, end_str)
        if h1y.empty:
            continue
        try:
            start_p = float(h1y.iloc[0])
            end_p   = float(h1y.iloc[-1])
            t_price_ret = (end_p / start_p) - 1
        except Exception:
            continue

        # TTM dividend yield component (use start-of-period price for yield)
        ttm_dps = ttm_dividends_per_share(t)
        t_div_yield = (ttm_dps / start_p) if start_p > 0 else 0.0
        t_total_ret = t_price_ret + t_div_yield

        t_daily   = h1y.pct_change().dropna()
        t_ann_vol = float(t_daily.std() * np.sqrt(252) * 100)

        # Individual beta from yfinance (5Y monthly regression vs S&P)
        t_beta = infos.get(t, {}).get("beta")
        try:
            t_beta = float(t_beta) if t_beta is not None else None
        except (ValueError, TypeError):
            t_beta = None

        # For ETFs that track the S&P, override to ~1
        if t == "SPY":
            t_beta = 1.00

        ticker_risk[t] = {
            "ann_ret":  t_total_ret * 100,
            "ann_vol":  t_ann_vol,
            "beta":     t_beta,
            "ttm_dps":  ttm_dps,
        }

        weight = (net_positions[t]["net_qty"] * cur_p) / total_port_val if total_port_val > 0 else 0.0
        weighted_ann_ret += t_total_ret * weight

        if t_beta is not None:
            weighted_beta_num += t_beta * weight
            weighted_beta_den += weight

    ann_ret_pct = weighted_ann_ret * 100

    # Weighted beta — renormalize if any tickers missing beta
    if weighted_beta_den > 0:
        weighted_beta = weighted_beta_num / weighted_beta_den
    else:
        weighted_beta = np.nan

    # Sharpe = (annualized return - rf) / annualized vol
    if ann_vol > 0:
        sharpe = (ann_ret_pct - RF_ANNUAL * 100) / ann_vol
    else:
        sharpe = np.nan

    risk_vals = {
        "Ann. Return":     f"{ann_ret_pct:.2f}%",
        "Ann. Volatility": f"{ann_vol:.2f}%",
        "Sharpe Ratio":    f"{sharpe:.2f}",
        "Beta (weighted)": f"{weighted_beta:.2f}" if not np.isnan(weighted_beta) else "—",
    }

    cum      = (1 + port_daily.fillna(0)).cumprod()
    roll_max = cum.expanding().max()
    dd_series = (cum - roll_max) / roll_max * 100

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

tab_h, tab_p, tab_a, tab_pnl, tab_r, tab_sec, tab_nw, tab_alpha = st.tabs([
    "Holdings", "Performance", "Allocation",
    "P&L & Dividends", "Risk & Factors", "Securities", "Net Worth", "Alpha",
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
            ret_pct = (hist_from / base - 1) * 100
            label   = company_label(t, infos.get(t, {}))
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=ret_pct.index, y=ret_pct.values,
                                       mode="lines", line=dict(width=2),
                                       fill="tozeroy",
                                       fillcolor="rgba(31,119,180,0.1)" if ret_pct.iloc[-1] >= 0 else "rgba(214,39,40,0.1)",
                                       line_color="#1f77b4" if ret_pct.iloc[-1] >= 0 else "#d62728"))
            fig_s.add_hline(y=0, line_color="gray", line_dash="dash", opacity=0.4)
            fig_s.update_layout(title=label, xaxis_title="", yaxis_title="Return (%)",
                                 height=280, margin=dict(t=40, b=20),
                                 showlegend=False)
            with cols[i % 2]:
                st.plotly_chart(fig_s, use_container_width=True)

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

        # ─── Per-ticker pies (original) ───────────────────────────────────────
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

    # Build dividend events list (one event per pay date per ticker)
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

        # Pivot to wide: rows = dates, cols = tickers, values = $ received
        div_pivot = div_df.pivot_table(
            index="date", columns="ticker", values="amount", aggfunc="sum"
        ).fillna(0).sort_index()

        # Cumulative sum so the line/area only ever climbs
        div_cumul = div_pivot.cumsum()

        # Order columns per TICKER_ORDER
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

# ─── Risk & Factors ───────────────────────────────────────────────────────────

with tab_r:
    if not risk_vals:
        st.warning("Need more price history to compute risk metrics.")
    else:
        st.subheader("Portfolio Risk Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ann. Return",      risk_vals["Ann. Return"])
        c2.metric("Ann. Volatility",  risk_vals["Ann. Volatility"])
        c3.metric("Sharpe Ratio",     risk_vals["Sharpe Ratio"])
        c4.metric("Beta (weighted)",  risk_vals["Beta (weighted)"])
        st.caption(
            f"Assumed risk-free rate: {RF_ANNUAL*100:.1f}% p.a. | "
            "Ann. Return = value-weighted 1Y total return (price + TTM dividends) per holding. "
            "Ann. Volatility = stdev of portfolio daily returns × √252. "
            "Sharpe = (Ann. Return − Rf) ÷ Ann. Volatility. "
            "Beta = value-weighted average of each holding's 5Y beta (from yfinance, sourced from S&P regression)."
        )

        # Per-ticker risk breakdown — now includes individual beta
        if ticker_risk:
            st.subheader("Per-Ticker Risk (1Y)")
            tk_risk_rows = []
            for t in active_tickers:
                if t not in ticker_risk:
                    continue
                rv = ticker_risk[t]
                tk_risk_rows.append({
                    "Company":         company_label(t, infos.get(t, {})),
                    "1Y Total Return": fmt_pct_sign(rv["ann_ret"]),
                    "1Y Ann. Vol":     fmt_pct(rv["ann_vol"]),
                    "Beta (5Y)":       fmt_num(rv["beta"]) if rv["beta"] is not None else "—",
                })
            st.dataframe(pd.DataFrame(tk_risk_rows), use_container_width=True, hide_index=True)

        st.subheader("Drawdown from Peak")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=perf["date"].iloc[1:], y=dd_series.values,
            fill="tozeroy", line=dict(color="#d62728", width=1.5), name="Drawdown",
        ))
        fig_dd.update_layout(xaxis_title="Date", yaxis_title="Drawdown (%)",
                             height=280, yaxis_ticksuffix="%")
        st.plotly_chart(fig_dd, use_container_width=True)

        if not ff4.empty:
            st.subheader("Fama-French 4-Factor Exposures")
            st.caption("OLS regression on Market, SMB, HML, UMD. Requires >= 30 trading days per security.")
            ff4_rows = []
            for t in active_tickers:
                h = history.get(t)
                if h is None or h.empty:
                    continue
                result = run_ff4(h.pct_change().dropna(), ff4)
                if result:
                    ff4_rows.append({"Company": company_label(t, infos.get(t, {})), **result})
            port_daily_ff4 = perf.set_index("date")["port_val"].pct_change().dropna()
            port_result    = run_ff4(port_daily_ff4, ff4)
            if port_result:
                ff4_rows.append({"Company": "PORTFOLIO (combined)", **port_result})
            if ff4_rows:
                st.dataframe(pd.DataFrame(ff4_rows), use_container_width=True, hide_index=True)
                st.caption(
                    "**Beta > 1**: amplifies market moves. **SMB > 0**: small-cap tilt. "
                    "**HML > 0**: value tilt. **UMD > 0**: momentum tilt."
                )

# ─── Securities Summary ───────────────────────────────────────────────────────

with tab_sec:
    st.subheader("Securities Overview")

    spy_info = get_ticker_info("SPY")
    h_ytd_spy = hist_ytd.get("SPY", pd.Series(dtype=float))
    spy_ytd_ret = None
    if not h_ytd_spy.empty:
        spy_ytd_slice = h_ytd_spy[h_ytd_spy.index >= pd.Timestamp(ytd_start)]
        if not spy_ytd_slice.empty:
            spy_ytd_ret = (float(h_ytd_spy.iloc[-1]) / float(spy_ytd_slice.iloc[0]) - 1) * 100

    h_5y_spy = hist_5y.get("SPY", pd.Series(dtype=float))
    spy_5y_ret = None
    if not h_5y_spy.empty and len(h_5y_spy) > 10:
        spy_5y_ret = (float(h_5y_spy.iloc[-1]) / float(h_5y_spy.iloc[0]) - 1) * 100

    with st.expander("S&P 500 Benchmark (SPY)", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Market Cap", fmt_mcap(spy_info.get("marketCap")))
        m2.metric("Beta", "1.00")
        m3.metric("YTD Return", fmt_pct_sign(spy_ytd_ret) if spy_ytd_ret is not None else "—")
        m4.metric("5Y Return",  fmt_pct_sign(spy_5y_ret)  if spy_5y_ret  is not None else "—")

    for ticker in active_tickers:
        info  = infos.get(ticker, {})
        label = company_label(ticker, info)

        desc = info.get("longBusinessSummary") or info.get("description") or ""
        sentences = [s.strip() for s in desc.split(". ") if s.strip()]
        short_desc = ". ".join(sentences[:2]) + ("." if len(sentences) >= 2 else "")

        h_ytd = hist_ytd.get(ticker, pd.Series(dtype=float))
        ytd_ret = None
        if not h_ytd.empty:
            ytd_slice = h_ytd[h_ytd.index >= pd.Timestamp(ytd_start)]
            if not ytd_slice.empty:
                ytd_ret = (float(h_ytd.iloc[-1]) / float(ytd_slice.iloc[0]) - 1) * 100

        h_5y = hist_5y.get(ticker, pd.Series(dtype=float))
        five_yr_ret = None
        if not h_5y.empty and len(h_5y) > 10:
            try:
                five_yr_ret = (float(h_5y.iloc[-1]) / float(h_5y.iloc[0]) - 1) * 100
            except Exception:
                five_yr_ret = None

        mktcap = info.get("marketCap")
        beta   = info.get("beta")

        with st.expander(label, expanded=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Market Cap", fmt_mcap(mktcap))
            m2.metric("Beta",       fmt_num(beta) if beta else "—")
            m3.metric("YTD Return", fmt_pct_sign(ytd_ret)    if ytd_ret    is not None else "N/A")
            m4.metric("5Y Return",  fmt_pct_sign(five_yr_ret) if five_yr_ret is not None else "N/A")
            if short_desc:
                st.caption(short_desc)

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
