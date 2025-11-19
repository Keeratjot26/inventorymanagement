# Final_streamlit_ready.py
# Smart Inventory Analytics - MongoDB Integrated Version (patched for Streamlit Cloud)
# Run with: streamlit run Final_streamlit_ready.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta
import time
from pymongo import MongoClient, errors as pymongo_errors
import sys

# quick debug to surface Python env early in logs
st.set_page_config(page_title="Smart Inventory Analytics", layout="wide", initial_sidebar_state="expanded")
st.write("Environment: Streamlit startup - Python", sys.version.split()[0])

# ---------------- CONFIG ----------------
MONGO_URI = "mongodb+srv://arshnoorkaur:Arshnoor1740@cluster0.ea5r0.mongodb.net/inventory?retryWrites=true&w=majority"
DB_NAME = "inventory"
PRODUCTS_COLLECTION = "products"
SALES_COLLECTION = "sales"
BACKEND_LABEL = "MongoDB Atlas – inventory DB"

# ---------------- STYLES ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg,#f8feff 0%, #ffffff 50%);
    }
    .app-title {
        font-size:34px;
        font-weight:700;
        color:#003c3c;
        padding-bottom:8px;
    }
    .kpi {
        background: linear-gradient(180deg,#ffffff, #f2fbfb);
        border-radius:12px;
        padding:18px;
        box-shadow: 0 6px 18px rgba(6,60,60,0.06);
        border-left: 6px solid rgba(0,140,140,0.95);
        transition: transform .12s ease-in-out;
    }
    .kpi:hover { transform: translateY(-4px); }
    .kpi-label { color:#006565; font-size:14px; opacity:0.9; }
    .kpi-value { color:#004d4d; font-size:24px; font-weight:700; margin-top:6px; }
    .card {
        background: #ffffff;
        border-radius:12px;
        padding:18px;
        box-shadow: 0 6px 20px rgba(5,60,60,0.04);
        border: 1px solid #e8f7f7;
    }
    .badge-safe { background:#e6fff8; color:#006d6d; padding:6px 10px; border-radius:8px; font-weight:600; }
    .badge-near { background:#fff6e6; color:#8a5a00; padding:6px 10px; border-radius:8px; font-weight:600; }
    .badge-exp { background:#ffecec; color:#9b1e1e; padding:6px 10px; border-radius:8px; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- UTIL: MONGODB DATA LOADER ----------------
@st.cache_data(ttl=120)
def load_mongo_data():
    """
    Loads products and sales collections, merges them (if possible) and returns a dataframe.
    Returns (df, None) on success, (None, error_message) on failure.
    """
    try:
        # use a short server selection timeout so app fails fast if DB is unreachable
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Force a connection attempt (will raise if cannot connect/authenticate)
        client.server_info()
    except Exception as e:
        return None, f"MongoDB connection failed: {e}"

    try:
        db = client[DB_NAME]
        products_collection = db[PRODUCTS_COLLECTION]
        sales_collection = db[SALES_COLLECTION]

        products_df = pd.DataFrame(list(products_collection.find()))
        sales_df = pd.DataFrame(list(sales_collection.find()))

        products_df = products_df.drop(columns=['_id'], errors='ignore')
        sales_df = sales_df.drop(columns=['_id'], errors='ignore')

        if sales_df.empty and products_df.empty:
            return None, "Both products and sales collections are empty."

        # Try to merge on a reasonable key
        merge_key = None
        for cand in ['product_id', 'product', 'name', 'product_name', 'sku', 'code']:
            if cand in sales_df.columns and cand in products_df.columns:
                merge_key = cand
                break

        if merge_key:
            df_merged = pd.merge(
                sales_df,
                products_df,
                on=merge_key,
                how='left',
                suffixes=('', '_prod')
            )
        else:
            df_merged = sales_df.copy()

        return df_merged, None

    except Exception as e:
        return None, f"Error when querying MongoDB collections: {e}"

# ---------------- SMALL HELPER TO AUTO-DETECT COLUMNS ----------------
def find_first_col(df, keywords):
    for kw in keywords:
        for c in df.columns:
            if kw in c.lower():
                return c
    return None

# ---------------- DATA CLEANING FUNCTION ----------------
def clean_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None:
        return pd.DataFrame()  # return empty df to avoid crashes downstream
    df = df_raw.copy()
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_").lower())

    if 'product' not in df.columns:
        prod_col = find_first_col(df, ['product', 'name', 'item', 'sku'])
        if prod_col:
            df.rename(columns={prod_col: 'product'}, inplace=True)

    if 'product' not in df.columns:
        df['product'] = df.index.astype(str)

    if 'quantity' not in df.columns:
        qty_col = find_first_col(df, ['quantity', 'qty', 'units', 'sold'])
        if qty_col:
            df.rename(columns={qty_col: 'quantity'}, inplace=True)
        else:
            df['quantity'] = 0

    if 'selling_price' not in df.columns:
        sp_col = find_first_col(df, ['selling_price', 'sell_price', 'unit_price', 'price', 'sale_price'])
        if sp_col:
            df.rename(columns={sp_col: 'selling_price'}, inplace=True)
        else:
            df['selling_price'] = 0.0

    if 'purchase_cost' not in df.columns:
        pc_col = find_first_col(df, ['purchase_cost', 'buy_price', 'cost'])
        if pc_col:
            df.rename(columns={pc_col: 'purchase_cost'}, inplace=True)
        else:
            df['purchase_cost'] = 0.0

    if 'stock_level' not in df.columns:
        stock_col = find_first_col(df, ['stock', 'inventory', 'on_hand', 'available'])
        if stock_col:
            df.rename(columns={stock_col: 'stock_level'}, inplace=True)
        else:
            df['stock_level'] = 0

    for col in ['selling_price', 'purchase_cost', 'quantity', 'stock_level']:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

    if 'invoicedate' not in df.columns:
        dt_col = find_first_col(df, ['invoice_date', 'sale_date', 'date', 'order_date'])
        if dt_col:
            df['invoicedate'] = pd.to_datetime(df[dt_col], errors='coerce')
    else:
        df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')

    if 'expiry_date' in df.columns:
        df['expiry_date'] = pd.to_datetime(df['expiry_date'], errors='coerce')

    total_sales_existing = find_first_col(df, ['total_sales', 'sales_amount', 'amount', 'revenue'])
    if total_sales_existing and total_sales_existing != 'total_sales':
        df['total_sales'] = pd.to_numeric(df[total_sales_existing], errors='coerce').fillna(0)
    else:
        df['total_sales'] = df['selling_price'] * df['quantity']

    profit_existing = find_first_col(df, ['total_profit', 'profit'])
    if profit_existing and profit_existing != 'total_profit':
        df['total_profit'] = pd.to_numeric(df[profit_existing], errors='coerce').fillna(0)
    else:
        df['profit'] = df['selling_price'] - df['purchase_cost']
        df['total_profit'] = df['profit'] * df['quantity']

    return df

# ---------------- LOAD DATA ----------------
with st.spinner("Loading data from MongoDB Atlas..."):
    df_raw, err = load_mongo_data()

if err:
    st.error("Failed to load data from MongoDB:")
    st.error(err)
    st.stop()

df = clean_df(df_raw)

# Early safety check
if df is None or df.empty:
    st.warning("Data is empty after loading/cleaning. Dashboard will show placeholders.")
    df = pd.DataFrame(columns=['product', 'quantity', 'selling_price', 'purchase_cost', 'stock_level', 'invoicedate', 'total_sales', 'total_profit'])

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")

with st.sidebar.expander("Data Source"):
    st.markdown("**Backend:** " + BACKEND_LABEL)
    if st.button("🔁 Refresh data from MongoDB"):
        load_mongo_data.cache_clear()
        df_raw_new, err2 = load_mongo_data()
        if err2:
            st.sidebar.error("Refresh failed: " + err2)
        else:
            df = clean_df(df_raw_new)
            st.sidebar.success("Data refreshed")
            time.sleep(0.5)
            st.experimental_rerun()

# ---- Date range filter ----
if 'invoicedate' in df.columns and not df['invoicedate'].isna().all():
    min_date = df['invoicedate'].min()
    max_date = df['invoicedate'].max()
    date_range = st.sidebar.date_input("Invoice Date Range", value=(min_date, max_date))
else:
    date_range = None

shelf_life_days = st.sidebar.number_input("Assumed shelf life (days)", min_value=1, max_value=365, value=30)

if 'invoicedate' in df.columns and not df['invoicedate'].isna().all():
    default_analysis_date = df['invoicedate'].max().date()
else:
    default_analysis_date = datetime.today().date()

analysis_date = st.sidebar.date_input("Assume today's date is", value=default_analysis_date)
analysis_date = pd.to_datetime(analysis_date)

# ---- Apply date filter ----
df_filtered = df.copy()
if date_range is not None and 'invoicedate' in df_filtered.columns:
    start, end = date_range
    df_filtered = df_filtered[(df_filtered['invoicedate'] >= pd.to_datetime(start)) & (df_filtered['invoicedate'] <= pd.to_datetime(end))]

# ---------------- HEADER ----------------
st.markdown('<div class="app-title">Smart Inventory Analytics Dashboard</div>', unsafe_allow_html=True)

# ---------------- TABS ----------------
tab_dashboard, tab_forecast, tab_alerts, tab_admin = st.tabs(["📊 Dashboard", "🔮 Forecast", "⚠️ Alerts", "⚙️ Admin"])

# ---------------- DASHBOARD TAB ----------------
with tab_dashboard:
    total_revenue = float(df_filtered['total_sales'].sum())
    total_profit = float(df_filtered['total_profit'].sum())
    if 'stock_level' in df_filtered.columns and df_filtered['stock_level'].replace(0, np.nan).mean() > 0:
        inventory_turnover = df_filtered['quantity'].sum() / df_filtered['stock_level'].replace(0, np.nan).mean()
    else:
        inventory_turnover = np.nan
    products_at_risk = df_filtered[df_filtered['stock_level'] <= df_filtered.get('quantity', 0)].shape[0]

    k1, k2, k3, k4 = st.columns([1.8, 1.8, 1.2, 1.2])
    k1.markdown(f'<div class="kpi"><div class="kpi-label">Total Revenue</div><div class="kpi-value">₹{total_revenue:,.0f}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi"><div class="kpi-label">Total Profit</div><div class="kpi-value">₹{total_profit:,.0f}</div></div>', unsafe_allow_html=True)
    itv_display = f"{inventory_turnover:.2f}" if not np.isnan(inventory_turnover) else "N/A"
    k3.markdown(f'<div class="kpi"><div class="kpi-label">Inventory Turnover</div><div class="kpi-value">{itv_display}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi"><div class="kpi-label">Products at Risk</div><div class="kpi-value">{products_at_risk}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([3, 1.25])

    # ---- Left: Monthly trend & heatmap ----
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Monthly Sales Trend")

        if 'invoicedate' in df_filtered.columns:
            monthly = (df_filtered.dropna(subset=['invoicedate']).groupby(df_filtered['invoicedate'].dt.to_period('M'))['quantity'].sum().reset_index())
            if not monthly.empty:
                monthly['month_str'] = monthly['invoicedate'].dt.strftime('%Y-%m')
                fig = px.line(monthly, x='month_str', y='quantity', markers=True, labels={'month_str': 'Month', 'quantity': 'Units Sold'})
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No valid invoice dates to show trend.")
        else:
            st.info("No 'invoicedate' column in data.")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- TOP 30 PRODUCTS HEATMAP ----------
        st.markdown('<div class="card" style="margin-top:18px">', unsafe_allow_html=True)
        st.subheader("Profitability Heatmap (Top 30 Products)")

        if 'invoicedate' in df_filtered.columns and 'product' in df_filtered.columns:
            df_heat = df_filtered.dropna(subset=['invoicedate']).copy()
            df_heat['month'] = df_heat['invoicedate'].dt.month

            prod_profit = (df_heat.groupby('product')['total_profit'].sum().abs().sort_values(ascending=False))
            top_products = prod_profit.head(30).index
            df_heat_top = df_heat[df_heat['product'].isin(top_products)]
            heat = df_heat_top.pivot_table(values='total_profit', index='product', columns='month', aggfunc='sum', fill_value=0)
            heat = heat.loc[top_products]
            if not heat.empty:
                fig2, ax2 = plt.subplots(figsize=(10, max(3, min(10, heat.shape[0] * 0.35))))
                sns.heatmap(heat, cmap='YlGnBu', ax=ax2, robust=True)
                ax2.set_xlabel("Month")
                ax2.set_ylabel("Product")
                st.pyplot(fig2, width="stretch")
            else:
                st.info("Not enough data to render heatmap.")
        else:
            st.info("Need 'product' and 'invoicedate' columns for heatmap.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Right: Expiry (derived) & stock levels ----
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Expiry Alerts (Derived)")

        if 'invoicedate' in df_filtered.columns:
            df_exp = df_filtered.dropna(subset=['invoicedate']).copy()
            if not df_exp.empty:
                df_exp['expiry_date'] = df_exp['invoicedate'] + pd.to_timedelta(shelf_life_days, unit='D')
                df_exp['days_to_expiry'] = (df_exp['expiry_date'] - analysis_date).dt.days
                df_exp = df_exp[df_exp['days_to_expiry'].between(-60, 60)]
                if df_exp.empty:
                    st.info("No products near expiry for the selected date.")
                else:
                    alerts = df_exp.sort_values('days_to_expiry').head(10)
                    for _, r in alerts.iterrows():
                        days = int(r['days_to_expiry'])
                        if days <= 0:
                            badge = '<span class="badge-exp">Expired</span>'
                        elif days <= 30:
                            badge = '<span class="badge-near">Near Expiry</span>'
                        else:
                            badge = '<span class="badge-safe">Safe</span>'
                        product_name = r.get('product', '(No name)')
                        expiry_str = r['expiry_date'].date().isoformat()
                        st.markdown(f"**{product_name}** — expiry: {expiry_str} (*{days} days from selected date*) &nbsp; {badge}", unsafe_allow_html=True)
            else:
                st.info("No valid invoice dates to derive expiry.")
        else:
            st.info("No 'invoicedate' column available to derive expiry.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Inventory Levels")

        if 'stock_level' in df_filtered.columns and 'product' in df_filtered.columns:
            inv = (df_filtered.groupby('product')['stock_level'].sum().sort_values(ascending=False).reset_index())
            if not inv.empty:
                fig3 = px.bar(inv.head(10), x='product', y='stock_level')
                st.plotly_chart(fig3, width="stretch")
            else:
                st.info("No stock data to plot.")
        else:
            st.info("Need 'product' and 'stock_level' for inventory chart.")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FORECAST TAB (7-DAY MOVING AVERAGE) ----------------
with tab_forecast:
    st.header("Demand Trend (7-Day Moving Average)")

    product_list = df['product'].unique().tolist() if 'product' in df.columns else []
    sel_product = st.selectbox("Select product", ["All"] + product_list)

    history_days = st.slider("History window (days)", 30, 365, 90)

    if 'invoicedate' in df.columns:
        base = df.dropna(subset=['invoicedate']).copy()
        if sel_product != "All" and 'product' in base.columns:
            base = base[base['product'] == sel_product]
        if not base.empty:
            ddf = (base.groupby('invoicedate')['quantity'].sum().reset_index().sort_values('invoicedate'))
            if not ddf.empty:
                ddf['MA7'] = ddf['quantity'].rolling(window=7, min_periods=1).mean()
                last_date = ddf['invoicedate'].max()
                show = ddf[ddf['invoicedate'] >= (last_date - pd.Timedelta(days=history_days))]
                title_name = "All Products" if sel_product == "All" else sel_product
                figf = px.line(show, x="invoicedate", y=["quantity", "MA7"], labels={"value": "Units", "variable": "Series", "invoicedate": "Date"}, title=f"Demand & 7-Day Moving Average — {title_name}")
                st.plotly_chart(figf, width="stretch")
            else:
                st.info("No demand data to show.")
        else:
            st.info("No data after filtering by product/date.")
    else:
        st.info("No 'invoicedate' column found in data.")

# ---------------- ALERTS TAB ----------------
with tab_alerts:
    st.header("Dynamic What-If Simulation")
    pct = st.slider("Sales increase %", 0, 500, 20) / 100
    if 'stock_level' in df_filtered.columns and 'quantity' in df_filtered.columns:
        sim = df_filtered.copy()
        sim['projected_sales'] = (sim['quantity'] * (1 + pct)).astype(int)
        sim['remaining'] = sim['stock_level'] - sim['projected_sales']
        def sim_flag(x):
            if x < 0:
                return "❌ Out"
            if x <= 20:
                return "⚠️ Low"
            return "✅ Safe"
        sim['status'] = sim['remaining'].apply(sim_flag)
        cols_to_show = [ c for c in ['product', 'stock_level', 'quantity', 'projected_sales', 'remaining', 'status'] if c in sim.columns ]
        st.dataframe(sim[cols_to_show], width="stretch")
    else:
        st.info("Need 'stock_level' and 'quantity' columns for What-If simulation.")

# ---------------- ADMIN TAB ----------------
with tab_admin:
    st.header("Admin / Export & Debug")
    if st.button("Download cleaned CSV"):
        st.download_button("Download", df_filtered.to_csv(index=False), file_name="inventory_cleaned.csv", mime="text/csv")
    st.subheader("Debug: Raw Columns & Sample Data")
    st.write("Columns in raw MongoDB merged data:")
    st.code(list(df_raw.columns))
    st.write("First 5 rows (cleaned):")
    st.dataframe(df.head(), width="stretch")

# ---------------- FOOTER ----------------
st.markdown("<br><div style='text-align:center; color:#006565;'>Made with ♥ — Smart Inventory Analytics</div>", unsafe_allow_html=True)
