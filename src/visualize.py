import os
import sys
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from environment import TransactionEnvironment
from network import QNetwork
from pipeline import prepare_data
from evaluate import compute_benchmarks

st.set_page_config(page_title="DQN Portfolio Agent", layout="wide")
st.title("DQN Portfolio Agent — Live Evaluation Demo")

DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_data.parquet')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
TC = 0.0005

CONFIGS = [
    ("no_factors", "base",          "Base (paper replication)"),
    ("no_factors", "sharpe",        "Sharpe reward"),
    ("no_factors", "sharpe+regime", "Sharpe + HMM regime"),
    ("factors",    "sharpe+regime", "Sharpe + regime + factors"),
]

REGIME_LABELS = {0: "Bear", 1: "Crisis", 2: "Bull"}
REGIME_COLORS = {0: "#E24B4A", 1: "#BA7517", 2: "#1D9E75"}


@st.cache_data(show_spinner="Loading data...")
def load_data(use_factors):
    df_train, df_val, df_test, fcols, regime_weights = prepare_data(
        parquet_path=DATA_PATH, use_factors=use_factors
    )
    return df_train, df_val, df_test, fcols, regime_weights


@st.cache_resource(show_spinner="Loading model weights...")
def load_ensemble(factor_label, reward_fn, state_dim):
    nets = []
    for hdim in [32, 64, 128]:
        path = os.path.join(RESULTS_DIR, f"qnet_{factor_label}_{reward_fn}_{hdim}.pt")
        if not os.path.exists(path):
            st.error(f"Weight file not found: {path}")
            st.stop()
        net = QNetwork(state_dim, hdim)
        net.load_state_dict(torch.load(path, map_location='cpu'))
        net.eval()
        nets.append(net)
    return nets


@st.cache_data(show_spinner="Running evaluation on test set...")
def run_evaluation(_q_nets, _df_test, feature_cols, _hash):
    dates = sorted(_df_test['date'].unique())
    agent_d, bh_d = [], []
    prev_inv = set()
    records = []

    for date in dates:
        dd = _df_test[_df_test['date'] == date]
        if len(dd) == 0:
            continue
        inv, all_r = [], []
        for _, row in dd.iterrows():
            feat = row[feature_cols].values.astype(float)
            pos = 1 if row['ticker'] in prev_inv else 0
            regime = row['regime']
            st_t = torch.FloatTensor(np.append(feat, [pos, regime])).unsqueeze(0)
            with torch.no_grad():
                votes = sum(n(st_t).argmax(1).item() for n in _q_nets)
            if votes > len(_q_nets) / 2:
                inv.append(row['ticker'])
            all_r.append(row['ret'])

        regime_today = int(dd['regime'].mode()[0])

        if len(inv) >= 3:
            r = dd[dd['ticker'].isin(inv)]['ret']
            new = set(inv) - prev_inv
            cost = len(new) / len(inv) * TC
            day_ret = r.mean() - cost
        else:
            day_ret = np.mean(all_r)
            inv = []

        agent_d.append(day_ret)
        bh_d.append(np.mean(all_r))
        prev_inv = set(inv)
        records.append({
            'date': date,
            'picks': inv,
            'n_picks': len(inv),
            'regime': regime_today,
            'agent_ret': day_ret,
            'bh_ret': np.mean(all_r),
        })

    mom_d, rev_d = compute_benchmarks(_df_test, tc=TC)
    return np.array(agent_d), np.array(bh_d), mom_d, rev_d, records



st.sidebar.header("Configuration")
config_idx = st.sidebar.selectbox(
    "Select model config",
    range(len(CONFIGS)),
    format_func=lambda i: CONFIGS[i][2]
)
factor_label, reward_fn, config_name = CONFIGS[config_idx]
use_factors = factor_label == "factors"

df_train, df_val, df_test, fcols, regime_weights = load_data(use_factors)
state_dim = len(fcols) + 2
q_nets    = load_ensemble(factor_label, reward_fn, state_dim)
agent_d, bh_d, mom_d, rev_d, records = run_evaluation(
    q_nets, df_test, fcols, str(df_test['date'].iloc[0])
)

a_cum  = np.cumprod(1 + agent_d) - 1
b_cum  = np.cumprod(1 + bh_d)   - 1
mo_cum = np.cumprod(1 + mom_d)  - 1
re_cum = np.cumprod(1 + rev_d)  - 1
dates  = [r['date'] for r in records]

st.sidebar.markdown("---")
st.sidebar.header("Step through time")
day_idx = st.sidebar.slider("Day", 0, len(records) - 1, len(records) - 1)
today = records[day_idx]


col1, col2, col3, col4 = st.columns(4)
col1.metric("DQN Agent CR", f"{a_cum[day_idx]:+.1%}")
col2.metric("Buy & Hold CR", f"{b_cum[day_idx]:+.1%}", f"{a_cum[day_idx]-b_cum[day_idx]:+.1%} excess")
col3.metric("Stocks picked today", f"{today['n_picks']} / {len(df_test[df_test['date']==today['date']])}")
regime_id = today['regime']
col4.metric("Market regime", REGIME_LABELS[regime_id])

st.markdown("---")


left, right = st.columns([2, 1])

with left:
    st.subheader("Cumulative return over time")
    chart_df = pd.DataFrame({
        'DQN Agent':  a_cum[:day_idx+1] * 100,
        'Buy & Hold': b_cum[:day_idx+1] * 100,
        'Momentum':   mo_cum[:day_idx+1] * 100,
        'Reversion':  re_cum[:day_idx+1] * 100,
    }, index=pd.to_datetime(dates[:day_idx+1]))
    st.line_chart(chart_df)

with right:
    st.subheader("Today's agent picks")
    st.caption(f"Date: {str(today['date'])[:10]}")

    regime_color = REGIME_COLORS[regime_id]
    regime_name  = REGIME_LABELS[regime_id]
    st.markdown(
        f"<div style='background:{regime_color}22;border-left:4px solid {regime_color};"
        f"padding:8px 12px;border-radius:4px;margin-bottom:12px'>"
        f"<b>Regime: {regime_name}</b><br>"
        f"<small>{'High volatility / crisis market' if regime_id==1 else 'Negative drift, moderate vol' if regime_id==0 else 'Positive drift, low vol'}</small>"
        f"</div>",
        unsafe_allow_html=True
    )

    if today['picks']:
        dd = df_test[df_test['date'] == today['date']]
        pick_df = dd[dd['ticker'].isin(today['picks'])][['ticker', 'ret']].copy()
        pick_df['ret'] = pick_df['ret'].map(lambda x: f"{x:+.2%}")
        pick_df.columns = ['Ticker', 'Return']
        st.dataframe(pick_df.reset_index(drop=True), use_container_width=True, height=300)
    else:
        st.info("No picks today — holding cash (cross-sectional mean)")

    st.markdown(f"**Day return:** {today['agent_ret']:+.3%}")
    st.markdown(f"**vs Buy & Hold:** {today['agent_ret']-today['bh_ret']:+.3%}")


st.markdown("---")
st.subheader("Portfolio activity")

b1, b2 = st.columns(2)

with b1:
    fig2, ax2 = plt.subplots(figsize=(6, 2.5))
    n_picks = [r['n_picks'] for r in records[:day_idx+1]]
    ax2.bar(range(len(n_picks)), n_picks, color='#534AB7', alpha=0.6, width=1.0)
    ax2.axhline(np.mean(n_picks), color='#D85A30', ls='--', linewidth=1, label=f'Avg: {np.mean(n_picks):.1f}')
    ax2.set_ylabel("Stocks picked")
    ax2.set_xlabel("Trading day")
    ax2.set_xticks([])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, axis='y')
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close()

with b2:
    fig3, ax3 = plt.subplots(figsize=(6, 2.5))
    regime_seq = [r['regime'] for r in records[:day_idx+1]]
    colors_seq = [REGIME_COLORS[r] for r in regime_seq]
    ax3.bar(range(len(regime_seq)), [1]*len(regime_seq), color=colors_seq, width=1.0)
    patches = [mpatches.Patch(color=REGIME_COLORS[k], label=REGIME_LABELS[k]) for k in REGIME_COLORS]
    ax3.legend(handles=patches, fontsize=8, loc='upper left')
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_title("Market regime over time")
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close()