# Portfolio Optimization with Deep Reinforcement Learning

---

## Part 1 — Crosby: Motivation & Problem Setup (~1 min)
*Slide: title slide*

Every trading day, for every stock in a portfolio, a fund manager faces the same question: **invest or hold cash?** We built a reinforcement learning agent that learns to answer that question from data.

Our baseline is a 2021 paper by Pigorsch and Schäfer. They showed a Deep Q-Network could trade a large cross-section of stocks by treating each asset as its own sequential decision problem. The reward signal is clever: holding cash isn't free — the agent receives the average return across all stocks in the universe when it sits out, essentially what a passive investor would earn — so missing a rally has real cost.

We trained and tested on US equity data from WRDS, a standard academic data source, covering over 3,000 stocks from 2010 to 2021. We reproduced the paper's approach on a 50-stock subset, then asked: what happens when you make the agent risk-aware, market-regime-aware, and factor-aware? Those three extensions are the contribution of this project.

---

## Part 2 — Aryan: Methods (~1 min)
*Slide: architecture diagram — Data → Environment → DQN → Portfolio*

The environment is a per-asset MDP. Each episode is one stock's price history. The agent sees a 33-dimensional state vector — a snapshot of everything relevant about a stock on a given day: 17 price-based signals like moving averages and volatility measures, 10 company fundamentals like profit margins and debt ratios, the stock's current price, whether the agent already holds it, and a market regime label. It outputs one of two actions: invest or cash.

We chose DQN specifically because financial markets are partially observable, which mirrors the setting DQN was originally designed for. The agent is a two-hidden-layer neural network. We train three networks and combine them with majority vote — a stock enters the portfolio only if at least two of three networks agree. We train for 500,000 steps per agent, saving the best model by validation performance.

*Slide: three extensions — Sharpe reward / HMM / factors*

The three extensions stack on top of each other. The **Sharpe reward** adds a rolling risk-adjustment term — it penalizes the agent for taking volatile positions, not just chasing raw returns. The **HMM** — implemented from scratch in PyTorch using the forward-backward algorithm and EM — learned three market regimes purely from price behavior, no hand-labeling required, with Viterbi decoding for inference. The **factor model** adds cross-sectional rank features inspired by academic finance research — specifically, where each stock ranks relative to its peers on size, valuation, and recent volatility. These are signals historically associated with return differences across stocks.

A few explicit assumptions: we trade 50 stocks rather than the full market, transaction costs are a fixed percentage with no slippage, and all models are trained on pre-2019 data and evaluated strictly out-of-sample on 2020–2021 — so the agent never sees the test period during training.

---

## Part 3 — Andres: Demo (~2 min)
*Screen: Streamlit app, base config loaded*

This is the agent running live on the test set — January 2020 through June 2021 — using the actual trained model weights.

*Switch to Sharpe config, drag slider to day 1*

Each day, the ensemble votes on every stock. The table shows which stocks were selected and their individual returns. The regime indicator shows the HMM's current market state.

*Drag slider slowly through March 2020*

Watch March 2020. The regime shifts to Crisis. The agent's selections change and the cumulative return takes a hit — but it recovers faster than buy-and-hold, and far faster than momentum which collapsed permanently.

*Switch back to base config at the same date*

Compare this to the base agent at the same moment — same crash, but the Sharpe-reward agent holds up better because it learned to avoid concentrated volatile positions. That difference in behavior during the crash is what drives the gap in final returns.

*Drag to end, Sharpe config*

By end of period the Sharpe agent is at +303% versus +114% for buy-and-hold.

---

## Part 4 — Manuel: Results, Takeaways, Conclusion (~2 min)
*Slide: metrics table*

Here are the numbers across all configurations on the test set.

| Strategy | Cum. Return | Sharpe | Max DD |
|---|---|---|---|
| Buy & Hold | +114% | 1.65 | -31.6% |
| Momentum | -21% | -0.21 | -62.5% |
| Reversion | +141% | 1.60 | -29.5% |
| DQN Base | +189% | 1.52 | -28.8% |
| DQN Sharpe | **+303%** | **2.27** | **-21.4%** |
| DQN Sharpe+Regime | +296% | 1.53 | -37.1% |
| DQN Factors | +225% | 1.87 | -21.4% |

The Sharpe reward was the single biggest improvement — 189% to 303%, Sharpe ratio up from 1.52 to 2.27, max drawdown reduced from -28% to -21%. The agent learned not just to chase returns, but to avoid volatile positions.

*Slide: extension findings*

The regime and factor extensions tell an interesting story. They did not improve cumulative return further, but they changed how the agent behaves. The regime agent selected nearly twice as many stocks per day on average — 16.8 versus 7.7 — meaning it built more diversified portfolios. The factor agent achieved a Sharpe ratio of 1.87, between base and Sharpe-only, suggesting the additional cross-sectional information helped risk management even when it didn't boost raw return. These extensions are doing something real — they just optimize for different things than peak return.

*Slide: takeaway*

**The takeaway: a DQN with a Sharpe-aware reward beat every benchmark through the most volatile market in a decade, on both return and risk. The extensions revealed that smarter information doesn't always mean higher returns — sometimes it means better-constructed portfolios. That distinction matters in practice.**