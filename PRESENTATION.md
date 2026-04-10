# Presentation Script: Portfolio Optimization with Deep Reinforcement Learning

Video presentation, **6 minutes max**. Each section is what the presenter says, with visuals noted in *italics*. No low-level code, hyperparameters, or implementation details.

**Presenters:**
1. Aryan Kalaskar: Motivation and problem setup
2. Manuel Torres: Key methods summary
3. Crosby Sayan: Demo
4. Andres Cruz: Results, takeaways, and conclusion

---

## Part 1, Aryan Kalaskar: Motivation and Problem Setup *(~1 min)*

*Slide: title slide with the four authors and project title.*

Hi everyone, I'm Aryan, and together with Manuel, Crosby, and Andres, we built a **Portfolio Optimization system using Deep Reinforcement Learning**.

*Slide: problem statement, "Every day, for every stock: hold cash or invest?"*

The question we set out to answer is simple: **can a reinforcement learning agent learn to trade a basket of stocks better than standard benchmarks**, like buying and holding the market, or following simple momentum and mean-reversion rules?

The setup works like this. Every trading day, our agent looks at each stock in the universe, its moving averages, volatility, fundamentals, and makes a binary decision: **invest or hold cash**. We equal-weight whatever it selects, charge transaction costs, and track performance over time.

Our starting point is a 2021 paper by Pigorsch and Schäfer on Deep Q-Networks for stock trading. We reproduced their approach, then extended it in three ways: a **Sharpe-ratio-aware reward** to penalize volatility, a **Hidden Markov Model** for market-regime awareness, and **Fama-French-style factor features** for cross-sectional information.

I'll hand it to Manuel to walk through the methods.

---

## Part 2, Manuel Torres: Key Methods Summary *(~1 min)*

*Slide: high-level architecture diagram showing Data, Environment, DQN Agent, Portfolio.*

Thanks, Aryan. I'm Manuel. Let me give you the high-level picture of how this works.

**The environment** follows a per-asset MDP formulation from the paper. Each stock is its own episode. The agent sees a feature vector, technical indicators, fundamentals, and optionally factor ranks and a market regime label, and picks one of two actions: invest or cash. The key design choice is that holding cash isn't free: the agent receives the market average return as a cash reward, so sitting out a rally has real opportunity cost.

*Slide: diagram showing DQN with ensemble voting, three networks, majority vote.*

**The agent** is a Deep Q-Network, a neural network that estimates the value of each action. We train three networks at different capacities, then combine them with **majority voting**: a stock only enters the portfolio if at least two of three networks agree. This ensemble approach adds robustness.

*Slide: three extensions listed: Sharpe reward, HMM regimes, factor features.*

Beyond the base paper, our **Sharpe reward** variant encourages risk-adjusted returns, the **HMM** identifies bull, bear, and high-volatility market states to condition the agent's behavior, and the **factor features** give the agent cross-sectional information about size, value, momentum, and quality.

Now Crosby will show you what this looks like in practice.

---

## Part 3, Crosby Sayan: Demo *(~2 min)*

*Screen: show the system running, walking through inputs and outputs, not code.*

Thanks, Manuel. I'm Crosby, and I'm going to show you the end product.

*Screen: show the data pipeline output, the 50-stock universe with feature summary.*

Here's our stock universe, 50 stocks selected for cap diversity: large-caps, small-caps, and mid-caps. For each stock on each day, the system computes a feature vector from raw market data, things like moving averages, volatility measures, and fundamentals.

*Screen: show HMM regime visualization over the full time period.*

This is our Hidden Markov Model's view of the market. You can see it identifying three distinct regimes: a calm state, a high-volatility state, and a bear state. Notice how cleanly it picks up the COVID crash in early 2020 as a regime shift.

*Screen: show the agent's daily portfolio selection, which stocks are in, which are out.*

Here's the agent in action on the test set. Each day, the three ensemble networks vote on every stock. Green means at least two networks agreed to invest, gray means cash. You can see the agent dynamically rotating in and out of positions based on market conditions.

*Screen: show equity curves, the agent's portfolio value over time vs benchmarks.*

And here's the bottom line: the portfolio equity curve. The blue line is our best agent, the Sharpe-reward variant. You can see it tracking the benchmarks early on, then pulling decisively ahead during the COVID recovery. The key thing to notice is not just the higher return, but the **shallower drawdown** during the crash, the agent learned to reduce exposure when conditions deteriorated.

Over to Andres for the full results.

---

## Part 4, Andres Cruz: Results, Takeaways, and Conclusion *(~2 min)*

*Slide: results summary table, cumulative return, Sharpe, max drawdown, win rate.*

Thanks, Crosby. I'm Andres. Let me put numbers to what you just saw.

Our test window starts January 2020, right into the COVID crash and recovery. **Buy and hold** returned about +114% with a max drawdown of -32%. **Momentum lost money** at -21%. **Mean reversion** did well at +141%.

Our best agent, the **Sharpe-reward DQN**, returned **+303%**, with a Sharpe ratio of **2.27** and a max drawdown of only **-21%**. That's roughly 2.7 times buy-and-hold's return with a shallower drawdown.

*Slide: comparison chart, bar chart of cumulative returns across all strategies.*

So the DQN clearly outperformed the benchmarks on this test window, and the Sharpe-aware reward was the key differentiator.

*Slide: honest findings, extensions underperformed.*

Now, an honest finding: we expected the HMM regime and factor extensions to improve things further, but they actually **underperformed the pure Sharpe agent** on this window. We believe this is because the COVID regime shift was too abrupt for the HMM to label cleanly out-of-sample, and adding more input features without proportionally more training data hurt sample efficiency.

*Slide: key assumptions and limitations.*

To be transparent about limitations: we tested on a single crisis window, used a simplified transaction cost model without slippage, and trained on a 50-stock subset rather than the full market. These are deliberate proof-of-concept choices, not claims of production readiness.

*Slide: clear takeaway, centered and bold.*

**Our takeaway: a straightforward DQN with a Sharpe-aware reward beat all benchmarks through the most volatile market in a decade, on a risk-adjusted basis. The extensions we hoped would help revealed that more information doesn't automatically mean better decisions, and understanding that tradeoff is the next step.**

Thank you, we're happy to take questions.

---

## Timing Notes

- **Total target: ≤ 6 minutes.** Aim for 5:30 to leave margin.
- If running long, Crosby can trim the HMM regime visual walkthrough.
- Visuals should do the heavy lifting: charts, equity curves, and tables, not text-heavy slides.
- No code on screen, only the end-product and results.
