# Proposal deck - mirror of proposal_deck.pptx

## Concept Drift-Aware Graph Neural Networks for Equity Selection
*Do company-relation graphs improve stock selection - and can drift-awareness keep that edge as markets change?*
[Course: Data Mining]  |  [Your Name]  |  [Date]
> Notes: One-sentence pitch: stocks co-move through business relations; a graph neural network exploits that structure, and because markets drift, we test whether adapting the model (and the graph) to drift preserves the gain.

## 2. The idea in one line
- "When a company becomes more valuable, related companies tend to become more valuable too."
- Stock selection is relational: sectors, supply chains, common exposures move together
- Classical ML treats each stock as an independent row - it throws this structure away
- GNNs learn over a company-relation graph: neighbors pass information to neighbors
- But markets are non-stationary: both returns AND the relations themselves drift
> Notes: This is the core hypothesis. The graph encodes who is related to whom; the GNN uses those edges to share information between stocks. Drift is the twist: yesterday's relations may not carry tomorrow's signal, which motivates the whole drift-aware design.

## 3. Task: cross-sectional stock selection
- Each rebalance date: score every stock -> rank -> long the top decile
- Benchmark: equal-weight portfolio of the whole universe
- Chosen from the course menu (trend / price / selection) - and why:
  - Ranking is the ML-friendly formulation (Feng et al. 2019: selection = ranking)
  - Price forecasting must beat 'tomorrow = today', a notoriously hard baseline
- Trend forecasting stays inside as the engine: features predict next-period performance
> Notes: Explain we deliberately did NOT pick price forecasting: the naive baseline is too strong. Selection only requires correct relative ordering across stocks, which is exactly what a ranking loss trains.

## 4. Research questions
- RQ1 - With vs without graph:
  - Does a relation-graph GNN rank stocks better than non-graph baselines (Lasso, XGBoost, MLP) at equal feature sets?
- RQ2 - Drift awareness:
  - Does adapting to concept drift (scheduled or detector-triggered retraining, time-varying graph) preserve the advantage as markets change?
- A negative RQ1 is still a reportable finding: 'the graph did not help - here is when and why'
> Notes: RQ1 is the thesis: the with/without-graph comparison under identical features and evaluation. RQ2 layers drift handling on top. Emphasize the design makes both questions answerable even if results are negative.

## 5. Data
- Source: Yahoo Finance via yfinance (course constraint) - daily OHLCV
- Universe: S&P 100 constituents (large caps: liquid, clean, long histories)
- History: ~10 years, 2015-2025 - spans calm and crisis regimes (2020 COVID, 2022 rate cycle)
- Node features per stock-day (factor-style, Fama-French / Gu-Kelly-Xiu tradition):
  - Momentum at several horizons, rolling volatility, volume z-score, sector
- Pulled once, cached locally - reproducible experiments, no live-API flakiness
> Notes: yfinance constraint is set by the professor. S&P 100 (not 500) keeps training cheap on CPU and data quality high. Cache-then-experiment avoids rate limits and makes results reproducible.

## 6. Method: the company-relation graph
- Nodes = stocks; node features = the factor-style features from the data slide
- Edges = multiple relation types in one graph (HATS design):
  - Sector membership (static) + rolling return correlation above threshold (LSTM-RGCN design)
- Models: GCN (uniform neighbor averaging) and GAT (attention - which neighbor matters?)
- Output: one score per stock -> ranking loss (listwise, RSR-style), not per-stock MSE
- Backtest: top-decile equal-weight portfolio vs equal-weight benchmark
> Notes: GCN vs GAT is itself a small ablation: does attention over relations beat uniform aggregation? Correlation edges make the graph partially learned from data, which sets up the dynamic-graph rung later.

## 7. Experimental design: the drift ladder
- Four rungs, each strictly more adaptive - every rung is a complete result on its own:
  - 1. Static: train once on early data, freeze
  - 2. Walk-forward: retrain every N weeks on a rolling window
  - 3. Drift-triggered: ADWIN monitors ranking error; retrain only when drift is signaled
  - 4. Dynamic graph: correlation edges recomputed over time (stretch: EvolveGCN-style)
- At EVERY rung: GNN vs non-graph baselines with identical features
- With/without graph is measured within rungs; drift benefit is measured across rungs
> Notes: This is the project's spine. Rung 1 vs baselines answers RQ1; rung-to-rung deltas answer RQ2. ADWIN via the river library. The ladder is prioritized: if time runs out, we still have complete results at whatever rung we reached.

## 8. Evaluation: how we'll know it works
- Portfolio (top decile vs equal-weight benchmark):
  - Cumulative return, Sharpe ratio, maximum drawdown, turnover
- Ranking quality (model-level, portfolio-free):
  - Rank information coefficient (Spearman: predicted score vs realized next-period return)
  - Top-decile hit rate
- Regime split: report metrics separately for calm vs high-volatility periods
- Hygiene: features at t predict t->t+1 only; strictly walk-forward, no lookahead
> Notes: Two levels: portfolio metrics (what a practitioner cares about) and rank metrics (what isolates the model from portfolio construction noise). The regime split is where drift-awareness should show its value if it has any.

## 9. Grounded in the literature (all references verified)
- Cohen & Frazzini 2008, Menzly & Ozbas 2010: linked firms predict each other's returns - the economic basis for our hypothesis
- Feng et al. 2019 (RSR): selection as ranking over stock relations - our task framing
- Kim et al. 2019 (HATS): multiple relation types with attention - our graph design
- Li et al. 2020 (LSTM-RGCN): correlation-based adjacency - our edge construction
- Pareja et al. 2020 (EvolveGCN), Wu et al. 2020 (MTGNN): dynamic graphs - the top rung
- Bifet & Gavalda 2007 (ADWIN), Gama et al. 2014 (drift taxonomy) - our drift machinery
- Gu, Kelly & Xiu 2020: ML baseline ladder - our baseline suite; Lopez de Prado 2018: backtest hygiene
- Full review: 23 references + synthesis table + gap analysis (docs/literature_review.md)
> Notes: Every reference was individually checked; two unverifiable suggestions were excluded during verification. v2 of the review adds the economic-foundations section, a comparison table of the core papers, and an explicit gap analysis - the four gaps map one-to-one onto our research questions and drift ladder.

## 10. Feasibility: already underway
- Environment verified (smoke test, 2026-10-30, all checks passed):
  - PyTorch 2.14 (CPU) + PyG 2.8.1; GCN/GAT convolutions run; yfinance download OK
- End-to-end proof-of-concept already runs (scripts/poc_selection_demo.py):
  - 98 stocks: data -> features -> correlation graph (294 edges) -> GCN -> ranked top-decile list
  - Untrained GCN - proves the machinery; learning good scores IS the project
- Literature review v2 complete: 23 references, synthesis table, gap analysis
- CPU is sufficient: S&P 100 x daily data is small; no exotic dependencies
> Notes: If asked to show it: run scripts/poc_selection_demo.py - about 5 seconds from the local cache, no network needed. Talking point: the demo's top scores went to the correlated energy majors (COP, CVX, XOM) with near-identical scores - that is message passing averaging graph neighbors, visible even in an untrained model. Full ranking saved in docs/poc_demo_output.md.

## 11. Timeline & deliverables (9-week template)
- W1-2: data download, caching, feature engineering
- W3-4: graph construction + non-graph baselines (Lasso, XGBoost, MLP)
- W5-6: GCN/GAT models + backtest harness
- W7-8: drift ladder experiments (rungs 1-4)
- W9+: final report and presentation
- Deliverables: code repository, backtest report, literature review (done), final paper + slides
> Notes: Weeks are a template until the course deadline is fixed - map to calendar dates once known. Note baselines come BEFORE the GNN: with/without comparison needs the without half built first.

## 12. Risks & mitigations
- Graph does not help -> RQ1 answered negatively; analyze when/why - still a valid contribution
- yfinance data quality -> pull once, validate ranges, universe restricted to liquid large caps
- Overfitting / lookahead bias -> strictly walk-forward evaluation, no time shuffling
- Scope creep -> drift ladder is prioritized; every rung stands alone as a result
- CPU-only training -> verified sufficient for S&P 100 x daily; universe size caps compute
> Notes: Biggest scientific risk (no graph gain) is handled by design: the comparison IS the thesis, either way. Biggest engineering risk (data flakiness) is handled by cache-once-then-experiment.
