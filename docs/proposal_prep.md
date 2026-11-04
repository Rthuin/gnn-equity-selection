# Proposal Presentation Prep

**Created 2026-11-04.** Companion to `proposal_deck.pptx` (12 slides, speaker notes
included) and `literature_review.md` (v2, 23 references).

Assumed format: **~10–12 min talk + questions.** Adjust with the tachometer in §3.

---

## 1. Memorize these three things (not the whole deck)

**The pitch (say it in the first 20 seconds, slide 1):**

> "Stocks co-move through business relations — sectors, supply chains, common
> exposures. A graph neural network exploits that structure. But markets drift,
> so we test whether adapting the model — and the graph — to drift preserves
> the advantage."

**The two questions the project answers (slide 4):**

> **RQ1:** Does a relation-graph GNN rank stocks better than non-graph baselines
> (Lasso, XGBoost, MLP) *at identical features*?
> **RQ2:** Does drift-awareness — scheduled retraining, detector-triggered
> retraining, a time-varying graph — preserve that advantage as markets change?

**The drift ladder (slide 7 — the project's spine, expect a question on it):**

> "Four rungs, each strictly more adaptive: static model → walk-forward
> retraining → ADWIN-triggered retraining → dynamic graph. At every rung the
> GNN races the same non-graph baselines. Rung 1 answers RQ1; the jumps
> between rungs answer RQ2."

If you can say these three blocks from memory, you can survive any slide
failure and any interruption.

## 2. Numbers to have at your fingertips

| Fact | Number |
|---|---|
| Universe | S&P 100 (~98 downloadable via yfinance) |
| History | ~10 years daily, 2015–2025 (spans 2020 COVID, 2022 rate cycle) |
| PoC demo | 98 stocks → 294-edge correlation graph → GCN → ranking, ~5 s offline |
| Portfolio rule | Long top decile, equal-weight; benchmark = equal-weight whole universe |
| Literature | 23 references, all individually verified; 4-gap analysis |
| Environment | PyTorch 2.14 (CPU) + PyG 2.8.1 — verified by smoke test |
| Baselines | Lasso → XGBoost → MLP (Gu–Kelly–Xiu ladder), identical features |

## 3. Run of show (~11.5 min)

| # | Slide | Time | The ONE point |
|---|---|---|---|
| 1 | Title + pitch | 0:30 | The sentence from §1 |
| 2 | The idea | 1:00 | Relations carry information; classical ML throws them away |
| 3 | Task | 0:45 | Selection = ranking; why NOT price forecasting |
| 4 | Research questions | 1:00 | RQ1 / RQ2; negative RQ1 is still a finding |
| 5 | Data | 0:45 | yfinance constraint → S&P 100, cached once, reproducible |
| 6 | Graph method | 1:15 | Nodes/edges/two edge types; GCN vs GAT; ranking loss |
| 7 | **Drift ladder** | **1:30** | The spine — slow down here |
| 8 | Evaluation | 1:00 | Two levels (rank IC + portfolio), regime split, no lookahead |
| 9 | Literature | 0:45 | "All verified; here's the gap we fill" |
| 10 | Feasibility | 1:00 | Already runs end-to-end; demo on request |
| 11 | Timeline | 0:30 | Weeks are a template until deadline is fixed |
| 12 | Risks | 0:45 | Biggest risk (no graph gain) is handled by design |

Over time? Cut from slides 3, 5, 9 — never from 7.

## 4. Question bank (professor's likely questions → speakable answers)

Answers are written short and plain — practice saying them out loud, not reading them.

### Motivation

**Q: Why selection, not price forecasting?**
> "Price forecasting has to beat 'tomorrow = today' — a notoriously strong
> baseline. Selection only needs the *relative* ranking to be right. And
> ranking is exactly what a ranking loss trains. Feng et al. 2019 use the
> same framing."

**Q: Why should a graph help at all? What's the economic logic?**
> "Cohen and Frazzini, 2008, Journal of Finance: economically linked firms —
> customers and suppliers — predict each other's returns, because information
> diffuses slowly across links. The graph is a machine built to exploit
> exactly that documented effect."

**Q: What if the graph doesn't help?**
> "Then RQ1 has a negative answer — and we report *when* and *why* it failed.
> The controlled comparison is the thesis either way. Negative results here
> are still informative."

**Q: Is this data mining or just investing?**
> "It's a supervised learning problem: features predict a future ranking.
> Data pipeline, feature engineering, model comparison, drift adaptation,
> honest evaluation — the standard data mining workflow."

### Method

**Q: What is concept drift, exactly?**
> "The relation between features and the target changes over time — sudden,
> gradual, incremental, or recurring, per the Gama taxonomy. Example: momentum
> works in some regimes and dies in others. In our project the *relations
> between companies* can drift too — that's the dynamic-graph rung."

**Q: Why GCN and GAT?**
> "GCN is the simple baseline graph model — every neighbor counts equally.
> GAT adds attention — the model learns *which* neighbor matters. Comparing
> them is a built-in ablation. Both are standard PyTorch Geometric layers,
> which keeps it feasible on CPU."

**Q: How do you build the edges?**
> "Two relation types in one graph, following HATS: sector membership —
> static — plus rolling return correlation above a threshold. Later the
> correlation edges get recomputed over time — that's the dynamic-graph rung."

**Q: Why a ranking loss instead of mean squared error?**
> "Selection only needs the correct *order*. Per-stock regression wastes
> capacity predicting return magnitudes we don't need. The relational ranking
> paper — RSR — shows ranking losses fit selection better."

**Q: What is ADWIN and why that detector?**
> "An adaptive-window drift detector with theoretical guarantees — when the
> error distribution inside the window changes significantly, it signals
> drift and we retrain. It's in the `river` library. Neri 2021 shows generic
> detectors can underperform on financial series, so the detector itself is
> one of our experimental variables, not a fixed assumption."

### Evaluation

**Q: How do you avoid lookahead bias?**
> "Strictly walk-forward: features at time t predict only t to t+1; training
> data always ends before the test period; no shuffling. We follow
> López de Prado's backtesting hygiene."

**Q: What metrics, and why two levels?**
> "Rank IC — Spearman correlation between predicted scores and realized
> returns — measures the model without portfolio noise. Portfolio metrics —
> cumulative return, Sharpe, drawdown, turnover — measure what a practitioner
> cares about. We report both, split by calm vs volatile regimes, because
> that's where drift-awareness should show its value."

**Q: Why an equal-weight benchmark?**
> "It's the dumbest fair version of our own strategy. If we can't beat
> equal-weighting the whole universe, there's nothing to talk about."

**Q: With four rungs and several models, isn't overfitting guaranteed?**
> "Three guards: baselines come first, evaluation is strictly walk-forward,
> and every rung reports — we don't cherry-pick the best one afterward."

### Feasibility

**Q: Why S&P 100?**
> "Large caps are liquid with clean, long histories — and 100 stocks keeps
> CPU training cheap. The smoke test and proof-of-concept already run in
> seconds."

**Q: What have you actually done so far?**
> "Environment verified — PyTorch and PyTorch Geometric on CPU. An
> end-to-end proof-of-concept: 98 stocks, data to features to a 294-edge
> correlation graph to a GCN to a ranked list, in about five seconds. And a
> literature review with 23 verified references and a gap analysis."

**Q: What's the hardest part?**
> "Getting a stable signal at all — equity prediction has low
> signal-to-noise. That's why we measure at two levels and split by regime.
> The engineering is already de-risked by the proof-of-concept."

**Q: Can you show it running?**
> Run: `.venv\Scripts\activate` then `python scripts\poc_selection_demo.py`
> (~5 s, offline, uses the local cache).
> Talking point: *"The top scores went to the correlated energy majors —
> COP, Chevron, Exxon — with nearly identical scores. That's message passing
> averaging graph neighbors, visible even in an untrained model."*

## 5. Vocabulary you must be able to define in one sentence

| Term | One-liner |
|---|---|
| Message passing | Each node updates its representation using its neighbors' features |
| GCN | A layer that averages neighbor information with fixed normalization |
| GAT | Like GCN, but learned attention decides each neighbor's weight |
| Node / edge | A stock / a relation between two stocks (sector or correlation) |
| Top decile | The best-ranked 10% of stocks we buy |
| Equal-weight | Every held stock gets the same dollar amount |
| Walk-forward | Train on the past, test on the next period, roll forward, repeat |
| Rank IC | Spearman correlation of predicted ranks vs realized return ranks |
| Sharpe ratio | Return per unit of volatility |
| Max drawdown | Worst peak-to-trough loss |
| Turnover | How much the portfolio changes per rebalance (cost proxy) |
| Concept drift | The feature→target relation changes over time |
| ADWIN | Drift detector: grows/shrinks a window, signals when error shifts |
| Regime split | Report metrics separately for calm vs high-volatility periods |

## 6. Rehearsal plan

1. **Pass 1 — read aloud, sitting down.** Read the deck with its speaker notes
   top to bottom once. Goal: your mouth learns the words.
2. **Pass 2 — timed, standing up.** No reading; glance at §1's three blocks
   only. Record yourself on your phone. Check: ≤ 12 minutes, and slides 2, 7,
   8 make sense without you narrating every bullet.
3. **Pass 3 — mock Q&A.** Have someone fire questions from §4 in random order
   (Claude can play the professor — just ask). Practice the negative-result
   answer until it sounds confident, not defensive.
4. **Day before:** fill the deck placeholders (name/course/date, timeline
   weeks — ask Claude with the details), export the deck to PDF as a backup,
   run the demo once to confirm the cache is intact.
5. **Day of:** PDF on a USB stick + emailed to yourself; arrive early enough
   to test the display; keep §1's three blocks on a cue card.

## 7. Still open before the meeting

- [ ] Fill deck placeholders: name / course / date (title slide) and calendar
      weeks (timeline slide) — needs the course deadline
- [ ] Confirm meeting length and format (this doc assumes ~10–12 min +
      questions; rescale §3 if different)
- [ ] One full timed rehearsal done
- [ ] Demo cache verified (`data/sp100_close.csv` exists; run the demo once)
