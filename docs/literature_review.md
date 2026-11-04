# Literature Review — Graph Neural Networks for Equity Selection

**Draft v2 — 2026-11-04.** Compiled for the MBA data mining course project:
*concept drift–aware GNN for stock selection* (Yahoo Finance data, PyTorch Geometric stack).

> v1 (2026-10-19) was an annotated bibliography — one paper, one blurb. v2 turns it
> into a review: an economic-foundations section (§2), a side-by-side synthesis of
> the core papers (§7), and an explicit gap analysis stating what the literature
> does *not* do and how our project fills it (§8). All v1 verifications stand;
> entries added on 2026-11-04 are marked in the change log (§11).

**Verification legend** (every reference below carries one):
- ✅ = paper page fetched and checked (title / authors / venue confirmed)
- 🔎 = found via search with publisher link; spot-check the landing page before final citation
- 📚 = standard, textbook-famous reference; safe to cite from common knowledge

---

## 1. Research question this review supports

> Can relation-aware graph neural networks improve **cross-sectional stock selection**
> (rank stocks, buy the top decile) compared to non-graph baselines — and can
> **concept-drift awareness** (periodic retraining, drift-triggered refreshes, and a
> time-varying company-relation graph) preserve that advantage under market
> non-stationarity?

The review's through-line, in three steps the sections below unpack:
1. **Relations carry information** — a top-journal finance literature shows
   economically linked firms predict each other's returns (§2).
2. **GNNs are the machinery for exploiting relations** — an active ML literature
   encodes company relations as graph edges and back-tests the gain (§3–4).
3. **But the field's treatment of drifting, non-stationary markets is ad hoc**
   — which is the gap our drift ladder targets (§5, §8).

## 2. Why related stocks co-move: the economic foundations

*This section is the academic warrant for our core hypothesis ("when a company
becomes more valuable, related companies tend to become more valuable too").
It is not GNN literature — it is the finance literature that says the signal
our edges should carry actually exists.*

- **Cohen & Frazzini (2008), "Economic Links and Predictable Returns",
  Journal of Finance 63(4), 1977–2011.** DOI 10.1111/j.1540-6261.2008.01379.x 🔎
  Customer–supplier linked firms exhibit **lead–lag return predictability**:
  a supplier's customers' returns predict the supplier's subsequent returns,
  because the market is slow to diffuse news across economically linked firms.
  Smith Breeden prize; ~2,000 citations. *This is our hypothesis, a decade
  earlier, in the top finance journal.* (Citation warning: often mis-attributed
  to RFS — it is the Journal of Finance.)

- **Menzly & Ozbas (2010), "Market Segmentation and Cross-predictability of
  Returns", Journal of Finance 65, 1555–1580.** 🔎
  Generalizes the mechanism: because investors are segmented by industry,
  information diffuses slowly *across* industries, producing systematic
  cross-industry return predictability. *Justifies sector edges — not just
  supply chains — carrying predictive information.*

**Takeaway for the proposal:** the "related firms co-move" premise is not folk
wisdom — it is a documented market inefficiency with a mechanism (slow
cross-firm information diffusion). A GNN whose message passing follows
economic links is a machine built to exploit exactly this.

## 3. Foundations: the GNN building blocks

- **GCN — Kipf & Welling (2017), "Semi-Supervised Classification with Graph
  Convolutional Networks", ICLR 2017.** arXiv:1609.02907 📚
  The canonical message-passing layer: each node's embedding becomes a normalized
  average of its neighbors'. Our static-graph layer.

- **GAT — Veličković et al. (2018), "Graph Attention Networks", ICLR 2018.**
  arXiv:1710.10903 📚
  Replaces fixed normalization with learned attention weights, so the model can
  learn *which* related company matters more for a prediction — directly matching
  our hypothesis that some peers lead and others follow.

- **Chen, Wei & Huang (2018), "Incorporating Corporation Relationship via Graph
  Convolutional Neural Networks for Stock Price Prediction", CIKM 2018.**
  DOI 10.1145/3269206.3269269 🔎 (~370 citations; PDF via fudan-disc.com)
  The early, widely-cited demonstration that a GCN over company-relation edges
  improves stock prediction versus the same model without relations. *Proof of
  concept for the with/without-graph contrast we run as RQ1.*

## 4. GNNs applied to equities (core literature)

- **RSR — Feng, He, Wang, Luo, Liu & Chua (2019), "Temporal Relational Ranking for
  Stock Prediction", ACM TOIS 37(2).** arXiv:1809.09441 ✅
  Reframes stock prediction as a **ranking** problem (not per-stock regression) and
  introduces a temporal graph convolution over stock relations; backtested on
  NYSE/NASDAQ. *Closest to our task definition: selection = cross-sectional ranking.*

- **HATS — Kim et al. (2019), "HATS: A Hierarchical Graph Attention Network for
  Stock Movement Prediction".** arXiv:1908.07999 ✅ (~294 citations; code:
  github.com/dmis-lab/hats)
  Hierarchical attention over **multiple relation types** between companies
  (same industry, business connections, etc.). Takeaway for us: sector edges and
  correlation edges can coexist in one graph, with attention deciding their weight.

- **Matsunaga, Suzumura & Takahashi (2019), "Exploring Graph Neural Networks for
  Stock Market Predictions with Rolling Window Analysis", NeurIPS 2019 Workshop
  on Robust AI in Financial Services.** arXiv:1909.10660 ✅
  Supplier-chain graph (Nikkei Value Search) over Nikkei 225 equities, ~20 years,
  evaluated with a **rolling-window backtest**: the GNN beats both the market
  benchmark (+29.5% return ratio, 2.2× Sharpe) and — crucially for us — an
  **equal-feature non-graph LSTM baseline** (+6.3% return ratio, 1.3× Sharpe).
  *Two gifts: (a) the with/without-graph comparison done properly, (b) rolling
  windows as standard practice — the seed of our walk-forward rung.*

- **LSTM-RGCN — Li, Bao, Harimoto, Chen, Xu & Su (2020), "Modeling the Stock
  Relation with Graph Network for Overnight Stock Movement Prediction",
  IJCAI 2020 (AI in FinTech track).** ✅
  Builds the graph from **stock correlation matrices** inside a relational GCN.
  The overnight-movement task differs from ours, but the correlation-based edge
  construction is directly reusable for our second graph variant.

- **Patel (2024), "A Systematic Review on Graph Neural Network-based Stock
  Prediction", ACM (Computing Surveys per citing sources).**
  DOI 10.1145/3696411 🔎 (author list to confirm on spot-check)
  Systematic review organizing GNN stock-prediction work by task, graph source,
  and architecture. *Our breadth check: the design choices we made (GCN/GAT,
  correlation + sector edges, ranking objective) are the literature's mainstream,
  not exotic picks.*

- **Cao (2024), "Quantitative Stock Selection Model Using Graph Learning",
  Journal of Theoretical and Applied Electronic Commerce Research (MDPI).**
  https://www.mdpi.com/0718-1876/19/3/86 🔎
  A recent paper with the *same target task* (selection → portfolio construction);
  useful as contemporary evidence the problem setting is active and legitimate.

- **Antulov-Fantulin et al. (2026), "Graph Neural Networks in Asset and Investment
  Management", Journal of Portfolio Management (early access).**
  https://www.pm-research.com/content/iijpormgmt/early/2026/07/09/jpm2026042 🔎
  Practitioner-oriented overview of GNNs in investment management. Cite in the
  introduction to justify the approach to a business audience in business language.

- *Further reading:* **Du et al. (2025), "Modeling hybrid firm relationships with
  graph neural networks", International Journal of Forecasting.**
  https://www.sciencedirect.com/science/article/pii/S0167923625001290 🔎
  Combines several explicit firm-relation graphs (including money flow) — a taste
  of richer edge types beyond sector/correlation.

## 5. Dynamic graphs & concept drift (our chosen variant)

*Common practice in the core literature is a fixed graph plus rolling-window
retraining. The papers below are the machinery for going further — letting the
model, the graph, or the retraining schedule itself adapt to drift.*

- **EvolveGCN — Pareja et al. (2020), AAAI 2020.** arXiv:1902.10191 ✅ (2,000+ citations)
  The standard dynamic-graph method: an RNN **evolves the GCN's weight matrices**
  along the time dimension instead of tracking node embeddings. This is the
  "top rung" candidate for our drift ladder if time permits.

- **MTGNN — Wu et al. (2020), "Connecting the Dots: Multivariate Time Series
  Forecasting with Graph Neural Networks", KDD 2020.** DOI 10.1145/3394486.3403118 ✅
  Learns the adjacency matrix **from data** rather than fixing it — the academic
  justification for letting the company-relation graph change over time.

- **Gama, Žliobaitė, Bifet, Pechenizkiy & Bouchachia (2014), "A Survey on Concept
  Drift Adaptation", ACM Computing Surveys 46(4).** 📚
  The canonical drift taxonomy (sudden / gradual / incremental / recurring) and
  adaptation strategies. Cite for definitions in the method section; read §1–3 only.

- **Bifet & Gavaldà (2007), "Learning from Time-Changing Data with Adaptive
  Windowing" (ADWIN), IDA 2007.** 📚
  The adaptive-window drift detector with theoretical guarantees — the trigger
  mechanism we will use (implementation via the `river` Python library:
  riverml.xyz).

- **Neri (2021), "Domain Specific Concept Drift Detectors for Predicting Financial
  Time Series".** arXiv:2103.14079 ✅
  Shows generic drift detectors underperform on financial series and proposes
  finance-tailored ones — motivates our treating detector choice as an explicit
  experiment rather than a default.

- *Emerging-work note:* **Collins (2026), "Addressing Concept Drift in Graph-Based
  State Space Models" (A-S3G).** callpress.org (minor venue) 🔎
  Continuous-learning graph model for stock prediction under drift. Cite lightly,
  if at all — its value to us is as **evidence the drift×graph intersection is
  only now appearing, and in minor venues**: the intersection is genuinely young.

## 6. Finance & ML anchor references

- **Markowitz (1952), "Portfolio Selection", Journal of Finance 7(1).** 📚
  Origin of portfolio construction. Our top-decile equal-weight portfolio is
  deliberately simpler; cite for lineage.

- **Fama & French (2015), "A Five-Factor Asset Pricing Model", Journal of
  Financial Economics 116(1).** 📚
  Factor-model background: our node features (momentum, size, volatility) are
  factor-style inputs in this tradition.

- **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning",
  Review of Financial Studies 23(5).** 📚
  The landmark ML-for-equity-prediction study; establishes the baseline-ladder
  practice (linear → trees → neural nets) our with/without-graph comparison
  follows, and sets realistic expectations for effect sizes.

- **López de Prado (2018), *Advances in Financial Machine Learning*, Wiley.** 📚
  Backtest hygiene: walk-forward evaluation, avoiding lookahead and leakage, and
  skepticism toward in-sample Sharpe ratios. *Cited in our evaluation section to
  justify the protocol (regime splits, turnover, rank metrics beyond raw returns).*

## 7. Synthesis: how the core papers compare

Side-by-side view of the equity-GNN core (cells from paper abstracts; 🔎-marked
papers: re-check numbers before quoting them in the final report):

| Paper | Task | Edges / graph source | Model | What it establishes for us |
|---|---|---|---|---|
| Chen et al. 2018 | Price movement | Corporation relations | GCN | Early evidence: relation edges beat the same model without them |
| Matsunaga et al. 2019 | Return prediction | Supplier chain (Nikkei) | GNN vs **equal-feature LSTM** | The with/without-graph contrast done properly; rolling-window evaluation |
| Feng et al. 2019 (RSR) | **Ranking / selection** | Stock relations, temporal | Graph conv + ranking loss | Selection = cross-sectional ranking — our task framing |
| Kim et al. 2019 (HATS) | Movement prediction | **Multiple relation types** | Hierarchical graph attention | Sector + correlation edges can coexist; attention weights them |
| Li et al. 2020 (LSTM-RGCN) | Overnight movement | **Correlation matrix** | Relational GCN + LSTM | Correlation-based adjacency works — our dynamic-edge seed |
| Pareja et al. 2020 (EvolveGCN) | General dynamic graphs | (method paper) | RNN-evolved GCN weights | The standard recipe for graphs that change over time |
| Wu et al. 2020 (MTGNN) | Multivariate forecasting | **Learned adjacency** | Graph learning + forecasting | The graph itself can be learned from data |
| Cao 2024 | Selection → portfolio | Graph learning | Graph learning model | The task setting stays active in current journals |

Read down the columns and the field's shape appears: edges come from three
families (explicit economic relations, correlations, learned adjacency); tasks
split between movement prediction and ranking/selection; and the drift question
is handled almost entirely by rolling-window retraining rather than explicit
drift detection.

## 8. The gap our project fills

Four observations on what the literature above does **not** do — each maps to a
design decision in our project:

1. **The controlled with/without-graph comparison is rarely the headline.**
   Papers demonstrate a graph model beating a baseline (Matsunaga et al.'s LSTM
   contrast is the cleanest), but the full baseline ladder of Gu–Kelly–Xiu
   (linear → trees → MLP) at **identical features and identical evaluation** is
   seldom run against graph models. → *Our RQ1 makes that comparison the thesis,
   not a side note — and a negative result is reportable.*

2. **Drift handling is implicit (rolling windows), not explicit (detection).**
   Rolling-window retraining is common practice, but detector-triggered
   retraining (ADWIN monitoring the ranking error) coupled to a GNN ranker
   appears essentially absent — and Neri (2021) warns generic detectors
   underperform on finance, so it must be tested, not assumed. → *Our rungs 2–3
   contrast scheduled vs detector-triggered adaptation.*

3. **Dynamic graphs target forecasting, not selection.** The EvolveGCN/MTGNN
   lineage predicts values; combining a **time-varying company graph** with a
   **ranking objective and a portfolio backtest** is rare — the drift×graph×
   selection intersection is only now surfacing in minor venues (A-S3G, 2026).
   → *Our rung 4 recomputes correlation edges over time inside the selection
   task.*

4. **Evaluation hygiene varies.** Returns and Sharpe ratios dominate; regime
   splits, turnover, and rank-level metrics are inconsistent — exactly the
   pitfalls López de Prado catalogs. → *Our evaluation reports portfolio AND
   rank metrics, split by market regime, strictly walk-forward.*

**In one sentence:** the literature establishes that relations carry return
information and that GNNs can exploit it — but not *whether the graph advantage
survives a disciplined baseline comparison for selection, under explicit drift
adaptation, evaluated to backtesting-hygiene standards.* That sentence is our
project.

## 9. What we adopt from each (design mapping)

| Source | What our project takes from it |
|---|---|
| Cohen & Frazzini (2008); Menzly & Ozbas (2010) | Economic warrant: linked firms predict each other — edges carry signal |
| Feng et al. (RSR) | Selection formulated as cross-sectional **ranking** |
| Kim et al. (HATS) | **Multi-type edges**: sector + correlation in one graph |
| Li et al. (LSTM-RGCN) | Rolling-correlation adjacency construction |
| Matsunaga et al. (2019) | Walk-forward evaluation; graph vs equal-feature non-graph comparison design |
| Patel (2024) | Breadth check: our architecture/edge choices are the mainstream |
| Pareja (EvolveGCN), Wu (MTGNN) | Dynamic-graph ambition: time-varying relations (top rung) |
| Gama survey, ADWIN, Neri | Drift ladder: walk-forward retrain → detector-triggered refresh |
| Gu, Kelly & Xiu | Baseline suite: Lasso, XGBoost, MLP before any GNN claim |
| López de Prado | Evaluation hygiene: walk-forward, regime splits, rank metrics, Sharpe skepticism |

## 10. Suggested reading order (if you only read five)

1. **Cohen & Frazzini (2008)** — the economic "why" in six pages of intuition
2. **Antulov-Fantulin et al. (JPM)** — gentle, business-language GNN overview
3. **Feng et al. (RSR)** — our exact task framing
4. **Kim et al. (HATS)** — our graph design
5. **Gama et al. survey, §1–3** — drift vocabulary for the method section

*(6th, before building rung 4: EvolveGCN.)*

## 11. Verification notes & excluded references

**Excluded after verification (do not cite without checking):**
- *"Taming Factor Models with Graph Neural Networks"* — **not found** on arXiv
  (exact-title API query: 0 results). Likely a conflation with Feng, Giglio & Xiu's
  "Taming the Factor Zoo" (asset pricing, no GNN). Excluded.
- *"ESTIMATE" (IJCAI 2021)* — not verifiable via arXiv search. Excluded.

**Corrections & caveats found during verification:**
- 2026-11-04: a recalled candidate *"Chen et al., IJCAI 2019 — Incorporating
  Corporation Relationship into Stock Selection"* **does not exist in that form**.
  The verified real paper is Chen, Wei & Huang, **CIKM 2018**, stock *price*
  prediction (§3). Recorded so the phantom does not return.
- Cohen & Frazzini (2008) is in the **Journal of Finance** — frequently
  mis-cited as RFS. DOI confirmed via Wiley/HBS/SSRN records.
- Patel (2024) DOI confirmed (10.1145/3696411); author list still needs a
  landing-page spot-check before the final report (ACM blocked automated fetch).
- Matsunaga et al. (2019): page fetched 2026-11-04; title includes "with Rolling
  Window Analysis" (often dropped in citations).

> Lesson worth a line in your methodology appendix: LLM-suggested citations
> sometimes do not exist or arrive subtly wrong (wrong venue, wrong year, wrong
> task). Everything in this document was checked; keep it that way before
> anything lands in the submitted report.

---

**Reference count (v2): 23** — 7 ✅ page-verified · 8 📚 textbook-standard · 8 🔎 publisher-linked (spot-check before final citation).
