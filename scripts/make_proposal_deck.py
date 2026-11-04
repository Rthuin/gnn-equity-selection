"""Build the project-proposal deck: docs/proposal_deck.pptx + docs/proposal_deck.md.

Single source of truth: the SLIDES list below. Edit content here, rerun:
    .venv/Scripts/python.exe scripts/make_proposal_deck.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT_PPTX = ROOT / "docs" / "proposal_deck.pptx"
OUT_MD = ROOT / "docs" / "proposal_deck.md"

# Each slide: {"title": str, "bullets": [(level, text), ...], "notes": str}
SLIDES = [
    {
        "title": "Concept Drift-Aware Graph Neural Networks for Equity Selection",
        "subtitle": "Do company-relation graphs improve stock selection - and can "
        "drift-awareness keep that edge as markets change?",
        "meta": "[Course: Data Mining]  |  [Your Name]  |  [Date]",
        "notes": "One-sentence pitch: stocks co-move through business relations; "
        "a graph neural network exploits that structure, and because markets drift, "
        "we test whether adapting the model (and the graph) to drift preserves the gain.",
    },
    {
        "title": "The idea in one line",
        "bullets": [
            (0, '"When a company becomes more valuable, related companies tend to become more valuable too."'),
            (0, "Stock selection is relational: sectors, supply chains, common exposures move together"),
            (0, "Classical ML treats each stock as an independent row - it throws this structure away"),
            (0, "GNNs learn over a company-relation graph: neighbors pass information to neighbors"),
            (0, "But markets are non-stationary: both returns AND the relations themselves drift"),
        ],
        "notes": "This is the core hypothesis. The graph encodes who is related to whom; "
        "the GNN uses those edges to share information between stocks. Drift is the twist: "
        "yesterday's relations may not carry tomorrow's signal, which motivates the whole "
        "drift-aware design.",
    },
    {
        "title": "Task: cross-sectional stock selection",
        "bullets": [
            (0, "Each rebalance date: score every stock -> rank -> long the top decile"),
            (0, "Benchmark: equal-weight portfolio of the whole universe"),
            (0, "Chosen from the course menu (trend / price / selection) - and why:"),
            (1, "Ranking is the ML-friendly formulation (Feng et al. 2019: selection = ranking)"),
            (1, "Price forecasting must beat 'tomorrow = today', a notoriously hard baseline"),
            (0, "Trend forecasting stays inside as the engine: features predict next-period performance"),
        ],
        "notes": "Explain we deliberately did NOT pick price forecasting: the naive baseline is too "
        "strong. Selection only requires correct relative ordering across stocks, which is exactly "
        "what a ranking loss trains.",
    },
    {
        "title": "Research questions",
        "bullets": [
            (0, "RQ1 - With vs without graph:"),
            (1, "Does a relation-graph GNN rank stocks better than non-graph baselines "
                "(Lasso, XGBoost, MLP) at equal feature sets?"),
            (0, "RQ2 - Drift awareness:"),
            (1, "Does adapting to concept drift (scheduled or detector-triggered retraining, "
                "time-varying graph) preserve the advantage as markets change?"),
            (0, "A negative RQ1 is still a reportable finding: 'the graph did not help - here is when and why'"),
        ],
        "notes": "RQ1 is the thesis: the with/without-graph comparison under identical features "
        "and evaluation. RQ2 layers drift handling on top. Emphasize the design makes both "
        "questions answerable even if results are negative.",
    },
    {
        "title": "Data",
        "bullets": [
            (0, "Source: Yahoo Finance via yfinance (course constraint) - daily OHLCV"),
            (0, "Universe: S&P 100 constituents (large caps: liquid, clean, long histories)"),
            (0, "History: ~10 years, 2015-2025 - spans calm and crisis regimes (2020 COVID, 2022 rate cycle)"),
            (0, "Node features per stock-day (factor-style, Fama-French / Gu-Kelly-Xiu tradition):"),
            (1, "Momentum at several horizons, rolling volatility, volume z-score, sector"),
            (0, "Pulled once, cached locally - reproducible experiments, no live-API flakiness"),
        ],
        "notes": "yfinance constraint is set by the professor. S&P 100 (not 500) keeps training "
        "cheap on CPU and data quality high. Cache-then-experiment avoids rate limits and makes "
        "results reproducible.",
    },
    {
        "title": "Method: the company-relation graph",
        "bullets": [
            (0, "Nodes = stocks; node features = the factor-style features from the data slide"),
            (0, "Edges = multiple relation types in one graph (HATS design):"),
            (1, "Sector membership (static) + rolling return correlation above threshold (LSTM-RGCN design)"),
            (0, "Models: GCN (uniform neighbor averaging) and GAT (attention - which neighbor matters?)"),
            (0, "Output: one score per stock -> ranking loss (listwise, RSR-style), not per-stock MSE"),
            (0, "Backtest: top-decile equal-weight portfolio vs equal-weight benchmark"),
        ],
        "notes": "GCN vs GAT is itself a small ablation: does attention over relations beat uniform "
        "aggregation? Correlation edges make the graph partially learned from data, which sets up "
        "the dynamic-graph rung later.",
    },
    {
        "title": "Experimental design: the drift ladder",
        "bullets": [
            (0, "Four rungs, each strictly more adaptive - every rung is a complete result on its own:"),
            (1, "1. Static: train once on early data, freeze"),
            (1, "2. Walk-forward: retrain every N weeks on a rolling window"),
            (1, "3. Drift-triggered: ADWIN monitors ranking error; retrain only when drift is signaled"),
            (1, "4. Dynamic graph: correlation edges recomputed over time (stretch: EvolveGCN-style)"),
            (0, "At EVERY rung: GNN vs non-graph baselines with identical features"),
            (0, "With/without graph is measured within rungs; drift benefit is measured across rungs"),
        ],
        "notes": "This is the project's spine. Rung 1 vs baselines answers RQ1; rung-to-rung "
        "deltas answer RQ2. ADWIN via the river library. The ladder is prioritized: if time runs "
        "out, we still have complete results at whatever rung we reached.",
    },
    {
        "title": "Evaluation: how we'll know it works",
        "bullets": [
            (0, "Portfolio (top decile vs equal-weight benchmark):"),
            (1, "Cumulative return, Sharpe ratio, maximum drawdown, turnover"),
            (0, "Ranking quality (model-level, portfolio-free):"),
            (1, "Rank information coefficient (Spearman: predicted score vs realized next-period return)"),
            (1, "Top-decile hit rate"),
            (0, "Regime split: report metrics separately for calm vs high-volatility periods"),
            (0, "Hygiene: features at t predict t->t+1 only; strictly walk-forward, no lookahead"),
        ],
        "notes": "Two levels: portfolio metrics (what a practitioner cares about) and rank metrics "
        "(what isolates the model from portfolio construction noise). The regime split is where "
        "drift-awareness should show its value if it has any.",
    },
    {
        "title": "Grounded in the literature (all references verified)",
        "bullets": [
            (0, "Cohen & Frazzini 2008, Menzly & Ozbas 2010: linked firms predict each other's "
                "returns - the economic basis for our hypothesis"),
            (0, "Feng et al. 2019 (RSR): selection as ranking over stock relations - our task framing"),
            (0, "Kim et al. 2019 (HATS): multiple relation types with attention - our graph design"),
            (0, "Li et al. 2020 (LSTM-RGCN): correlation-based adjacency - our edge construction"),
            (0, "Pareja et al. 2020 (EvolveGCN), Wu et al. 2020 (MTGNN): dynamic graphs - the top rung"),
            (0, "Bifet & Gavalda 2007 (ADWIN), Gama et al. 2014 (drift taxonomy) - our drift machinery"),
            (0, "Gu, Kelly & Xiu 2020: ML baseline ladder - our baseline suite; "
                "Lopez de Prado 2018: backtest hygiene"),
            (0, "Full review: 23 references + synthesis table + gap analysis (docs/literature_review.md)"),
        ],
        "notes": "Every reference was individually checked; two unverifiable suggestions were "
        "excluded during verification. v2 of the review adds the economic-foundations "
        "section, a comparison table of the core papers, and an explicit gap analysis - "
        "the four gaps map one-to-one onto our research questions and drift ladder.",
    },
    {
        "title": "Feasibility: already underway",
        "bullets": [
            (0, "Environment verified (smoke test, 2026-10-30, all checks passed):"),
            (1, "PyTorch 2.14 (CPU) + PyG 2.8.1; GCN/GAT convolutions run; yfinance download OK"),
            (0, "End-to-end proof-of-concept already runs (scripts/poc_selection_demo.py):"),
            (1, "98 stocks: data -> features -> correlation graph (294 edges) -> GCN -> ranked top-decile list"),
            (1, "Untrained GCN - proves the machinery; learning good scores IS the project"),
            (0, "Literature review v2 complete: 23 references, synthesis table, gap analysis"),
            (0, "CPU is sufficient: S&P 100 x daily data is small; no exotic dependencies"),
        ],
        "notes": "If asked to show it: run scripts/poc_selection_demo.py - about 5 seconds "
        "from the local cache, no network needed. Talking point: the demo's top scores went "
        "to the correlated energy majors (COP, CVX, XOM) with near-identical scores - that "
        "is message passing averaging graph neighbors, visible even in an untrained model. "
        "Full ranking saved in docs/poc_demo_output.md.",
    },
    {
        "title": "Timeline & deliverables (9-week template)",
        "bullets": [
            (0, "W1-2: data download, caching, feature engineering"),
            (0, "W3-4: graph construction + non-graph baselines (Lasso, XGBoost, MLP)"),
            (0, "W5-6: GCN/GAT models + backtest harness"),
            (0, "W7-8: drift ladder experiments (rungs 1-4)"),
            (0, "W9+: final report and presentation"),
            (0, "Deliverables: code repository, backtest report, literature review (done), final paper + slides"),
        ],
        "notes": "Weeks are a template until the course deadline is fixed - map to calendar "
        "dates once known. Note baselines come BEFORE the GNN: with/without comparison needs "
        "the without half built first.",
    },
    {
        "title": "Risks & mitigations",
        "bullets": [
            (0, "Graph does not help -> RQ1 answered negatively; analyze when/why - still a valid contribution"),
            (0, "yfinance data quality -> pull once, validate ranges, universe restricted to liquid large caps"),
            (0, "Overfitting / lookahead bias -> strictly walk-forward evaluation, no time shuffling"),
            (0, "Scope creep -> drift ladder is prioritized; every rung stands alone as a result"),
            (0, "CPU-only training -> verified sufficient for S&P 100 x daily; universe size caps compute"),
        ],
        "notes": "Biggest scientific risk (no graph gain) is handled by design: the comparison "
        "IS the thesis, either way. Biggest engineering risk (data flakiness) is handled by "
        "cache-once-then-experiment.",
    },
]


def build_pptx():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Title slide
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = SLIDES[0]["title"]
    slide.placeholders[1].text = f'{SLIDES[0]["subtitle"]}\n{SLIDES[0]["meta"]}'
    slide.notes_slide.notes_text_frame.text = SLIDES[0]["notes"]

    # Content slides
    for spec in SLIDES[1:]:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = spec["title"]
        body = slide.placeholders[1].text_frame
        body.clear()
        first = True
        for level, text in spec["bullets"]:
            para = body.paragraphs[0] if first else body.add_paragraph()
            first = False
            para.text = text
            para.level = level
            for run in para.runs:
                run.font.size = Pt(16)
        slide.notes_slide.notes_text_frame.text = spec["notes"]

    prs.save(OUT_PPTX)


def build_md():
    lines = ["# Proposal deck - mirror of proposal_deck.pptx", ""]
    lines.append(f"## {SLIDES[0]['title']}")
    lines.append(f"*{SLIDES[0]['subtitle']}*")
    lines.append(f"{SLIDES[0]['meta']}")
    lines.append(f"> Notes: {SLIDES[0]['notes']}")
    for i, spec in enumerate(SLIDES[1:], start=2):
        lines.append(f"\n## {i}. {spec['title']}")
        for level, text in spec["bullets"]:
            lines.append(f"{'  ' * level}- {text}")
        lines.append(f"> Notes: {spec['notes']}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_pptx()
    build_md()
    print(f"Wrote {OUT_PPTX} ({len(SLIDES)} slides)")
    print(f"Wrote {OUT_MD}")
