# Synthetic Documentation Generation & Evaluation

This project generates and evaluates synthetic versioned technical documentation (v1.2 → v1.3) using multiple LLMs, consensus scoring, and Pi Scorer rubrics.

---

## Approach

### Step 1 — Research & Observations
- Reviewed real-world version guides and changelogs.  
- Documented common structure, terminology, and formatting patterns.  
- Noted typical LLM behaviors and failure modes for changelog/guide generation.

### Step 2 — Prompt Design
- Created three prompt variants: **best**, **medium**, and **worst** to test prompt sensitivity.  
- Ran each prompt with **OpenAI**, **Claude**, and **Llama** (9 total generations).  
- Selected the **best prompt** (consistent, highest-quality outputs) for the next stages.

### Step 3 — Model & Parameter Exploration
- With the best prompt fixed, explored `temperature`, `top_p`, and `max_tokens`.  
- Performed a grid search across parameter combinations for the three model families.  
- Produced **144 candidate** v1.2 + changelog pairs.

### Step 4 — Consensus Scoring (grounding metrics)
Computed statistical metrics to assess generation fidelity and diversity:
1. **Word repetition** (bigram repetition — inverted so lower repetition → higher score).  
2. **Cosine similarity (prompt → output)** — measures prompt adherence.  
3. **Cosine similarity (across outputs)** — measures agreement / consensus among multiple generations.  
- Aggregated these into a `consensus_score` to rank candidates.

### Step 5 — Pi Scorer Evaluation
- Applied Pi Scorer rubrics to all candidates. Metrics used:
  - Total score  
  - Realism  
  - Prompt adherence  
  - Clarity  
  - Factual consistency  
  - Completeness  
  - Technical accuracy

### Step 6 — Correlation Analysis
- Hypothesis: consensus and Pi scores should correlate positively.  
- Observed result: **negative correlation** (example: −0.37; other analyses show ranges like −0.44 to −0.55).  
- Interpretation: consensus metrics emphasize diversity/consensus properties, while Pi rubrics favor grounded, human-aligned quality — these objectives can conflict.

### Step 7 — Quadrant Selection
- Plotted candidates on a Pi-score vs. consensus-score plane.  
- Selected the **High Pi + High Consensus** quadrant (the “Goldilocks” region) to pick the best model + parameter settings.

### Step 8 — Final Generation (v1.3)
- Used the selected best model + parameters + best prompt to generate the final **v1.3 guide** from the chosen v1.2 + changelog.  
- Evaluated the v1.3 output using the final Pi Scorer rubrics tailored to release notes.

### Step 9 — Assessment
- Compared raw Pi scores and calibrated Pi scores for the final v1.3.  
- Performed manual spot checks and rubric-driven checks to confirm professional quality and fidelity.

---

## Repository / Package

### Scripts
- `text_generation.py` — run grid search and generate grounding data (v1.2 + changelogs).  
- `ensemble_consensus.py` — compute consensus/statistical rubrics for all candidates.  
- `pi_labs_scorer.py` — call Pi Scorer and collect raw rubric scores.  
- `cal_pi_scores.py` — calibrate Pi scores for the v1.2 guide & changelog (if applicable).  
- `analysis.ipynb` — exploratory data analysis, correlation studies, visualizations.  
- `create_quadrants.py` — create Pi vs consensus quadrants to pick the Goldilocks candidates.  
- `generate_v1.3.py` — generate v1.3 using the selected best model + parameters.  
- `evaluate_answer.py` — evaluate and (optionally) calibrate the v1.3 output.

---

## Plots — Interpretation (summary)

**Image 1 — Correlation Matrix**
- **Pi vs Consensus:** inversely related (example range: `-0.44` to `-0.55`) — higher Pi scores often align with *less* similarity to ensemble outputs.  
- **Pi rubric cluster:** Pi rubrics (prompt adherence, completeness, technical accuracy) are highly correlated (`0.88–0.97`), indicating they capture a common notion of “task-aligned quality.”  
- **Factual consistency:** weak or orthogonal correlation with other metrics (`~ -0.27 to 0.16`), meaning factual quality behaves independently and must be measured explicitly.

**Model family highlights**
- **Claude**
  - Strong task-following (prompt adherence, completeness, clarity `~0.93–0.98`).  
  - Poorer factual consistency on many configurations (`~0.15–0.35`).  
  - Pi totals moderate (`~0.55–0.65`), suggesting Pi weights factual correctness significantly.

- **Llama**
  - Moderate, steady performance across metrics — fewer extreme highs but also fewer catastrophic lows.  
  - Shows parameter instability at extreme settings (large swings across metrics).  
  - Best consensus scores (`~0.60–0.70`) — often closest to the ensemble average (a “safe” choice).

- **OpenAI**
  - Most consistent across parameter sweeps (lowest variance).  
  - Highest ceiling for prompt adherence (`~0.90–1.0`) — excels at strict instruction following.  
  - Still exhibits low factual consistency on average (`~0.07–0.25`) — a common weakness across families.

---

## Shortcomings & Recommended Fixes

1. **Low factual consistency in grounding data**  
   - Impact: v1.3 may inherit factual errors from grounding data.  
   - Fix: curate a small set of *real* version guides as positive examples for factual consistency; pick negative examples from grid candidates with low factual scores. Use these to calibrate or retrain the factual-consistency evaluator (or use preference pairs to align scoring).

2. **Metric alignment vs. diversity trade-off**  
   - Impact: consensus rewards diversity/ensemble alignment while Pi rewards grounded, conservative outputs.  
   - Fix: keep a small set of top candidates from the Goldilocks quadrant rather than a single winner to preserve useful variation; use human-in-the-loop checks on edge cases.




