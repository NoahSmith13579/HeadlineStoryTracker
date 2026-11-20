# Headline Story Tracker: Project Guide

*For the project owner: what you are building, how it fits together, what you will deliver, and what to say about it. Written November 19, 2026. Companion to the Agent Implementation Spec.*

## 1. The project in one page

**What it is.** A live news story tracker. Every hour at minute 30 UTC, a GitHub Action pulls headlines from 25-40 outlets, groups them into stories (all headlines about the same real-world event), keeps story identities stable over time, and publishes static JSON to a GitHub Pages site. There is no backend. GitHub Actions is the compute, git is the database, and the visitor's browser is the only runtime for the site.

**The research question.** Can a learned same-story model beat off-the-shelf embedding similarity on near-miss stories, and how quickly does it decay as news vocabulary drifts? Everything else in the project exists to answer this credibly.

**What it shows a reader.** Not that you can call a pretrained model, but that you can define a fuzzy task precisely, build your own evaluation data and measure its noise, train and compare models against honest baselines, avoid the standard evaluation traps, run a system on messy live data and watch it degrade, and say plainly where it fails.

**Who does what.** The agent writes code, tests, and documentation. You do the work that makes the project yours and makes it credible: choosing and vetting sources, labeling, error analysis, running the GPU notebook, deciding when to promote a model, and writing the narrative. The write-up is the part people will actually read, so protect time for it.

**Minimum credible version.** Phases 0-5 plus the write-up: a live baseline, a gold set with an agreement score, a learned model, calibration, and a cluster-level comparison. Phases 6-10 deepen the project but are not required for it to stand on its own.

## 2. Architecture

### 2.1 Data flow

```
RSS feeds (25-40 outlets)
        |  hourly at :30 UTC, GitHub Actions cron
        v
 ingest -> clean -> headlines/YYYY-MM-DD.jsonl      (data branch)
        |
        v
 embed (MiniLM) -> blocking (top-15 neighbors, within 72 h)
        |
        v
 pair features + learned model -> P(same story)     (model trained offline)
        |
        v
 average-linkage clustering -> stories
        |
        v
 tracking (stable story IDs) -> publish JSON -> GitHub Pages site

 Offline and weekly:
 gold labels -> features -> LightGBM / cross-encoder -> calibration -> metrics.json
```

### 2.2 Where each piece runs

| Component | Runs on | When | Produces |
| --- | --- | --- | --- |
| Ingest, cluster, publish | GitHub Actions | hourly at minute 30 UTC | headline files on `data`, site JSON on Pages |
| Retrain and evaluate | GitHub Actions | weekly | `metrics.json`, a candidate model |
| Cross-encoder fine-tuning | Colab GPU | when you run it | weights as a release asset |
| Labeling | Your browser, local | when you label | gold files in `eval/gold/` |
| Site | Visitor's browser | on page load | renders the static JSON |

### 2.3 Design decisions and the reasons behind them

- **Actions as the backend, git as the database.** Free, auditable, and it forces small outputs. The cost is that schedules can run late and public-repo schedules get disabled after 60 days of inactivity, so every stage is idempotent and a keepalive step is required.
- **Two branches.** `main` holds code, gold labels, and small models. `data` holds the churn-heavy headline files, pruned at 60 days, so history on `main` stays clean.
- **Gold snapshot on `main`.** Your labels stay reproducible even after raw headlines are pruned from `data`.
- **Headlines only.** It keeps you clear of most copyright and terms-of-service trouble and keeps the repo small.
- **Pairwise model plus clustering.** This separates what the model decides (are these two headlines the same story) from how stories are assembled, and it yields interpretable probabilities.
- **Stable IDs by member overlap.** Re-clustering every run would reshuffle story IDs. Matching on member overlap lets stories persist, merge, and split in a way you can observe and report.
- **Shadow mode, fallback, no auto-promotion.** The learned model runs beside the baseline for a week, falls back automatically on failure, and only you promote new models.

## 3. The ML problem, precisely

- **Task.** Given two headlines (with source and time), predict the probability that they report the same specific real-world event or development. Clustering is a second step that consumes those probabilities.
- **Story definition.** Same event, not same topic. A follow-up on the same incident is the same story. A different incident of the same type is not. Your annotation guide makes this operational, and changing it changes every number you report.
- **Candidate pairs.** You never score all N-squared pairs. Blocking proposes each headline's 15 nearest neighbors within 72 hours. All evaluation happens on this candidate distribution, because random pairs are mostly trivial negatives and make every model look good.
- **Labels.** You label at the story level by correcting a seed clustering, and the tooling derives many pair labels from each correction. Pairs carry an ambiguous flag.
- **Why probabilities matter.** Clustering treats scores as distances. Uncalibrated scores make the cut threshold fragile and hard to explain, so calibration is part of the method rather than decoration.
- **Why the baseline stays.** The tuned cosine-threshold system ships first, remains in the repo, and is the comparison point for every claim you make.

## 4. Concepts you should be able to explain

For each one, be ready to say what it is, why it appears in this project, and what could go wrong.

- **Sentence embeddings and cosine similarity.** A model maps each headline to a vector so that similar meanings land close together. Cosine similarity measures the angle between vectors. It captures topic well but struggles to separate two different events on the same topic, which is exactly the gap your model targets.
- **Bi-encoder versus cross-encoder.** A bi-encoder embeds each headline separately (fast, cacheable). A cross-encoder reads both headlines together and can compare details, but must run once per pair (slower). Your pipeline uses the first for blocking and the second as a stronger, costlier scorer.
- **Blocking and the recall ceiling.** Blocking limits which pairs are ever scored. If the true partner is not among a headline's top 15 neighbors, no downstream model can recover it. You report blocking recall so readers know the ceiling.
- **Engineered pair features.** Cosine, lexical overlap, entity overlap, number conflicts, time gap, and source information. The conflict features (entities or numbers present in only one headline, such as 3 dead versus 5 dead) carry the signal that raw similarity misses.
- **Gradient boosting and SHAP.** LightGBM builds an ensemble of small decision trees and works well on mixed tabular features. SHAP attributes each prediction to features, so you can say which signals the model relies on and check that it matches intuition.
- **Calibration.** A calibrated model's 0.8 means roughly 80% of such pairs really are the same story. Isotonic regression fits a monotone correction on held-out data. A reliability diagram plots predicted probability against observed frequency, and expected calibration error summarizes the gap.
- **Clustering from pair probabilities.** Average linkage merges two groups based on their average pairwise distance. Single linkage merges on the closest pair and suffers from chaining, where one bad bridge pair fuses two stories. Showing that failure next to the fix is a strong figure.
- **Cluster metrics.** B-cubed precision and recall score each headline by how pure and complete its cluster is. ARI measures agreement between two partitions, corrected for chance. V-measure balances homogeneity and completeness. Each can mislead alone (for example, many singletons can flatter precision), so you report several.
- **Leakage, group splits, and bootstrapping by story.** Pairs from the same story are not independent. If they land on both sides of a split, scores inflate. You split by story, add a temporal holdout, and bootstrap by resampling stories so the intervals reflect the true unit of independence.
- **Label noise and Cohen's kappa.** Your judgments are fuzzy at the edges. Relabeling a sample days later gives a self-agreement kappa that sets a ceiling on how much to trust small metric differences. It measures consistency, not correctness.
- **Anchoring bias.** You correct a seed clustering rather than labeling from scratch, so your labels lean toward the seed's choices. This can flatter the baseline. Shuffling the display and hiding scores reduces it but does not remove it, so it belongs in your limitations.
- **Active learning and why the audit needs a random slice.** Uncertainty sampling labels the pairs the model is least sure about, which is label-efficient. Those samples are biased by design, so accuracy claims must come from the random slice only.
- **Weak supervision.** Free noisy labels (identical titles, matching wire credits) expand training data. You measure their precision, use them for training only, and run an ablation to show whether they help.
- **Concept drift.** News vocabulary and event types shift. You test it by freezing a model on week 1, scoring later weeks, and comparing with retrained alternatives, alongside unsupervised signals such as singleton rate and embedding centroid shift.
- **Story identity tracking.** Matching each run's clusters to the previous run's by member overlap (Hungarian assignment), with explicit handling of merges, splits, and deaths.
- **Stacking and out-of-fold predictions.** To feed the cross-encoder's score into LightGBM, the score for each training pair must come from a model that never saw that pair's story. Otherwise the stacker learns from leaked, over-optimistic scores.

## 5. Deliverables

| Deliverable | Phase | Where it lives | What it shows |
| --- | --- | --- | --- |
| Live story site | 2 | `site/`, Pages | A pipeline that runs unattended |
| Ingestion and data health panel | 1, 9 | `data` branch, `health.json` | Robustness on messy live data |
| Annotation guide | 3 | `eval/gold/` | A precisely defined task |
| Gold dataset (800-1,000 headlines) | 3 | `eval/gold/` | You built your own evaluation data |
| Agreement report (kappa, interval, disagreement analysis) | 3 | `metrics.json` | Awareness of label quality |
| Blocking recall report | 3 | `metrics.json` | Ceiling analysis |
| Baseline suite (cosine, TF-IDF, title match, HDBSCAN) | 2-5 | `src/hst/models`, `metrics.json` | Honest comparison |
| Learned pairwise model with SHAP and ablations | 4 | `models/`, `metrics.json` | The core ML contribution |
| Calibration and reliability diagram | 4 | `metrics.json` | Probability quality |
| Cluster-level evaluation and chaining demo | 5 | `metrics.json`, `docs/figures` | Systems-level thinking |
| Cross-encoder and stacking | 6 | `notebooks/`, release asset | Deep learning and leakage-safe stacking |
| Shadow deployment and fallback | 7 | pipeline | Production discipline |
| Learning curve (active versus random) | 8 | `metrics.json` | Label efficiency |
| Drift experiment | 8 | `metrics.json` | Temporal robustness |
| Live audit time series | 9 | `eval/audit/` | An ongoing, honest score |
| Model page, README, datasheet, model card | 9 | `site/`, repo | Communication |
| Forecast scoreboard (stretch) | 10 | `forecast_log.jsonl` | Predict-then-verify |

## 6. Structuring your write-up

Aim for 2,500-4,000 words as the README or a standalone post, with the live site linked at the top. State the evaluation protocol before the results so readers know the numbers can be trusted.

| Section | What it must contain |
| --- | --- |
| 1. Summary | The research question, a one-paragraph answer with the headline result and its interval (even if the answer is that the learned model helped only modestly), and the link to the site |
| 2. Problem and definition | The story definition and 4-6 real example pairs: clear positives, clear negatives, and near-miss negatives |
| 3. System overview | The architecture diagram, and how static hosting shaped the design |
| 4. Data | How sources were chosen, collection, cleaning rules, and a table of the messy things you hit and what you did (dead feeds, missing timestamps, syndicated copies, schema changes) |
| 5. Labels | The guide, the process, set size and time span, kappa with its interval, disagreement categories, and anchoring bias |
| 6. Methods | Baselines, features, models, calibration, clustering, and identity tracking |
| 7. Evaluation protocol | Splits, leakage controls, metrics, intervals, and what was held out and touched once |
| 8. Results | Main comparison table, PR curves, cluster metrics, ablations, calibration, error taxonomy |
| 9. Operating the system | Stability, shadow-mode findings, drift, learning curve, live audit |
| 10. Limitations | Everything in section 7 of this guide that applies |
| 11. Next steps | What you would do with more labels, time, or compute |
| 12. Reproducing it | Commands, runtime, and what data is and is not included |

**Figures worth making:** PR curves for all models; reliability diagrams before and after calibration; a cluster-level comparison across methods with intervals; the feature-group ablation chart; the learning curve (active versus random sampling); the drift chart by week; two or three chaining examples; the live audit series; the data health panel.

**Writing principles.** Lead with the question and the answer. Report intervals rather than single numbers. Show null and negative results, because they are evidence that you measured honestly. Use real headlines as examples. Tie every claim to a script or file in the repo. Do not generalize beyond your sources and time period.

## 7. Topics to discuss in the write-up

**Design choices**

- Why a pairwise model plus clustering, rather than fine-tuning embeddings end to end or prompting an LLM. Be fair to the alternatives. An LLM judge may do better on hard pairs but is costlier and harder to schedule. Running one offline on the test pairs as an extra baseline would be an interesting addition (it is not in the agent spec, so it is your call).
- Why a story is an event, and what that definition excluded (opinion pieces, roundups, live blogs).
- Why average linkage, and what chaining looked like when you tried single linkage.
- Why a 72-hour window and k = 15, and what the blocking recall number says about those choices.
- Why headlines only.

**Evaluation integrity**

- Splits by story, a temporal holdout, a test set touched once, thresholds tuned on validation only, and a separately tuned cluster threshold.
- Random and uncertainty-sampled labels kept apart.
- Intervals from story-level bootstrap.
- The full test set versus the clean subset, and what the gap says about label ambiguity versus model weakness.
- If the learned model does not beat the baseline, say so and diagnose why.

**Data and ethics**

- Feed terms, robots.txt, and why you store no article text.
- Source selection: English only, limited to outlets with usable RSS feeds, and your attempt at regional and editorial variety. State clearly that you are clustering coverage, not measuring truth or bias.
- Time period and event coverage. One dominant story can dominate the gold set, so report how much each story contributes.

**Limitations to state plainly**

- A single annotator and a small gold set, so intervals are wide.
- Anchoring on the baseline seed during labeling.
- A short time span and few major events, so generalization is uncertain.
- The live audit samples are small.
- The drift experiment is limited by how many separate labeled weeks you have.

**Failure stories worth telling.** Evolving stories whose headlines change as facts emerge, number conflicts the features missed, syndicated rewrites, templated headlines such as morning briefings, and stories the blocking step never connected. Concrete failures with a diagnosis are more persuasive than a clean table.

## 8. Likely questions and the shape of a good answer

| Question | Shape of the answer |
| --- | --- |
| Why not just use embeddings with a threshold? | That was the baseline. Show the near-miss failures and the PR-AUC and B-cubed gap with intervals. |
| How do you know it is not overfit to your labels? | Group split by story, temporal holdout, test touched once, clean-subset results, live audit on fresh pairs. |
| How did you prevent leakage? | Splits by story, fit-on-train-only tests, out-of-fold stacking, random and uncertainty samples kept apart. |
| What does your kappa mean for the model? | It bounds which differences are trustworthy. Gains smaller than label noise are not meaningful. Discuss which categories drove disagreements. |
| How do you handle an evolving story? | The annotation rule, stable-ID tracking, merge and split handling, with examples. |
| What happens when a feed breaks? | Per-feed isolation, the health panel, never failing the run, retention, and keepalive. |
| Why calibrate? | Clustering consumes probabilities. Show reliability diagrams before and after, and the effect on cluster metrics. |
| How would this scale 100 times? | Approximate nearest-neighbor search for blocking, incremental clustering, a real queue and database. Name the first bottleneck you would hit. |
| Why Actions and static hosting? | The constraints make it reproducible, free, and auditable. State the limits you accepted. |
| What would you do with more time? | Multiple annotators, other languages, an LLM-judge baseline, a learned clustering objective, a longer drift study. |

## 9. Risks and how to handle them

| Risk | Mitigation |
| --- | --- |
| Labeling takes longer than planned | Start with about 400 headlines, run the whole pipeline, and grow the set later. Tell the agent if you plan to grow it so split assignment can be made append-only and existing assignments never change. |
| The learned model does not beat the baseline | It is a valid result. Check for leakage or bugs, analyze why, and do not tune on the test set. |
| A few huge stories dominate the gold set | Sample evenly across days and sources, report each story's share, and report metrics weighted per story as well as per pair. |
| Single-annotator credibility | Ask one person to label the same 100 pairs for inter-annotator agreement, or use an LLM as a clearly disclosed second annotator. |
| A source objects or changes its terms | Remove it, note it in the source list, and let the health panel show the gap. |
| Actions disabled by inactivity | Keepalive step, plus a weekly look at the Actions tab for the first two months. |
| Repo bloat | 60-day retention, headlines only, nothing large in git, and squash the `data` branch history if needed. |
| Scope creep | Phases 0-5 plus the write-up are the core. Treat phases 6, 8, and 10 as optional depth. |

## 10. Your tasks and rough time

**Calendar reality.** RSS feeds only expose recent items, so you cannot backfill history. The gold set needs at least four separate weeks of collected data and the drift experiment needs several labeled weeks, which means calendar time, not coding time, is the bottleneck. Start the scheduled ingestion as early as possible (Phases 1 and 2) and leave it running while you build the rest. This makes the realistic timeline longer than the four-week sketch from earlier in our conversation, probably six to eight weeks end to end.

| Phase | Your task | Rough time (my estimates; adjust to yourself) |
| --- | --- | --- |
| 0 | Set the Pages source to GitHub Actions | 5 minutes |
| 1 | Review the source list and check each outlet's feed terms | 1-2 hours |
| 3 | Label 800 or more headlines at story level | 6-10 hours, in short daily sessions |
| 3 | Review and finalize the annotation guide | 1 hour |
| 3 | Re-label 100 pairs after waiting 3-7 days | 30-45 minutes |
| 4 | Fill the cause column for about 100 error rows | 1 hour |
| 6 | Run the Colab notebook and publish the weights | 1-2 hours |
| 7 | Review the shadow-mode comparison and promote | 30 minutes |
| 8 | Weekly audit labeling, plus a one-time 50-pair weak-label audit | about 15 minutes per week, and 30 minutes once |
| 9 | Write the narrative sections and choose a license | 6-10 hours |

Labeling in short daily sessions beats one marathon, because your judgment gets inconsistent when you are tired and the agreement check will show it.
