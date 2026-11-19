# Headline Story Tracker: Agent Implementation Spec

*Phased build specification for a coding agent. Written November 19, 2026. Companion document: Headline Story Tracker: Project Guide (for the human owner).*

## 0. Instructions to the implementing agent

Read this section fully before writing code.

1. Build one phase at a time, in order. Do not begin a phase until the previous phase's acceptance criteria pass and the owner confirms.
2. Each phase has a Goal, Tasks, Files, and Acceptance criteria. A **HUMAN GATE** marks work only the owner can do (labeling, approving sources, changing repo settings, running Colab). When you reach one, stop, summarize what you built, and give the owner exact instructions.
3. Do not add scope. If something is ambiguous, choose the simplest option consistent with this spec, log it in `docs/DECISIONS.md` (date, decision, reason), and continue.
4. Every phase ends with passing `pytest`, clean `ruff check`, an entry in `docs/CHANGELOG.md`, and a report to the owner using the template in Appendix A.
5. Determinism: one `SEED = 1337` in `src/hst/config.py`, used everywhere. Pin dependencies.
6. Never commit secrets, embeddings, model caches, or article text. Never fetch article pages.
7. Write small pure functions with unit tests. Pipeline stages communicate through files that follow the schemas in section 4, and each stage must be runnable alone via the CLI (`python -m hst <command>`).
8. Failure policy: one failing feed must never fail a run. Record it in `health.json` and continue. A run fails only if zero feeds succeed or output validation fails.
9. Library APIs change. Check installed versions and current docs instead of relying on memory. If reality differs from this spec, follow reality and log it in `DECISIONS.md`.
10. Never report guessed numbers. Every metric must come from a script in the repo that the owner can re-run.

## 1. Project summary

A live news story tracker hosted entirely on GitHub Pages with no server. GitHub Actions runs on a schedule: it ingests RSS headlines from 25-40 outlets, groups them into stories (clusters of headlines about the same real-world event), tracks story identity across runs, and publishes static JSON that a plain-JavaScript frontend renders.

The ML core is a **learned pairwise same-story classifier**, P(same story | headline A, headline B), that replaces a hand-tuned cosine threshold. Pair probabilities feed hierarchical clustering. Around it sit a labeled gold set, honest baselines, calibration, a drift experiment, and a live audit.

**Research question:** can a learned same-story model beat off-the-shelf embedding similarity on near-miss stories, and how quickly does it decay as news vocabulary drifts?

**Definition of story (use everywhere, including the annotation guide):** two headlines are the same story if and only if they report the same specific real-world event or development. Same topic is not enough. A follow-up on the same incident is the same story. A different incident of the same type is a different story.

**Non-goals:** storing article text or summaries, any always-on server, user accounts, LLM calls at runtime, non-English clustering in v1, financial prediction.

## 2. Fixed technical decisions

| Area | Decision |
| --- | --- |
| Language and tooling | Python 3.11, `pyproject.toml`, pinned `requirements.txt`, `ruff`, `pytest`, `pydantic` for schemas. Pin current stable versions at first install. |
| Frontend | Plain HTML, CSS, ES modules, no build step. One charting library (Observable Plot or D3) vendored under `site/vendor/` so the site makes no third-party runtime requests. |
| Compute | GitHub Actions, `ubuntu-latest`, CPU only. Scheduled pipeline runs must finish in under 10 minutes. |
| Hosting | GitHub Pages deployed from the workflow (`actions/upload-pages-artifact` and `actions/deploy-pages`). Generated site JSON is never committed to `main`. |
| Branches | `main`: code, config, site source, gold labels, small model files. `data`: append-only headline JSONL and pipeline state, 60-day rolling retention. The workflow checks out `data` as a second worktree. |
| Gold data | Lives on `main` under `eval/gold/` and includes a **snapshot of headline text**, so labels stay reproducible after `data` is pruned. |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized. Never committed. Recompute, or restore with `actions/cache`. |
| NER | spaCy `en_core_web_sm`. |
| Classical models | scikit-learn, LightGBM, SHAP (offline only). Isotonic calibration. |
| Cross-encoder | MiniLM-class model fine-tuned offline in Colab. Weights are a GitHub Release asset that the workflow downloads and caches. Not stored in git. |
| Clustering | `scipy.cluster.hierarchy` average linkage on distance = 1 - P, run per connected component of the candidate-pair graph (never a full N x N matrix). |
| Schedules | `pipeline.yml` every 30 minutes (`*/30 * * * *`, configurable). `weekly.yml` on Sundays. Cron can run late, so every stage must be idempotent. Use a `concurrency` group so runs never overlap. |
| Keepalive | Scheduled workflows on public repos are disabled after 60 days without repository activity. Add a keepalive step (for example re-enabling the workflow with `gh workflow enable`, or a pinned keepalive action) and verify it works during the first month. |
| Legal and ethics | Store headline, URL, source, timestamps only. Respect each feed's terms and robots.txt. Send a descriptive User-Agent that includes the repo URL. |

## 3. Repository layout

```
.
├── .github/workflows/
│   ├── ci.yml                 # ruff + pytest on push and PR
│   ├── pipeline.yml           # cron: ingest, cluster, publish, deploy Pages
│   └── weekly.yml             # cron: retrain, evaluate, write metrics.json
├── config/
│   ├── sources.yaml
│   └── settings.yaml          # windows, thresholds, k, retention
├── src/hst/
│   ├── __main__.py            # CLI entry (python -m hst <command>)
│   ├── config.py  schemas.py
│   ├── ingest.py  clean.py  health.py
│   ├── embed.py  blocking.py  features.py
│   ├── models/                # baseline.py gbm.py cross_encoder.py calibrate.py
│   ├── cluster.py  tracking.py  publish.py
│   ├── labeling/              # snapshot.py export.py select.py
│   └── eval/                  # metrics.py splits.py agreement.py run_eval.py
├── tools/label/               # local story-level labeling UI (static files)
├── eval/gold/                 # headlines_snapshot.jsonl pairs.jsonl stories.jsonl ANNOTATION_GUIDE.md
├── models/                    # small committed artifacts + model_card.json
├── site/                      # index.html model.html app.js style.css vendor/
├── notebooks/                 # Colab notebooks for offline training
├── tests/  docs/  pyproject.toml  requirements.txt  README.md
```

The `data` branch holds:

```
headlines/YYYY-MM-DD.jsonl        # by first_seen_at date (UTC)
state/stories_state.json          # story registry for ID continuity
state/feed_state.json             # ETags, Last-Modified, failure counts
state/health_history.jsonl        # rolling 14 days
state/forecast_log.jsonl          # Phase 10 only
```

## 4. Data contracts

Define each as a `pydantic` model in `schemas.py`. Validate on every write. Files are UTF-8, one JSON object per line for `.jsonl`.

**Headline record** (`headlines/YYYY-MM-DD.jsonl`)

| Field | Type | Notes |
| --- | --- | --- |
| `id` | str | first 16 hex chars of sha1(`url_canonical`) |
| `source_id` | str | key from `sources.yaml` |
| `title_raw` | str | exactly as in the feed |
| `title_clean` | str | after the Phase 1 cleaning rules |
| `url`, `url_canonical` | str | canonical form has tracking params removed |
| `published_at` | str or null | ISO 8601 UTC |
| `published_at_source` | `feed` or `first_seen` | which timestamp was used |
| `first_seen_at` | str | ISO 8601 UTC |
| `lang` | str | ISO 639-1 code or `und` |
| `dup_of` | str or null | id of the earliest headline with the same normalized title |

**Source config** (`config/sources.yaml`): `id`, `name`, `feed_url`, `region`, `category` (wire, national, international, regional, specialty), `is_wire` (bool), `language`, `tos_note`, `tos_checked` (bool), `enabled` (bool).

**Pair label** (`eval/gold/pairs.jsonl`): `pair_id` (sha1 of the two ids, sorted), `a_id`, `b_id` (with `a_id < b_id`), `label` (1 or 0), `ambiguous` (bool), `label_source` (`human_story`, `human_pair`, `weak`, `active`), `round` (int), `labeled_at`.

**Story object** (inside `site/data/stories.json`): `story_id` (`s_` plus 8 hex), `title` (medoid headline), `keywords` (list), `first_seen`, `last_seen`, `n_headlines`, `n_sources`, `sources` (list of source ids), `hourly_counts` (last 72 integers), `headlines` (at most 12 of `id`, `title`, `url`, `source_id`, `published_at`), `merged_from` (list), `split_from` (str or null), `forecast` (null until Phase 10).

**stories.json top level:** `generated_at`, `window_hours`, `model_version`, `n_headlines_in_window`, `n_stories`, `stories`. Keep the file under 500 KB.

**health.json:** `generated_at`; `run` (`duration_s`, `n_new_headlines`, `n_feeds_ok`, `n_feeds_total`); `feeds` (list of `id`, `status`, `http_status`, `items`, `new_items`, `latency_ms`, `consecutive_failures`, `last_success`); `quality` (`null_timestamp_rate`, `dup_rate`, `non_english_rate`); `clustering` (`n_clusters`, `median_size`, `singleton_rate`, `id_persistence`).

**metrics.json:** written only by `eval/run_eval.py`. Top-level keys are added phase by phase: `agreement`, `blocking_recall`, `pair_level`, `cluster_level`, `calibration`, `ablations`, `learning_curve`, `drift`, `live_audit`. Every value that is an estimate carries a bootstrap 95% interval as `[lo, hi]`, plus `n` and `model_version`.

## 5. Phases

### Phase 0: Scaffold and CI

**Goal:** an empty but working repo with green CI and a deployed placeholder site.

**Tasks**

1. Create the layout from section 3, `pyproject.toml`, pinned `requirements.txt`, `ruff` and `pytest` config.
2. Implement `config.py` (loads `settings.yaml`) and `schemas.py` (all models from section 4).
3. Implement the CLI with stub subcommands: `ingest`, `embed`, `cluster`, `publish`, `labeling`, `eval`.
4. Write `ci.yml`. Write a `pipeline.yml` skeleton with `workflow_dispatch`, the cron line commented out, the `concurrency` group, minimal permissions (`contents: write`, `pages: write`, `id-token: write`), and a job that deploys a placeholder `site/index.html`.
5. Create the empty `data` branch.

**HUMAN GATE:** owner sets Pages source to GitHub Actions in repo settings.

**Acceptance**

- CI is green on a pull request.
- The Pages URL serves the placeholder.
- `python -m hst --help` lists all subcommands.
- A schema round-trip test passes for every model.

### Phase 1: Ingestion, cleaning, storage, health

**Goal:** reliable headline collection into JSONL on the `data` branch.

**Tasks**

1. **Sources.** Draft 25-40 candidate feeds across wire services, national, international, and regional outlets with varied editorial perspectives. Find feed URLs from each outlet's official RSS page. Fetch each and confirm it parses. Do not guess URLs. Leave out anything you cannot verify. Set `tos_checked: false` for all.
2. **`ingest.py`.** Use `httpx` (15 s timeout, 2 retries with backoff) and `feedparser`. Send conditional GET headers using stored ETag and Last-Modified. Extract only title, link, and published date. Discard summary and content fields at parse time.
3. **`clean.py`.** Decode HTML entities, apply Unicode NFKC, normalize whitespace, strip outlet-name suffixes and prefixes (such as `  | Outlet `, `LIVE:`, `WATCH:`), canonicalize URLs (lowercase host, drop fragments and tracking params such as `utm_*` and `fbclid`), and detect language with a fixed seed. Keep non-English items but mark them. Only English items enter clustering in v1.
4. **Timestamps.** Parse to UTC. If the date is missing, more than 10 minutes in the future, or older than 7 days at first sight, set `published_at` to null and use `first_seen_at`.
5. **Duplicates.** Skip records whose `id` already exists. If the normalized title matches an earlier headline from another outlet, keep the record and set `dup_of`. Syndication is signal and must not be dropped.
6. **Storage.** Append to the daily file by `first_seen_at` date. The operation must be idempotent.
7. **`health.py`.** Record per-feed status, latency, items, new items, and consecutive failures, plus the aggregate quality rates. Write `health.json` and append to `health_history.jsonl`.
8. **Retention script.** `hst ingest --prune` deletes daily files older than 60 days.
9. **Tests.** Use recorded fixtures: RSS and Atom feeds, malformed XML, missing dates, duplicate entries, entity-heavy titles, and a feed returning 304. Include at least 30 table-driven cleaning cases.

**HUMAN GATE:** owner reviews `sources.yaml`, checks each outlet's terms, and sets `tos_checked`.

**Acceptance**

- Running `ingest` twice in a row creates zero new records the second time.
- A simulated feed outage does not fail the run and shows up in `health.json`.
- At least 90% of enabled feeds succeed in a live run.
- No output file contains summary or body text.

### Phase 2: Baseline pipeline and MVP site

**Goal:** an end-to-end live site using a cosine-threshold baseline. This baseline stays in the repo permanently as the comparison point.

**Tasks**

1. **`embed.py`.** Encode `title_clean` in batches, L2-normalize, and cache keyed by a hash of the title text.
2. **`blocking.py`.** For each English headline in the 72-hour window, take the top k = 15 neighbors by cosine similarity among headlines within 72 hours. Use plain numpy. Return pairs with cosine and time gap, in deterministic order.
3. **`models/baseline.py`.** Pair score equals cosine similarity. The default `tau` is a placeholder in `settings.yaml` and gets tuned in Phase 5.
4. **`cluster.py`.** Build a sparse candidate graph, find connected components, and run average-linkage clustering inside each component, with distance 1 for non-candidate pairs. Cut at `1 - tau`.
5. **`tracking.py`.** Match new clusters to the registry by member Jaccard overlap using Hungarian assignment, with a minimum overlap of 0.5. Handle four cases: continuation (keep the ID), merge (the larger old ID survives, record `merged_from`), split (the largest piece keeps the ID, the others get new IDs with `split_from`), and death (no members for 72 hours, so archive). Persist to `state/stories_state.json`.
6. **Labels.** The representative title is the medoid headline. Keywords are the top 5 c-TF-IDF terms across stories, with stopwords removed.
7. **`publish.py`.** Write `stories.json` (top 100 stories by recency-weighted size, under 500 KB), `stories/<story_id>.json` detail files, and `health.json` into `site/data/` at build time.
8. **Frontend MVP.** A story list with representative headline, source count, first-seen time, and hourly sparkline. Clicking a story shows its headlines grouped by source. A header shows last updated time and feeds up of total.
9. **`pipeline.yml`.** Check out `main` and `data`, restore model caches, run ingest, embed, cluster, track, and publish, commit data changes to `data` (retry once on push conflict), then deploy Pages. Enable the cron.

**Acceptance**

- Over 48 hours, at least 90% of scheduled runs complete and deploy.
- Median runtime is under 10 minutes and `stories.json` is under 500 KB.
- `id_persistence` (share of stories present in two consecutive runs that keep their ID) has a median of at least 0.95.
- Unit tests cover tracking with synthetic continuation, merge, split, and death scenarios.

### Phase 3: Gold data, labeling tools, evaluation code

**Goal:** a defensible gold set, an annotation guide, agreement tooling, and tested metric code.

**Tasks**

1. **Snapshot.** `hst labeling snapshot --start --end` copies selected headlines (all fields) into `eval/gold/headlines_snapshot.jsonl`. Target 800-1,000 headlines drawn from at least 4 separate weeks, including at least one heavy breaking-news period and some quiet days. Sample evenly across days and sources.
2. **Seed clusters.** Run the baseline over the snapshot and write `stories_seed.jsonl`. Seeding from the baseline creates an anchoring bias in the human labels. Mitigate by shuffling cluster order in the UI and hiding scores. The owner must discuss this bias in the write-up.
3. **Labeling UI** (`tools/label/`, static HTML and JS, runs from a local file or `python -m http.server`). Story-level view with actions to move a headline, split a headline off, merge clusters, and flag as ambiguous. Keyboard shortcuts for all four. Autosave to `localStorage`. Import and export `stories.jsonl` by file. A second pair mode shows two headlines with the model score hidden, for the agreement check.
4. **`labeling/export.py`.** Convert story labels into pair labels for blocking candidates, plus random negatives and hard negatives (high cosine, different story). Write `pairs.jsonl` with `label_source: human_story`. Roundup or briefing headlines are labeled `roundup` at story level and excluded from pairs.
5. **`eval/agreement.py`.** Sample 100 pairs (50 positives, 50 hard negatives). Write a blank relabel file. After relabeling, compute Cohen's kappa, raw agreement, a bootstrap 95% interval, and a disagreements CSV. Write the `agreement` block of `metrics.json`.
6. **`eval/metrics.py`.** Pair level: precision, recall, F1, average precision, PR curve points. Cluster level: B-cubed precision, recall, F1, ARI, V-measure. Add `bootstrap_by_story` that resamples stories, not pairs. Add `eval/splits.py` with a group split by story, keeping only pairs whose two headlines are on the same side, and a temporal split.
7. **`ANNOTATION_GUIDE.md`.** Draft from the definition in section 1. Include rules for follow-ups, evolving counts, live blogs, opinion pieces, roundups, and reposted wire copy. The owner edits and owns the final version.
8. **Blocking recall report.** Compute the share of gold same-story pairs that appear among the candidate pairs. This is the ceiling for every later model. Write it to the `blocking_recall` block of `metrics.json`.

**HUMAN GATE:** the owner labels at least 800 headlines at story level, then waits 3-7 days and completes the agreement relabel. Stop and hand off.

**Acceptance**

- The labeling UI loads 1,000 headlines responsively and a round trip of export then import preserves every label.
- Metric functions match hand-computed values on small fixtures. Leakage test: no story appears in both train and test.
- `pairs.jsonl` validates against the schema.
- The blocking recall number is reported with a confidence interval.

### Phase 4: Features and classical models

**Goal:** a trained, calibrated, interpretable pairwise model, evaluated fairly against the baseline.

**Tasks**

1. **Fixed splits.** Create and commit `eval/gold/splits.json`: story id mapped to `train`, `val`, or `test` (about 60/20/20, split by story), plus a temporal holdout definition (train on earlier weeks, test on the latest week). Every later phase must reuse this file unchanged.
2. **`features.py`.** Compute about 25 features per candidate pair, grouped as below. Each feature is a pure function with a unit test.

| Group | Features |
| --- | --- |
| Semantic | MiniLM cosine; cosine from one second embedding model (choose once, record it) |
| Lexical | TF-IDF cosine (fit on train split only); character n-gram Jaccard; shared token count and ratio; length ratio |
| Entities | spaCy PERSON, ORG, GPE, LOC overlap (Jaccard); count of entities present in only one headline; whether the main entity of one headline appears in the other |
| Numbers | number-set overlap; number conflict flag (same noun after the number, different value, such as 3 dead versus 5 dead) |
| Time | absolute hours apart; same UTC day |
| Source | same outlet; both wire; one wire; source-pair prior (smoothed same-story rate, computed on train split only) |

3. **Feature rules.** Every feature must be symmetric (swapping A and B gives the same vector, enforced by a test). Anything fitted on data (TF-IDF vocabulary, source priors) is fitted on the training split only, enforced by a test that passes only training ids.
4. **Tuned baseline.** Choose the cosine threshold on the validation split to maximize pair F1. This is the number the learned model must beat.
5. **`models/gbm.py`.** Train a standardized logistic regression and a LightGBM classifier. Random search of at most 30 trials, grouped K-fold by story, early stopping. Compare class weighting against none.
6. **`models/calibrate.py`.** Fit isotonic regression on a held-out grouped fold. Store the calibrator as JSON (threshold arrays applied with `numpy.interp`), not a pickle, so it survives library upgrades. Report reliability-diagram data and expected calibration error.
7. **Evaluation.** `eval/run_eval.py --stage pair` reports PR-AUC and F1 (at the validation-chosen threshold) for the tuned cosine baseline, a TF-IDF baseline, logistic regression, and LightGBM, on the group-split test set and the temporal holdout, each with story-bootstrap 95% intervals. Report on both the full test set and a clean subset that excludes ambiguous-flagged pairs and pairs the owner disagreed with themselves on.
8. **Interpretation.** Write mean absolute SHAP values for the top 15 features to `pair_level.feature_importance`. Run feature-group ablations (drop one group, retrain, report the change in PR-AUC with intervals).
9. **Error taxonomy.** Export `eval/errors.csv` with the 50 most confident false positives and 50 most confident false negatives: both headlines, score, source, time gap, and an empty `cause` column.
10. **Artifacts.** `models/gbm.txt` (LightGBM native format), `models/calibrator.json`, and `models/model_card.json` (training data hash, date, split file hash, feature list, metrics, seed, library versions).

**HUMAN GATE:** the owner fills the `cause` column (suggested categories: same topic different event, evolving story, rewritten syndication, opinion piece, label error, other).

**Acceptance**

- Swap-symmetry and train-only-fit tests pass.
- All four methods are reported with intervals on both splits. If LightGBM does not beat the tuned baseline, do not tune further to force it. Check for leakage and feature bugs, then report the result as it stands and log it in `DECISIONS.md`.
- The model scores 5,000 candidate pairs in under 5 seconds on CPU.
- `model_card.json` is complete and the calibrator loads without pickle.

### Phase 5: Clustering from probabilities, baselines, replay

**Goal:** turn pair probabilities into stories, tune on cluster-level metrics, and compare against all baselines on identical data.

**Tasks**

1. **`cluster.py`.** Accept pair probabilities from any scorer (cosine baseline, or learned and calibrated). Average linkage within connected components, cut at `tau_cluster`. Also expose single and complete linkage for the chaining demonstration.
2. **Tune `tau_cluster`** on validation stories by maximizing B-cubed F1. Do not reuse the pair-level threshold. Persist the value in `settings.yaml` and `model_card.json`.
3. **Common interface.** Each method implements `fit_predict(headlines) -> labels` and runs over identical candidate sets and splits:
   - cosine threshold with average linkage (the Phase 2 baseline)
   - TF-IDF with agglomerative clustering
   - exact or near-exact normalized-title matching (MinHash)
   - HDBSCAN on embeddings
   - LightGBM probabilities with average linkage (the main system)
   - LightGBM probabilities with single linkage (the chaining demo)
4. **Evaluation.** On the group-split test stories and the temporal holdout, report B-cubed precision, recall, and F1, ARI, and V-measure, with story-bootstrap intervals. Stratify by day volume (heavy versus quiet days) and by story size (2, 3-5, 6 or more headlines).
5. **Chaining demo.** Export five concrete cases where single linkage merges distinct stories and average linkage does not, to `docs/figures/chaining_cases.json`.
6. **Replay mode.** `hst replay --start --end --method <name>` runs the pipeline hourly over stored history offline, writing story outputs and stability metrics to a scratch directory. Use it to compare id persistence, cluster-size distribution, and singleton rate across methods on identical data, and to produce the weekly drift experiment later.
7. **Optional:** a greedy pivot correlation-clustering variant, compared with average linkage.
8. **Output.** Write the `cluster_level` block of `metrics.json`.

**Acceptance**

- All methods are evaluated on the same splits and candidate sets, with intervals, in `metrics.json`.
- Replay of 14 days of history completes in under 15 minutes.
- The chaining cases file exists and contains real examples from the gold data.

### Phase 6: Cross-encoder and stacking (offline)

**Goal:** test whether a fine-tuned transformer reading both headlines adds signal beyond engineered features.

**Tasks**

1. **Notebook.** `notebooks/train_cross_encoder.ipynb` for Colab with a GPU. It loads `pairs.jsonl`, the snapshot, and `splits.json`, and fine-tunes a MiniLM-class cross-encoder with binary cross-entropy. Early stopping on validation PR-AUC. Run 3 seeds and report the spread.
2. **Out-of-fold scores.** To stack without leakage, train 5 grouped-fold cross-encoders and use out-of-fold scores for every training pair. Add a test asserting no pair's score came from a model that saw that pair's story. Document the compute cost.
3. **Release.** Provide `scripts/publish_model.sh` (uses `gh release create`) to upload the final weights as a release asset. Do not store weights in git.
4. **`models/cross_encoder.py`.** CPU batch inference loading from the cached release asset. Measure latency per 5,000 pairs. Evaluate dynamic int8 quantization and report any quality loss.
5. **Stack.** Add the cross-encoder score as a LightGBM feature. Recalibrate.
6. **Evaluate** at pair level and cluster level: cross-encoder alone, LightGBM alone, stacked. Include latency, intervals, and the clean-subset results.
7. **Optional stretch:** fine-tune the bi-encoder with MultipleNegativesRankingLoss using hard negatives (same event type, different event). Evaluate as a replacement for the cosine feature.
8. Update `metrics.json` and `model_card.json`.

**HUMAN GATE:** the owner runs the notebook on Colab following the agent's step-by-step instructions, then runs the publish script.

**Acceptance**

- Splits come from `splits.json`, with a leakage test proving no train story appears in test.
- The out-of-fold test passes.
- Results for all three configurations exist with intervals, and the inference cost per run is documented against the 10-minute budget.

### Phase 7: Production inference swap-in

**Goal:** replace the baseline scorer in the scheduled pipeline with the learned model, safely and reversibly.

**Tasks**

1. `pipeline.yml` loads `models/gbm.txt`, `models/calibrator.json`, and (if configured) the cross-encoder from the cached release asset. `settings.yaml: active_model` selects `baseline`, `gbm`, or `stacked`.
2. **Shadow mode.** For 7 days, run both the baseline and the learned model on every scheduled run. Publish the baseline. Log the learned model's clusters to `state/shadow/` and compute agreement (ARI between the two partitions, count of differing stories, id persistence for each).
3. Put a `model_version` string in `stories.json` and `health.json`, and show it in the site header.
4. **Fallback.** If model loading or inference fails, or the scoring stage exceeds 8 minutes, fall back to the baseline for that run and set `fallback: true` in `health.json`.
5. **Input guards.** Drop pairs with missing embeddings, clamp probabilities to \[0, 1\], assert the probability matrix is symmetric.
6. **Golden-file test.** A fixed small headline set must produce fixed expected clusters. Regenerate the expected file only through an explicit script when the model changes.

**HUMAN GATE:** the owner reviews the shadow comparison and approves promotion by changing `active_model`.

**Acceptance**

- A simulated model failure triggers the fallback and the run still deploys.
- Seven days of shadow logs exist.
- Promotion is a one-line settings change and is reversible the same way.
- Total runtime stays under budget.

### Phase 8: Weekly retraining, active learning, drift

**Goal:** a retrain-and-evaluate loop, label-efficient improvement, and drift evidence.

**Tasks**

1. **`weekly.yml`.** On Sundays, rebuild features from the gold set plus new labels, retrain, and evaluate on the fixed test split and the latest week. Write `metrics.json`. The job never changes the active model. It saves the candidate as a workflow artifact and opens a GitHub issue summarizing the metrics (`gh issue create`). The owner promotes manually.
2. **`labeling/select.py`.** Score the last 7 days of candidate pairs. Select 50 with probability closest to 0.5, capped at 3 pairs per story, plus 20 uniformly random pairs. Output a pair-mode file for the labeling tool and tag each pair with how it was sampled.
3. **`labeling/weak.py`.** Generate weak positive pairs from identical normalized titles across outlets, matching wire-credit or URL slug patterns, and syndication signals. Mark them `label_source: weak`. Use them for training only, never in any test set. Run an ablation with and without them, and have the owner audit 50 random weak pairs to measure their precision.
4. **Learning-curve experiment.** Train on 10%, 25%, 50%, and 100% of training stories, adding stories either at random or by uncertainty sampling, with at least 5 repeats. Report PR-AUC against number of labeled pairs with intervals.
5. **Drift experiment (`eval/drift.py`).** Train once on the earliest week. Score each later gold week. Compare with a model retrained on a rolling 2-week window and one retrained on all past data. Also report unsupervised drift signals per week: mean top-neighbor cosine, singleton rate, mean cluster size, centroid shift (cosine between weekly mean embeddings), and out-of-vocabulary rate against the training vocabulary. Write the `drift` block.
6. Add weekly trend lines for the unsupervised signals to the health history.

**HUMAN GATE:** about 15 minutes per week to label the sampled pairs.

**Acceptance**

- `weekly.yml` runs on schedule and on manual dispatch and never modifies the active model.
- The `learning_curve` and `drift` blocks exist with intervals.
- Weak-label precision has been measured and recorded.

### Phase 9: Live audit, Model page, write-up

**Goal:** public evidence and complete documentation.

**Tasks**

1. **Live audit.** Owner-labeled weekly files go to `eval/audit/YYYY-WW.jsonl` on `main`. `run_eval.py --stage live_audit` computes running precision and recall from the random-sampled portion only. The uncertainty-sampled portion is used for calibration checks and is never used for headline accuracy, because it is biased by design. Write the `live_audit` weekly time series with intervals.
2. **`site/model.html`.** Render only from `metrics.json` and `health.json`, with no hard-coded numbers. Sections: method summary and current `model_version`; methods table with PR-AUC and B-cubed F1 with intervals; PR curves; reliability diagram before and after calibration; ablation chart; learning curve; drift by week; live audit series; data health panel (feeds up, items per day, duplicate rate, null-timestamp rate, id persistence, singleton rate); error taxonomy table; a visible limitations box.
3. **Home page polish.** Source-coverage matrix per story, filter by source category, keyboard navigation, readable contrast, reduced-motion support, mobile layout.
4. **README and docs.** `scripts/render_readme_tables.py` fills result tables from `metrics.json`. The README follows the outline in the Project Guide, with placeholders where the owner writes narrative. Add `docs/METHODOLOGY.md` and `docs/DATASHEET.md` (what the gold set contains, how it was built and labeled, known biases and limits).
5. **Cleanup.** Add the owner's chosen LICENSE, remove dead code, confirm pinned dependencies, and verify that a fresh clone follows the README to a working local run.

**HUMAN GATE:** the owner writes the narrative sections and picks the license.

**Acceptance**

- A test greps `site/` and fails if any metric value appears hard-coded.
- A fresh clone reproduces `metrics.json` to within a documented tolerance.
- The Model page renders correctly with an empty or partial `metrics.json` (graceful degradation).

### Phase 10 (stretch): Growth forecasting scoreboard

**Goal:** a predict-then-verify layer with a public track record.

**Tasks**

1. **Target.** For a story with 2-3 sources at time t, will it reach at least 10 sources within 6 hours? Check the historical base rate first. If the positive rate falls outside roughly 5-40%, adjust the threshold and log the decision.
2. **Features at time t.** Current source and headline counts, growth over the last 1 and 3 hours, source-category mix, wire involvement, hour of day, story age, and cluster cohesion (spread of pair probabilities inside the story).
3. **Models.** A growth-rate extrapolation baseline, a LightGBM classifier, and optionally a quantile model for source count at +6 hours. Evaluate by walk-forward backtest with replay mode (train on weeks before w, test on week w).
4. **Prediction log.** Each scheduled run appends `story_id`, time, probability, and `model_version` to `state/forecast_log.jsonl`. A scorer fills outcomes after 6 hours plus a grace period. Handle merges and splits by scoring against the descendant story with the largest member overlap. The log is append-only and predictions are never rewritten.
5. **Scoreboard.** Reliability diagram, Brier score and log loss versus the baseline, rolling 7-day view, and number of scored predictions. Count only predictions that were logged before their outcomes were known.
6. Fill the `forecast` field of story objects.

**Acceptance**

- A test proves the log is append-only and the scorer is idempotent.
- The walk-forward backtest is reported with intervals.
- The scoreboard renders from JSON only.

## Appendix A: Phase report template

End every phase with a report containing: phase and date; what was built; commands to run it; acceptance checklist with pass or fail per item; numbers produced (each with the script and file path that generated it); decisions made (link to `DECISIONS.md`); open questions and risks; what is needed from the owner (the HUMAN GATE, with exact steps); and the proposed next phase.

## Appendix B: Initial `settings.yaml` values

| Key | Value |
| --- | --- |
| `seed` | 1337 |
| `window_hours` | 72 |
| `retention_days` | 60 |
| `blocking_k` | 15 |
| `tau_baseline` | 0.65 (placeholder, tuned in Phase 5) |
| `tau_cluster` | set in Phase 5 |
| `id_match_min_jaccard` | 0.5 |
| `story_death_hours` | 72 |
| `top_stories_published` | 100 |
| `max_headlines_per_story_published` | 12 |
| `runtime_budget_minutes` | 10 |
| `scoring_fallback_minutes` | 8 |
| `active_model` | `baseline` until the Phase 7 promotion |
| `pipeline_cron` | `*/30 * * * *` |

## Appendix C: Pitfall checklist

- **Leakage:** fitting TF-IDF, priors, or thresholds on data that includes test stories; splitting by pair instead of by story; scoring training pairs with a model that saw them (stacking).
- **Asymmetry:** features that change when A and B are swapped.
- **Time:** naive datetimes, feed dates without offsets, future-dated items, comparing local and UTC times.
- **Identity:** URL variants producing duplicate ids; story ids reshuffling between runs.
- **Actions:** cron drift and skipped runs; push conflicts on the `data` branch; caches evicted after a week without access or when the repo cache limit is reached (always handle a cold cache); model downloads from the hub failing (cache them and degrade gracefully).
- **Determinism:** language detection and any random sampling without a fixed seed; multithreaded training that changes results between runs.
- **Repo size:** committing embeddings, model caches, notebook outputs, or weights. Strip notebook outputs before commit.
- **Honesty:** tuning on the test split, reporting only the best seed, reporting uncertainty-sampled labels as accuracy, or hiding a null result.
