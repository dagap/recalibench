# recalibench: design

This document explains how recalibench works and why it is built the way it is. For results and a quick start, see the [README](../README.md).

## Problem

Classification APIs and LLM-based decision models return a probability with every answer. Teams then act on that number: "auto-approve when the model is at least 95% sure, send everything else to a person." That policy is only as good as the probability is honest.

A vendor can calibrate its model on average across its training mix. That doesn't make it calibrated on *your* task, *your* question wording or *your* option order. recalibench measures the gap and closes it with **post-hoc calibration**: a small mapping, fitted on a few thousand labelled cases from your own domain, that corrects the model's probabilities without retraining or fine-tuning it.

## Architecture

The pipeline is fixed code. Everything that varies between projects (datasets, questions, models, calibrators) is a plug-in.

```text
PIPELINE (fixed code)                          PLUG-INS (one file + one config each)

prepare      download, build items,   <------  datasets and tasks
             stratified fit/val/test split       osha_sir, workers_comp; 3 task configs
   |
score        every task x model:      <------  models
             cached, budgeted, batched           jev (hosted API), tfidf_logreg,
   |                                             structured_logreg, mock
calibrate    fit on fit, select on val <-----  calibrators
   |                                             identity, temperature, Platt, beta,
evaluate     test split only,                    isotonic, histogram, vector scaling,
             paired-bootstrap CIs                Dirichlet
   |
experiments  wording, option order, transfer, cross-model
   |
report, app  loop over artifacts/; no task- or model-specific code
```

Each stage writes its outputs next to a fingerprint of its inputs and skips work whose fingerprint hasn't changed, so re-runs are cheap and resumable.

### Plug-ins

There are four plug-in types. Each is a small `typing.Protocol` (`src/recalibench/core/protocols.py`) with a Hydra config group as its registry: a config file's `_target_` names the class. A plug-in can therefore live in this repository or in a separate private package, with its configs passed through `--config-dir`.

| Type | Contract | Registry |
| --- | --- | --- |
| Dataset | `download(raw_dir)` (idempotent, writes a manifest with SHA-256), `load(raw_dir) -> DataFrame` enforcing `schema()` | `conf/dataset/` |
| Task | Config only: dataset, question (binary or choice), the fields that form the model's input, a label rule | `conf/task/` |
| Model | `predict(items, question) -> [Prediction]`, `estimate_cost`, `supports`, pinned `version`; trainable models also `fit` | `conf/model/` |
| Calibrator | `fit(P, y)` / `predict(P)` on the full probability matrix | `conf/calibrator/` |

`core/registry.py` loads every plug-in through the registry and checks it against its protocol. A **contract test** (`tests/integration/test_contract.py`) defines a toy dataset, task, model and calibrator only under `tests/plugins/` and runs them end to end through the real CLI. It also asserts that `src/` never references them, so adding a plug-in can't require pipeline changes.

## Data and tasks

| Dataset | Source | Licence |
| --- | --- | --- |
| OSHA Severe Injury Reports, 2015–2025 | [osha.gov/severe-injury-reports](https://www.osha.gov/severe-injury-reports) (URL and SHA-256 pinned) | US government work, public domain |
| Workers' compensation claims (synthetic) | [OpenML 42876](https://www.openml.org/d/42876) | See the OpenML page |

| Task | Kind | Question | Label |
| --- | --- | --- | --- |
| `osha_amputation` | binary | "Did this incident result in an amputation?" | Amputation count > 0 |
| `osha_event_type` | 7-way choice | The incident's event type | First digit of the OIICS event code |
| `wc_large_loss` | binary | "Will this claim become a large loss (top 10% of cost)?" | Ultimate cost above the 90th percentile, fitted on the fit split only |

The builder (`tasks/builder.py`) deduplicates identical inputs, draws a stratified sample of 3,000 cases per task, and splits it 60/20/20 into fit, validation and test with a fixed seed. Guards against leakage:

- Label columns and anything derived from them (OSHA `Nature*`/`Event*`, the insurer's `InitialCaseEstimate`) can never appear in a model's input. A unit test enforces this for every task config.
- Thresholds such as the 90th percentile are fitted on the fit split only.
- Only `item.state` is sent to a model. The hosted-model layer and its adapters never read the ground-truth label, which a static test enforces. Trainable baselines see labels only when fitting on the fit split.

## Models

### Hosted models: `ApiModel`

`models/base_api.py` is the base class for hosted models. An adapter implements only the wire format: `build_request`, `parse_response`, `count_tokens`, `price`, `headers`, and a `fake_response` used offline. The base class owns everything else:

- **Async requests** with bounded concurrency, timeouts, and retries with jittered backoff on 429/529/5xx (tenacity). Other 4xx responses are fatal.
- **Repeats**: each item can be asked several times, with probabilities averaged and their spread kept.
- **A response cache** (diskcache) keyed on input, question, options, repeat and batching mode, so a re-run never pays twice.
- **Budget control**: a cost estimate before any spend, a run that refuses to start above the cap, and one shared budget across all models and stages. Every request reserves its worst case before it is sent.
- **Batching**: several questions (for example, the phrasings in the wording experiment) and all repeats of an item can share one request.
- **Offline mode**: with `mode=mock` the network call alone is replaced by a simulator, so the adapter's request and parsing code still runs. Mock and live outputs use separate cache and artifact directories and can never mix.

Failed items are excluded and reported; a run fails if more than 2% of a task's items fail.

**Jev** (`models/jev.py`) adapts TypeSafe's System One API, mapping binary questions to its `noul` primitive and choice questions to `choice`. The version is pinned (`jev-1.14.0`, never an alias), and it is part of every cache key and artifact path, so a silent upgrade can't mix versions.

### Cost model, measured

A 61-item live smoke test checked the adapter against the real API before any full run, and calibrated the cost estimator:

- Each request bills about 270 tokens of fixed overhead plus roughly 1 token per 3.9 characters. The estimator adds 300 tokens per request, so it errs high.
- Batching 3 questions about one case into a single request bills 2.5× fewer tokens for yes/no questions and 1.5× fewer for multiple choice. Answers match unbatched ones (mean absolute difference 0.004).
- Batched repeats vary as much as separate requests, so repeats are batched too, at about 2.6× lower cost.

The full live run, 15,000 requests, cost $0.55 against a worst-case estimate of $1.84.

### Local baselines

`tfidf_logreg` (TF-IDF over the text fields plus logistic regression) and `structured_logreg` (the same on the structured claim fields) are trained on the fit split. Their fit-split scores are **out-of-fold** (5-fold cross-fitting), so no calibrator ever learns from in-sample probabilities.

## Calibration

| Binary | Multiclass |
| --- | --- |
| identity, temperature, Platt, beta, isotonic, histogram binning | identity, temperature, vector scaling, Dirichlet (L2-regularised, strength chosen by CV) |

Every compatible calibrator is fitted on the fit split. One is selected on the validation split using the **one-standard-error rule** on paired per-case log-loss differences: candidates within one SE of the best are treated as tied, and a parametric family then beats a flexible one (isotonic, histogram). With a few hundred validation cases, a fixed tolerance let histogram binning "win" on noise. Identity is always a candidate, so "already calibrated" is a possible outcome. The test split is touched only once, by the evaluation.

`selected.json` records the fingerprint of the scores it was fitted on. Every consumer (the evaluation, the app, the live box) refuses a calibrator fitted on different scores or a different model version.

## Evaluation

All metrics are computed on the held-out test split, raw against the selected calibrator:

- **Calibration**: top-label ECE (equal-width and equal-mass bins), class-wise ECE, and smooth ECE (Błasiok & Nakkiran, 2023), which needs no bins and is steadier on small test sets.
- **Quality**: Brier score, log loss, accuracy, and AUROC for binary tasks.
- **Uncertainty**: paired bootstrap (1,000 resamples of test cases) for 95% intervals on each metric and on the before–after difference. Resampling with replacement inflates ECE slightly, so a bootstrap interval can sit just above its point estimate.
- **Selective prediction**, which is the business number. It is reported as risk–coverage curves, AURC, and the share of cases that can be auto-handled. Two policies are measured:
  - **Tuned**: the threshold is chosen on the validation split for a 95% accuracy target and reported on test.
  - **Trust the number**: auto-handle whenever stated confidence is at least 0.95. This shows directly what dishonest confidence costs.

Metrics are cross-checked against scikit-learn in the tests.

### Experiments

| Experiment | Question it answers |
| --- | --- |
| Wording | Does calibration depend on how the question is phrased? Three phrasings: plain, lay, and OSHA's regulatory definition. |
| Option order | Does listing the options in a different order shift the probabilities? This tests TypeSafe's documented first-option tendency. |
| Transfer | Does a calibrator fitted on one task work on another? |
| Cross-model | Every model side by side on each task, raw and calibrated. |

Experiments run their variants through the same score, calibrate and evaluate code as the main matrix, and are stored separately under `artifacts/_experiments/`.

## Outputs

- **`metrics.json`** per task × model is the single source for the report and the app. It holds test metrics with intervals, every candidate calibrator's scores, reliability bins and selective-prediction curves.
- **`reports/report.html`** is one self-contained file with inline SVG figures and no external assets, built by looping over whatever is in `artifacts/`. Adding a dataset or model adds a section with no template change.
- **The Streamlit app** (`app/streamlit_app.py`) is a thin UI over `src/recalibench/app/`:
  - Overview, Reliability, Decision policy (a threshold slider showing the auto-handled share, the error rate and the review workload), and Examples.
  - A live box scores a typed case with the real API and applies the live calibrator, under an app-wide budget cap.
  - It is served on localhost only.

## Engineering

- Python 3.13, managed with uv; Hydra for configuration and plug-in discovery; Typer CLI (`recal`).
- pyright in strict mode, ruff and deptry run on every change. Untyped third-party APIs (scikit-learn, matplotlib) are confined to small adapter modules.
- Over 300 tests run in parallel (pytest-xdist):
  - unit tests for metrics, calibrators, splits, label rules and adapter parsing against recorded responses;
  - integration tests of the API layer against respx mocks (retries, rate limits, timeouts, cache, budget, latency);
  - the plug-in contract test;
  - Streamlit `AppTest` checks of every page.
- Tests cannot open network connections: an autouse fixture blocks sockets.
- Only aggregate results are committed (`metrics.json`, experiment summaries, the HTML report). Raw scores, calibrators, case text, caches and credentials are excluded by `.gitignore`.

## Limitations

- Each test split has 600 cases (597 for event type), so differences under about 0.01 ECE are within noise; the intervals in the report show this.
- Jev returns probabilities rounded to two decimals, which bounds how finely confidence can be resolved.
- The option-order experiment covers one multiple-choice task. Its result (no first-option bias on this task) shouldn't be generalised to other option sets.
- The offline simulator's miscalibration settings are chosen by hand. Mock results demonstrate the pipeline, not any real model.
