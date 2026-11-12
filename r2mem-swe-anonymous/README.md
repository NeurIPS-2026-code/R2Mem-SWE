# R²-Mem adaptation to code-search agents

This supplement implements the transfer of R²-Mem's step-level experience learning to repository search and evidence-based tool use in mini-SWE-agent. It includes code-specific rubrics, offline experience construction, online retrieval and injection, and benchmark execution interfaces. The implementation and software checks are provided here; a controlled paired SWE-bench evaluation remains to be completed.

## Method and contents

`r2mem_swe/core.py` extracts visible trajectory prefixes and validates splits, exact evidence citations and scores. `rubrics.md` contains the complete preliminary criteria in English; `prompts.py` supplies model instructions. `pipeline.py` implements prepare → evaluate → learn. `bank.py` builds a local CPU embedding index. `agent.py` retrieves and temporarily injects experiences before the original decision/tool loop. `run_one.py` runs baseline or enhanced tasks. `scripts/` contains a resumable Verified **baseline-only** batch runner and a separate official grading adapter. Tests use synthetic traces/mocks, not reported benchmark results.

Planning evaluates prospective information gathering. Reflection evaluates explicit interpretation of preceding observations. The same reply may contain both, evaluated separately. Four dimensions each receive 0–3; totals ≥10 produce good practices and totals ≤5 corrective lessons. Intermediate/unobservable/out-of-scope steps are not learned. Pure edit mechanics are outside the search rubric. Learner outputs abstract IF–THEN lessons, not task-specific patches.

Module labels separate two logical banks in one file. Only lesson situations are embedded. Cosine retrieval considers up to 10 entries and returns up to 3 per module; there is no learned reranker. Both situation abstractions see the full visible history. Planning queries combine situation and issue; Reflection queries combine situation and latest feedback. Injection is removed from persistent history after each decision; decisions and observations remain. Model weights are unchanged.

The current-task memory is the interaction history. **No separate GAM temporary-memory integration/compression module is implemented.** The additional bank provides cross-task experience. Illustrative, not experimentally learned: “IF an edit command succeeds but verification does not find the change, THEN inspect the actual file and diff before claiming completion.”

## Installation

Prerequisites: Linux, Python 3.11, Git, working Docker, task images and sufficient CPU/RAM/disk, an OpenAI-compatible task model endpoint, an evaluator/learner endpoint, and a local sentence embedding model. Nothing here installs a Docker daemon, starts GPU servers or contains credentials. Run commands from this extracted directory. Use fresh environments/checkouts:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
mkdir -p external
git clone https://github.com/SWE-agent/mini-swe-agent.git external/mini-swe-agent
git -C external/mini-swe-agent checkout 04d809ceab9df28f9adaed044884180159172930
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e external/mini-swe-agent -r requirements.txt
python -m pip check
python -m r2mem_swe preflight --mini-checkout external/mini-swe-agent
```

Direct pins are not a complete transitive lockfile or universal platform guarantee. Save `pip freeze` with real runs. Upstream source/models/data/images are obtained separately under their licenses (THIRD_PARTY.md).

## Inputs and configuration

Edit `configs/swe_qwen25.example.yaml` for your task-model endpoint. Keep the same model and settings for both arms. The additional `configs/swe_verified_baseline.yaml` sets tool/container limits. Set `environment.executable` if using a Docker wrapper. Batch image operations use `DOCKER_EXECUTABLE` (default `docker`); official grading uses the SDK and `DOCKER_HOST`. Ensure they select the same authorized daemon.

Edit `configs/swe_aux_api.example.json` for offline evaluation/learning; credentials go only in the named environment variable. Set `allow_no_key=false` for authenticated remote services. Edit `configs/swe_online_api.example.json` to use the same task model for online situation abstraction. Zero API pricing means no billed API charge, not free compute.

Create `outputs/splits.json` from the split template with disjoint construction, development and evaluation IDs. Template IDs are placeholders. A few previously inspected public tasks are explicitly excluded from held-out evaluation. Check semantic duplicates/repository overlap separately; ID checks alone do not prove independence. Freeze rubrics and bank before evaluation.

Acquire compatible task images separately. Each agent-facing instance JSON contains only:

```json
{"instance_id":"REPLACE_WITH_TASK_ID","problem_statement":"Original issue text","image_name":"benchmark-image@sha256:REPLACE_WITH_DIGEST"}
```

Keep the full official row with reference/test patches separate for grading; never mount it in the agent container. Only the issue is passed as the task. Trajectory filenames must be `<instance_id>.traj.json`.

## Offline construction

Collect each construction task into a fresh output directory:

```bash
python -m r2mem_swe run \
  --mini-checkout external/mini-swe-agent \
  --manifest outputs/splits.json --instance outputs/construction-instance.json \
  --config swebench_backticks.yaml configs/swe_qwen25.example.yaml configs/swe_verified_baseline.yaml \
  --mode baseline --split construction --total-model-calls 120 \
  --output outputs/construction/task-1
python -m r2mem_swe prepare --manifest outputs/splits.json \
  --trajectories outputs/construction/*/*.traj.json --output outputs/records.json
python -m r2mem_swe evaluate --records outputs/records.json \
  --api-config configs/swe_aux_api.example.json --output outputs/evaluations
python -m r2mem_swe learn --records outputs/records.json \
  --evaluations outputs/evaluations --api-config configs/swe_aux_api.example.json \
  --output outputs/learned
python -m r2mem_swe index --bank outputs/learned/bank.json \
  --embedding-model /path/to/local/embedding-model --output outputs/index
```

Evaluation/learning make real API calls. Successful records are resumable only with matching inputs/configuration. Preparation extracts prefixes, not semantic labels. Failed validation is not a zero score. Review learned lessons for abstraction and leakage. Indexing uses local weights without a model download.

## Independent baseline / enhanced comparison

Copy the R2Mem config template to `outputs/r2mem.json`. Set matching `task_id`, `split`, `index_dir=outputs/index`, online `api_config`, fresh `audit_dir`, and `total_model_call_limit=120`. The manifest must match the bank exactly. For a development task:

```bash
python -m r2mem_swe run \
  --mini-checkout external/mini-swe-agent \
  --manifest outputs/splits.json --instance outputs/development-instance.json \
  --config swebench_backticks.yaml configs/swe_qwen25.example.yaml configs/swe_verified_baseline.yaml \
  --mode baseline --split development --total-model-calls 120 \
  --output outputs/development-baseline
python -m r2mem_swe run \
  --mini-checkout external/mini-swe-agent \
  --manifest outputs/splits.json --instance outputs/development-instance.json \
  --config swebench_backticks.yaml configs/swe_qwen25.example.yaml configs/swe_verified_baseline.yaml \
  --mode experience --split development --total-model-calls 120 \
  --r2mem-config outputs/r2mem.json --output outputs/development-experience
```

These are development defaults, not optimized benchmark hyperparameters. Both have 60 agent steps; abstraction calls additionally count toward the enhanced total cap. Equal caps do not mean equal actual tokens/time. Record online auxiliary costs and offline construction costs too. Use independent per-task output/audit paths. Submitted or nonempty patches do not establish success.

## Official grading

Install the pinned harness separately:

```bash
python3.11 -m venv .venv-harness
git clone https://github.com/SWE-bench/SWE-bench.git external/SWE-bench
git -C external/SWE-bench checkout 53448dd9e6c8868e6349b06dca29da80ae6b7187
.venv-harness/bin/python -m pip install -e external/SWE-bench
.venv-harness/bin/python scripts/swe_official_grade.py \
  --checkout external/SWE-bench --instance outputs/official-instance.json \
  --prediction outputs/development-baseline/prediction.json \
  --output outputs/development-baseline-grade
```

Use the matching full official instance. `--gold-check` runs a reference-patch infrastructure positive control, never a model result. The adapter retains official patch application and grading. Containers are limited to 2 CPUs/8 GiB with networking disabled; disclose these restrictions.

## Optional Verified baseline batch / nohup / resume

The batch downloads pinned Verified metadata, runs a deterministic three-task pilot plus a gold control, then permits all 500 baseline tasks. It does not automate paired R²-Mem evaluation. Image IDs/digests are recorded at runtime.

```bash
export MINI_CHECKOUT="$PWD/external/mini-swe-agent"
export HARNESS_CHECKOUT="$PWD/external/SWE-bench"
export HARNESS_PYTHON="$PWD/.venv-harness/bin/python"
# Existing directory on the Docker data filesystem:
export DOCKER_STORAGE_PATH=/var/lib/docker
mkdir -p outputs/logs
nohup .venv/bin/python -u scripts/swe_verified_batch.py pilot \
  --output outputs/verified-baseline > outputs/logs/pilot.log 2>&1 < /dev/null &
```

After successful pilot validation, run with `full` and a separate log. `status` reads progress. `--retry-errors` retries incomplete/infrastructure failures after repair, never completed model failures. Saved predictions are reused; an interruption before saving a prediction may require rerunning that task. Code/config changes require a new output directory. Use fresh output directories for this release.

The image-pull free-space guard is 150 GiB, not a guarantee of total capacity. Only image references pulled by the run are eligible for non-force cleanup; no prune is issued. Exclude pilot/inspected cases from future held-out comparisons. Incomplete runs have no final 500-task score.

## Local checks

```bash
python -m unittest discover -s tests -v
```

Integration tests skip if mini dependencies are absent. See VALIDATION.md for checks actually performed on this archive. Offline tests do not validate end-to-end benchmark results.
