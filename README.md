# ReasonHop

Python web server for explicit multi-hop question answering research.

## What this server provides

- FastAPI backend with clean architecture layers
- Explicit reasoning flow: retrieval → reasoning steps → answer synthesis
- Easy extension points for new retrievers and reasoners
- Endpoints for health and question answering

## Project structure

```
app/
	api/                # HTTP layer
		routes/
		dependencies.py
		schemas.py
		router.py
	application/        # use cases
		dto.py
		use_cases.py
	domain/             # core business abstractions
		models.py
		ports.py
	infrastructure/     # concrete implementations
		retrieval/
		reasoning/
	core/               # settings and logging
		config.py
		logging.py
	main.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run server

```bash
uvicorn app.main:app --reload
```

Server URL: `http://127.0.0.1:8000`
Swagger UI: `http://127.0.0.1:8000/docs`

## Baseline experiments

Run all baselines (1, 2, 3) on HotpotQA with shared evaluation metrics:

```bash
python run_baselines.py --split validation --limit 100 --top-k 3 --top-k-sentences 5
```

What this runs:

- Baseline 1: implicit reader over full provided context
- Baseline 2: BM25 top-K retrieval + same extractive reader
- Baseline 3: supporting-fact-naive (sentence scoring + extractive answer)
- Shared evaluation for all baselines: EM, F1, supporting-title recall, supporting-fact recall, average latency

Detailed output is saved to `outputs/baseline_eval.json`.

## Evaluation scripts

### Run on 100 samples (quick check)

```bash
python run_baselines.py --split validation --limit 100 --top-k 3 --top-k-sentences 5 --output outputs/baseline_eval.json
```

### Run on 1000 samples (stronger comparison)

```bash
python run_baselines.py --split validation --limit 1000 --top-k 3 --top-k-sentences 5 --output outputs/baseline_eval_1000.json
```

### Run on full validation split

```bash
python run_baselines.py --split validation --limit 0 --top-k 3 --top-k-sentences 5 --output outputs/baseline_eval_validation_full.json
```

`--limit 0` means no limit (entire split).

### Optional: run on full train split

```bash
python run_baselines.py --split train --limit 0 --top-k 3 --top-k-sentences 5 --output outputs/baseline_eval_train_full.json
```

This can take significantly longer than validation.

## Visualization scripts

Visualize baseline results:

```bash
python visualize_results.py --input outputs/baseline_eval.json --output-dir outputs/charts
```

Visualize 1000-sample run:

```bash
python visualize_results.py --input outputs/baseline_eval_1000.json --output-dir outputs/charts_1000
```

Visualize full-validation run:

```bash
python visualize_results.py --input outputs/baseline_eval_validation_full.json --output-dir outputs/charts_validation_full
```

Each output chart folder includes:

- `quality_overview.png`
- `exact_match.png`
- `f1.png`
- `supporting_title_recall.png`
- `supporting_fact_recall.png`
- `latency_ms.png`

## API

### Health

`GET /api/v1/health`

### Ask question

`POST /api/v1/qa/ask`

Example request:

```json
{
	"question": "Which university is located in the same city as the author of The Old Man and the Sea?",
	"contexts": [
		"The Old Man and the Sea was written by Ernest Hemingway.",
		"Ernest Hemingway lived in Havana, Cuba.",
		"University of Havana is the oldest university in Cuba."
	],
	"top_k": 3
}
```
