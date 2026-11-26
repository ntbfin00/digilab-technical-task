# Surrogate Modelling Service — Quick Start

A FastAPI service for background surrogate-model training with CPU PyTorch and
Weights & Biases (W&B) tracking. Run the commands below from the repository root.

## 1. Starting the service

### Local setup

Requires **Python 3.13 or newer** and a compatible PyTorch wheel.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

Install CPU-only PyTorch first so the project installation reuses that build.
The development extra includes the tools required to run the tests.

To enable training, configure a W&B project and entity you can write to, together
with your API key. Replace the placeholder values:

```sh
export WANDB_PROJECT=surrogates
export WANDB_ENTITY=your-team
export WANDB_API_KEY=your-key
export WANDB_MODE=online
python -m app
```

The service listens on **http://localhost:8000**. Interactive API documentation
is available at **http://localhost:8000/docs**.

For development with automatic reload:

```sh
python -m app --reload
```

The health endpoint works without W&B configuration. Training routes return
`503` if either `WANDB_PROJECT` or `WANDB_ENTITY` is missing. Training jobs require
valid credentials, network access, and online W&B mode; offline and disabled modes
are rejected.

Use one server process: live jobs and their datasets are held in process memory.
Restarting the service does not resume queued or running jobs. Shutdown waits for
accepted training jobs and artifact uploads to finish.

### Docker

Build the image, then forward the W&B variables configured above:

```sh
docker build -t surrogate-modelling-service .
docker run --rm --name surrogate-modelling-service -p 8000:8000 \
  -e WANDB_PROJECT -e WANDB_ENTITY -e WANDB_API_KEY -e WANDB_MODE \
  surrogate-modelling-service
```

Stop the container from another terminal with:

```sh
docker stop surrogate-modelling-service
```

## 2. Running the tests

After installing the development dependencies and activating the virtual
environment, run the default test suite:

```sh
python -m pytest tests -q
```

Run the opt-in component test separately. It exercises training, prediction, and
checkpoint restoration and is excluded from default test discovery:

```sh
python -m pytest tests/component_surrogate_training.py -q
```

The default suite can also be run with the standard-library test runner:

```sh
python -m unittest discover -s tests -v
```

Tests perform real CPU training but mock W&B calls, so they require no W&B
credentials or network access.

## 3. Example requests

Start the service before running these examples. Training submissions require
the W&B configuration described above.

### Check service health

```sh
curl http://localhost:8000/
```

Expected response:

```json
{"status":"running"}
```

### Submit training data as JSON

```sh
curl -i http://localhost:8000/training-runs \
  -H 'Content-Type: application/json' \
  -d '{
    "input_features": [[1, 2], [3, 4], [5, 6]],
    "target": [[2], [4], [6]],
    "input_feature_labels": ["temperature", "density"],
    "config": {"epochs": 5}
  }'
```

Use rectangular arrays of finite real numbers with matching sample counts and
at least two samples. Provide one string label per input feature. Booleans and
numeric strings are not accepted as numerical values.

The optional `config` object supports these defaults:

| Field | Default |
| --- | --- |
| `epochs` | `100` |
| `batch_size` | `32` |
| `learning_rate` | `0.001` |
| `hidden_size` | `32` |
| `validation_fraction` | `0.2` |
| `seed` | `42` (also accepts `null`) |

### Upload an NPZ dataset

Create a small archive using the installed NumPy dependency:

```sh
python - <<'PY'
import numpy as np

np.savez_compressed(
    "training.npz",
    input_features=np.array([[1., 2.], [3., 4.], [5., 6.]]),
    target=np.array([[2.], [4.], [6.]]),
    input_feature_labels=np.array(["temperature", "density"], dtype="U"),
)
PY
```

Upload it with an optional JSON configuration in the multipart `config` field:

```sh
curl -i http://localhost:8000/training-runs/upload \
  -F 'file=@training.npz' \
  -F 'config={"epochs":5}'
```

The archive must contain exactly `input_features`, `target`, and
`input_feature_labels`. Labels must be a one-dimensional NumPy Unicode array;
object arrays and pickle are not supported. Default decoder limits are 32 MiB
compressed and 128 MiB total expanded content.

### Poll training progress

Both submission endpoints return **HTTP 202 Accepted**, a progress JSON body
containing `model_id` and `dataset_id`, and a `Location` header with the absolute
polling URL. Acceptance confirms validation and queueing; training and artifact
upload continue in the background.

Copy the `Location` URL into a GET request, or replace `MODEL_UUID` below with the
returned `model_id`:

```sh
curl http://localhost:8000/training-runs/MODEL_UUID
```

Progress includes `status`, `completed_steps`, `total_steps`, `step_unit`,
training/validation losses, `best_validation_loss`, `message`, `run_url`, and
`artifact_reference`. Losses and tracking references may initially be `null`.

Poll until `status` becomes **`completed`** or **`failed`**. Intermediate states
are `queued`, `running`, and `saving`. Completion includes successful model
artifact upload and W&B run finalization. For failures, inspect `message`.

Common error responses are `422` for invalid data, configuration, or UUIDs;
`413` for oversized archives; `404` for a confirmed missing historical run; and
`503` for unavailable configuration or W&B/backend access.

# Decision making for the task

I prioritised a working, tested path from data ingestion through model training,
including the API, progress reporting, and model storage. Within the available
time, I completed these stages rather than providing incomplete implementations
of the entire pipeline. Dedicated test-set evaluation and MCMC-based inference
remain future work. The trained model already supports prediction and checkpoint
restoration in Python, providing a foundation for both stages.

Docker packaging, console logging, and an API health check were established early
so that the application could be run and inspected throughout development. I then
built the core, service, and API layers incrementally, adding tests at each stage.

## Architecture and trade-offs

I separated responsibilities to keep the implementation reusable:

- **Core:** data structures, the abstract `SurrogateTrainer` contract, and the
  concrete CNN backend. Training is synchronous here and independent of HTTP,
  file storage, and W&B.
- **Services:** dataset preparation, NPZ decoding, identifiers, background jobs,
  progress snapshots, and W&B integration.
- **API:** request validation, transport conversion, response schemas, and HTTP
  error handling. Both submission formats share the prepared-dataset training path.

The abstract trainer interface allows another surrogate backend to implement the
same training, prediction, progress, and checkpoint contracts. The current API
selects the CNN backend; exposing a choice of backends would be a separate extension.

For an initial CPU service, an in-process worker keeps execution straightforward.
W&B preserves synchronised metrics and artifacts,
but does not make the local queue durable. Production training currently
requires online W&B access.

## Reproducibility and verification

The default seed controls the dataset split, model initialisation, and batch
shuffling. Configuration and seed are recorded with each run.

Tests cover data validation, real CPU optimisation, seeded behaviour, checkpoint
restoration, threaded progress, failure handling, and HTTP contracts. W&B is mocked
in automated tests. The quick-start JSON and NPZ examples were also exercised with
real `curl` requests against a local server: both completed five training epochs
and produced checkpoints with W&B replaced by test doubles. This verifies the local
request-to-training path, not live W&B authentication or artifact upload.

Console logs include application lifecycle events and per-epoch training and
validation losses. Further observability work would add model IDs consistently to
service logs and monitor queue depth, job duration, and tracking failures.

## How I would extend the pipeline

### Independent model evaluation

I would add `app/core/evaluation.py` to compute metrics such as MSE and MAE on a
separate labelled test dataset. This would be distinct from the validation metrics
already used to select training weights. The service would retrieve the model
checkpoint, verify feature order and target compatibility, and store an evaluation
summary linked to the model ID. POST and GET endpoints would submit evaluation
requests and retrieve their results.

### Inference and parameter estimation

For MCMC-based inference, I would define an abstract sampler interface with
interchangeable implementations. A restored surrogate would supply predictions
to an explicitly defined likelihood, with configurable observations, noise, and
priors. The observed data and covariance would be supplied through an POST endpoint.
Each run would record the exact model artifact version, sampler settings,
seed, and data references. I would assess convergence and effective sample size
and expose posterior summaries and samples through the service/API layers.

### Further improvements

- Input feature selection with the selected schema preserved in checkpoints.
- An independent test split alongside the existing train/validation split.
- Hyperparameter optimisation with Optuna and tracked trial results.
- Surrogate-error estimation and uncertainty propagation.
- Non-uniform priors as part of the inference configuration.
