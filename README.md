# Fraud Detection Pipeline

This repository contains a modular fraud detection proof of concept built around multiple transaction datasets and a unified serving layer. It combines dataset-specific preprocessing adapters, pre-trained models, a FastAPI inference service, and a Streamlit demo interface for real-time scoring.

The project currently supports:

- PaySim: synthetic mobile money transaction fraud detection
- BAF: bank account application fraud detection
- IEEE-CIS: e-commerce transaction fraud detection
- FinBank demo: a new client dataset routed into the existing BAF pipeline through feature-role mapping, without retraining

## Project Highlights

- Unified prediction API for multiple fraud datasets
- Dataset-specific preprocessing adapters with shared decision outputs
- Pre-trained model artifacts included in the repository
- Streamlit frontend for live transaction simulation and alert monitoring
- Feature-role mapping design that helps onboard unseen datasets with different column names

## Repository Structure

```text
fraud_pipeline/
├── adapters/     # Dataset adapters and feature mapping logic
├── api/          # FastAPI app exposing prediction endpoints
├── core/         # Shared preprocessing, decision, evaluation, and feature-role logic
├── frontend/     # Streamlit demo app and sample CSV data
├── models/       # Serialized model and preprocessing artifacts
└── experiments/  # Reserved space for experiments
```

## How It Works

Each dataset has its own preprocessing flow and trained model assets. The API loads all required artifacts at startup and serves predictions through a common response format that includes:

- fraud probability
- decision (`pass`, `review`, or `block`)
- risk level
- risk factors
- dataset name
- model type

For standard datasets such as PaySim, BAF, and IEEE-CIS, requests are routed to their corresponding adapters and models.

For the FinBank demo dataset, the system maps unfamiliar column names into shared semantic roles and then renames them to the feature schema expected by the BAF pipeline. This demonstrates the main architectural idea of the project: reuse an existing fraud model for a new client dataset when the underlying business meaning of features is aligned.

## Running the API

From the repository root:

```bash
uvicorn api.main:app --reload
```

The API is designed to expose endpoints such as:

- `GET /health`
- `GET /datasets`
- `POST /predict/{dataset_type}`

Example dataset types include `paysim`, `baf`, and `ieee`.

## Running the Frontend Demo

In a separate terminal, start the Streamlit application:

```bash
streamlit run frontend/app.py
```

The frontend expects the FastAPI backend to be running at `http://localhost:8000`.

The demo includes pre-sampled CSV files for:

- PaySim
- BAF
- FinBank

The interface replays transactions, calls the backend API, and shows real-time fraud decisions and alert summaries.

## Main Components

### `api/main.py`

Loads models at startup and exposes a unified inference API.

### `adapters/`

Transforms raw dataset-specific inputs into model-ready features. The `finbank_adapter.py` file demonstrates how a previously unseen dataset can be aligned to the BAF schema.

### `core/feature_roles.py`

Defines the semantic feature-role framework used to compare and align features across datasets. Shared roles include temporal, monetary, and transaction signals, while richer datasets add roles such as identity, velocity, device, and stability.

### `models/`

Contains serialized model files and preprocessing artifacts used directly by the API.

## Notes

- Model artifacts are committed to the repository so the API can run immediately after dependencies are installed.
- This repository is structured as a working proof of concept rather than a production deployment package.
- The frontend and API assume a local development workflow.

## Suggested Next Improvements

- Add a `requirements.txt` or `pyproject.toml` for reproducible setup
- Add example request payloads for each dataset
- Add unit tests for adapters and API routes
- Add a short architecture diagram for the end-to-end scoring flow
