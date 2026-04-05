# Fraud Detection Dashboard

This repository contains a modular fraud detection proof of concept built around multiple transaction datasets and a unified serving layer. It combines dataset-specific preprocessing adapters, pre-trained models, a FastAPI inference service, and a beautiful HTML/JS dashboard for real-time scoring and AI explanations.

The project currently supports:
- **PaySim:** synthetic mobile money transaction fraud detection
- **BAF:** bank account application fraud detection
- **IEEE-CIS:** e-commerce transaction fraud detection
- **FinBank demo:** a new client dataset routed into the existing BAF pipeline through feature-role mapping, without retraining

## Project Highlights
- **Unified Prediction API:** A single backend API processing multiple fraud datasets.
- **Dataset-Specific Preprocessing Adapters:** Translates diverse datasets into shared decision outputs.
- **Pre-Trained Model Artifacts:** Included in the repository ready-to-run.
- **Live Monitoring Dashboard:** Vanilla HTML/JS frontend showing real-time transaction simulation and alerts.
- **ChatGPT-Powered Risk Insights:** (Optional) LLM integration to generate real-time professional explanations for why a transaction was flagged or approved.
- **Feature-Role Mapping Design:** Helps onboard unseen datasets with different column names.

## Repository Structure

```text
fraud_pipeline/
├── backend/
│   ├── adapters/     # Dataset adapters and feature mapping logic
│   ├── api/          # FastAPI app exposing prediction endpoints
│   ├── core/         # Shared preprocessing, decision logic, and AI explanations
│   ├── models/       # Model artifacts (.joblib)
│   └── .env          # Environment variables and API keys
├── frontend/         # Dashboard (index.html) and demo CSV data
├── notebooks/        # Data science pipelines and research
└── requirements.txt  # Project dependencies
```

## How It Works

Each dataset has its own preprocessing flow and trained model assets. The API loads all required artifacts at startup and serves predictions through a common response format that includes:
- Fraud probability
- Decision (`pass`, `review`, or `block`)
- Risk level & Risk factors
- Dataset name & Model type

For the **FinBank demo** dataset, the system maps unfamiliar column names into shared semantic roles, and renames them to the schema expected by the BAF pipeline. This demonstrates the main architectural idea of the project: reuse an existing fraud model for a new client dataset when the underlying business meaning of features is aligned.

## Running the Project

### 1. Configure API Keys (Optional but Recommended)
For AI-generated risk explanations to work on the dashboard, configure your `.env` file inside the `backend/` directory with your OpenAI key:
```env
ENABLE_LLM_EXPLANATIONS="true"
LLM_MODEL="gpt-4o"
LLM_API_KEY="sk-YOUR_API_KEY"
```

### 2. Install Dependencies
From the repository root, install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Start the Server
The FastAPI application serves both the API endpoints and the frontend dashboard automatically. Start the server using Uvicorn:

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Visit the Application
Open your browser and navigate to:
[http://localhost:8000](http://localhost:8000)

The demo includes pre-sampled CSV files for *PaySim, BAF, and FinBank*. The beautiful dashboard replays streaming transactions, scores them sequentially through the backend API, and displays real-time fraud decisions alongside LLM-generated explanations whenever you click on a transaction.
