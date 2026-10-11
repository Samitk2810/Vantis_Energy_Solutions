# Vantis Energy Solutions

An AI-powered rooftop solar information and lead-qualification experience for homes and small businesses in selected North Indian cities. Ask about solar options, share project requirements, or get help with an existing system.

**Live site:** [vantis-energy-solutions.onrender.com](https://vantis-energy-solutions.onrender.com)

## Explore

| Page | Description |
| --- | --- |
| [Solar assistant](https://vantis-energy-solutions.onrender.com/) | Explore solar, ask questions, and check service locations. |
| [Lead dashboard](https://vantis-energy-solutions.onrender.com/dashboard) | Review enquiries, lead scores, conversation details, and support requests. Requires the dashboard API key. |
| [WhatsApp demo](https://vantis-energy-solutions.onrender.com/whatsapp) | Try the assistant through a simulated WhatsApp-style conversation. |
| [API documentation](https://vantis-energy-solutions.onrender.com/docs) | Browse the FastAPI endpoints. |

## Features

- Conversational assistant for rooftop solar questions and project enquiries.
- Retrieval-augmented answers based on Vantis information.
- Lead qualification and scoring, with persistent conversation sessions.
- Lead dashboard with status filtering, city activity, support requests, and conversation transcripts.
- WhatsApp webhook demo.
- City availability information for selected locations in Uttar Pradesh, Delhi NCR, Punjab, Haryana, and Rajasthan.
- Responsive frontend for desktop and mobile.

## Technology

- **API:** Python, FastAPI, Pydantic
- **Assistant:** LangGraph, LangChain, Groq
- **Knowledge retrieval:** ChromaDB, Jina embeddings
- **Persistence:** SQLAlchemy and SQLite by default; configurable with `DATABASE_URL`
- **Frontend:** HTML, CSS, and vanilla JavaScript

## Run locally

### Requirements

- Python 3.10 or newer
- API credentials for Groq and Jina

### Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
JINA_API_KEY=your_jina_api_key
DASHBOARD_API_KEY=replace_with_a_long_random_secret
# Optional; defaults to a local SQLite database.
# DATABASE_URL=sqlite:///data/vantis.db
```

Start the development server:

```bash
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The API docs are available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

> Keep API keys private. Do not commit `.env` or real credentials. Set `DASHBOARD_API_KEY` to a unique secret before exposing the dashboard.

## API overview

| Method | Endpoint | Purpose | Access |
| --- | --- | --- | --- |
| `GET` | `/health` | Check database, vector store, and LLM configuration. | Public |
| `POST` | `/chat` | Send a message to the assistant. | Public |
| `POST` | `/webhook/whatsapp` | Send a simulated inbound WhatsApp message. | Public |
| `POST` | `/callback` | Request a callback for an existing lead. | Public |
| `GET` | `/leads` | List leads; supports status, city, limit, and offset filters. | Dashboard API key |
| `POST` | `/leads` | Create a lead. | Dashboard API key |
| `GET` | `/leads/{lead_id}` | Get a lead, callbacks, and conversation transcript. | Dashboard API key |
| `PATCH` | `/leads/{lead_id}` | Update lead details. | Dashboard API key |
| `GET` | `/stats` | Get lead and support statistics. | Dashboard API key |
| `GET` | `/support-requests` | List support requests. | Dashboard API key |

Dashboard endpoints use the `X-API-Key` request header. Interactive request and response schemas are available in the FastAPI docs.

### Chat request example

```json
{
  "message": "Do you install rooftop solar in Jaipur?"
}
```

The response includes a `session_id`; send it with later messages to continue the same conversation:

```json
{
  "session_id": "your-session-id",
  "message": "What size system might I need?"
}
```

## Project structure

```text
.
├── agents/       # Assistant graph and conversation agents
├── data/         # Vantis knowledge-base PDF and local runtime data
├── db/           # Database models and repository
├── frontend/     # Customer chat, dashboard, and WhatsApp demo pages
├── rag/          # Knowledge retrieval and vector store
├── scoring/      # Lead scoring
├── tools/        # Validation and service-area helpers
├── main.py       # FastAPI application and route definitions
└── requirements.txt
```

## Notes

- The assistant provides information, not a final system recommendation or guaranteed price, savings, subsidy, warranty, or installation date.
- Service availability is limited to selected cities and should be verified for each project.
- The frontend is configured for the deployed API. For a different backend, update the `API_BASE` value in the HTML files under `frontend/`.
- The deployed service may take a moment to respond after inactivity.
