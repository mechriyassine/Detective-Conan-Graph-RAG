# 🕵️ Detective Conan Graph RAG

_"There is always only one truth!" — Shinichi Kudo_

A GraphRAG (Graph Retrieval-Augmented Generation) application that solves crime mysteries using **Knowledge Graphs** and **Google Gemini AI**. Built with Neo4j, Vertex AI, and Streamlit.

![Detective Conan](https://img.shields.io/badge/Detective-Conan-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red?style=for-the-badge)
![Vertex AI](https://img.shields.io/badge/Vertex_AI-Gemini-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph_DB-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)

<p align="center">
  <img src="https://media.tenor.com/XSbD_RbSKfEAAAAd/detective-conan-conan.gif" alt="Detective Conan" width="300"/>
</p>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI                                 │
│              "Mouri Detective Agency Desk"                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   INGEST    │  │  RETRIEVE   │  │   SOLVE     │
│  Evidence   │  │   Clues     │  │   Crime     │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         NEO4J                                   │
│  ┌─────────────┐    ┌─────────────────────────────────────┐    │
│  │ Vector Store│    │     Knowledge Graph                 │    │
│  │ (Embeddings)│    │  (Person)-[:HAS_MOTIVE]->(Person)   │    │
│  │             │    │  (Object)-[:CAUSED_DEATH]->(Person) │    │
│  └─────────────┘    └─────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GOOGLE VERTEX AI                             │
│  ┌─────────────────┐       ┌─────────────────────────────┐     │
│  │ Gemini 2.5 Flash│       │ text-embedding-005          │     │
│  │ (Generation)    │       │ (Vector Embeddings)         │     │
│  └─────────────────┘       └─────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Ingest Evidence** → Gemini extracts entities (suspects, weapons, locations) and relationships from text files
2. **Build Knowledge Graph** → Entities and relationships are stored in Neo4j
3. **Vector Search** → Evidence text is embedded and stored for semantic search
4. **Solve Crime** → Combines graph traversal + vector search + Gemini to answer questions

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- [Neo4j Aura](https://neo4j.com/cloud/aura/) account (free tier available)
- [Google Cloud](https://console.cloud.google.com/) project with Vertex AI enabled

### 1. Clone the Repository

```bash
git clone https://github.com/mechriyassine/Detective-Conan-Graph-RAG.git
cd Detective-Conan-Graph-RAG
```

### 2. Install uv (if not already installed)

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create Virtual Environment

```bash
uv venv
```

### 4. Activate the Virtual Environment

```bash
# Windows (PowerShell)
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 6. Configure Environment Variables

```bash
# Copy the example file
cp .env-example .env

# Edit .env with your credentials
```

**Required variables in `.env`:**

```env
# Google Cloud Config
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_REGION=us-central1

# Neo4j Config
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
```

### 7. Run the Application

```bash
streamlit run graph_rag_vertex_neo4j.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
Detective-Conan-Graph-RAG/
├── graph_rag_vertex_neo4j.py   # Main application
├── data/
│   └── full_case_file.txt      # Crime evidence file
├── requirements.txt            # Python dependencies
├── .env-example                # Environment template
├── .env                        # Your credentials (gitignored)
└── .gitignore
```

---

## 🔧 Usage

1. **Start the app** with `streamlit run graph_rag_vertex_neo4j.py`
2. **Click "🔬 Ingest Evidence"** in the sidebar to process the case files
3. **Ask questions** in the chat, like:
   - "Who killed Chef Firass?"
   - "What was the murder weapon?"
   - "What is Layla's motive?"

---

## 📦 Dependencies

| Package                     | Purpose                           |
| --------------------------- | --------------------------------- |
| `streamlit`                 | Web UI framework                  |
| `neo4j`                     | Graph database driver             |
| `google-genai`              | Google Gemini AI SDK              |
| `google-cloud-aiplatform`   | Vertex AI platform                |
| `langchain-google-vertexai` | LangChain + Vertex AI integration |
| `langchain-community`       | Neo4j vector store                |
| `langchain-core`            | LangChain core utilities          |
| `python-dotenv`             | Environment variable management   |

---

## 🔑 Getting API Keys

### Neo4j Aura (Free)

1. Go to [console.neo4j.io](https://console.neo4j.io/)
2. Create a free instance
3. Copy the connection URI and password

### Google Cloud / Vertex AI

1. Go to [console.cloud.google.com](https://console.cloud.google.com/)
2. Create a new project (or use existing)
3. Enable the **Vertex AI API**
4. Authenticate: `gcloud auth application-default login`

---

## 📜 License

MIT License

---

## 🙏 Acknowledgments

- Inspired by **Detective Conan** (名探偵コナン)
- Built with [LangChain](https://langchain.com/), [Neo4j](https://neo4j.com/), and [Google Vertex AI](https://cloud.google.com/vertex-ai)
