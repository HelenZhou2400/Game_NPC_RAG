# Game Knowledge RAG Demo (Streamlit)

This repository contains a simple Retrieval-Augmented Generation (RAG) demonstration application built using Streamlit.
The system performs retrieval over a FAISS vector index generated from the *Video Game Text Dataset*, and uses a lightweight language model to answer questions about video game stories, mechanics, and universe lore.

The Streamlit interface provides a clean way to test the RAG pipeline locally or on cloud environments such as a GCP CPU VM.



## Dataset Source

This project uses text data from the Video Game Text Dataset, originally generated for *Library of Codexes*.

Video Game Text Dataset
Davis, Megan
Released: 2021-08-27
Repository: [https://github.com/Davis24/video-game-text-dataset](https://github.com/Davis24/video-game-text-dataset)
CFF Version: 1.2.0


## Repository Structure

```
.
├── app_streamlit.py               # Streamlit UI for querying the RAG system
├── rag_core.py                    # Core RAG logic: FAISS retrieval, context construction, LLM generation
├── requirements.txt               # Project dependencies
├── universal_game_index/          # Prebuilt FAISS index directory (created beforehand)
├── utils/
│   ├── rag.py                     # Script originally used for FAISS index creation and local RAG testing
│   └── benchmark.py               # Script used for performance benchmarking across environments
└── README.md
```

The `utils/` folder includes the original development scripts used during preprocessing, index generation, and environment-specific benchmarking.
The Streamlit app uses the refactored `rag_core.py`, which encapsulates the final RAG logic.


## How the RAG System Works

1. **Vector Retrieval**
   The dataset is embedded using a sentence-transformer model and stored in a FAISS index.
   A user query retrieves the top relevant documents from the index.

2. **Context Construction**
   Retrieved documents are merged into a context block passed to the language model.

3. **LLM Generation**
   A small HuggingFace language model (Qwen or similar) produces an answer conditioned on the retrieved context.

4. **Metrics Reporting**
   The Streamlit interface displays:

   * Retrieval time
   * Generation time
   * Total response time
   * Input token count
   * Output token count
   * Retrieved context snippets


## Running Locally

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the Streamlit application

```bash
streamlit run app_streamlit.py
```

Then open in a browser:

```
http://localhost:8501
```


## Benchmarking and Additional Scripts

The `utils/benchmark.py` script was used to measure performance across different environments, including:

* Local machine
* GCP CPU VM
* HPC GPU compute node

Metrics include latency, throughput, and token-level statistics.

The `utils/rag.py` script contains the original prototype RAG implementation used to generate the FAISS index and test retrieval logic before refactoring it into `rag_core.py`.

