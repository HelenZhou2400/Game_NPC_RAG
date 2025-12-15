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
├── universal_game_index/          # Prebuilt FAISS index directory (created beforehand with utils/rag.py)
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

## Running with GCP VM

#### Create a VM instance

- In the Google Cloud console, go to **Compute Engine → VM instances → Create instance**.
- Example settings:

   * OS: Ubuntu 22.04 LTS
   * Machine type: `e2-standard-4` (4 vCPUs, 16 GB RAM)
   * **Boot disk size**:

     * Recommended: **50–100 GB**
     * Hugging Face models and dataset/index caches (e.g., under `~/.cache/huggingface`) can be large; a small disk may fill up quickly.
- Add new **firewall rule** for streamlit port 8501
  - Go to **VPC network → Firewall → Create firewall rule**
  - Example configuration:

     * Name: `allow-streamlit-8501`
     * Targets: Specified target tags, ex: `streamlit-server`
     * Source IPv4 ranges: `0.0.0.0/0` (or restrict to your IP for more security)
     * Protocols and ports: `tcp:8501`
- Under **Network tags** for VM, add firewal rule tag (ex. `streamlit-server`).
- Create the VM instance

## Running with Docker


#### Build and Run with Docker

```bash
docker build -t game-rag:latest -f utils/Dockerfile .

# run scripts
docker run --rm game-rag:latest python3 utils/rag.py
docker run --rm game-rag:latest python3 utils/benchmark.py
```
#### Run Streamlit application

```bash
docker run --rm -p 8501:8501 game-rag:latest \
  python3 -m streamlit run app_streamlit.py \
  --server.address=0.0.0.0 --server.port=8501
```

Then open in a browser:

```
http://localhost:8501
```

Running with GPUs:
```
docker run --rm --gpus all game-rag:latest python3 utils/benchmark.py
```

## Running with GCP VM

#### Create a VM instance

- In the Google Cloud console, go to **Compute Engine → VM instances → Create instance**.
- Example settings:

   * OS: Ubuntu 22.04 LTS
   * Machine type: `e2-standard-4` (4 vCPUs, 16 GB RAM)
   * **Boot disk size**:

     * Recommended: **50–100 GB**
     * Hugging Face models and dataset/index caches (e.g., under `~/.cache/huggingface`) can be large; a small disk may fill up quickly.
- Add new **firewall rule** for streamlit port 8501
  - Go to **VPC network → Firewall → Create firewall rule**
  - Example configuration:

     * Name: `allow-streamlit-8501`
     * Targets: Specified target tags, ex: `streamlit-server`
     * Source IPv4 ranges: `0.0.0.0/0` (or restrict to your IP for more security)
     * Protocols and ports: `tcp:8501`
- Under **Network tags** for VM, add firewal rule tag (ex. `streamlit-server`).
- Create the VM instance


#### SSH into the VM and run with Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io git build-essential

mkdir ~/rag_demo
cd ~/rag_demo

git clone https://github.com/HelenZhou2400/Game_NPC_RAG .

docker build -t game-rag:latest -f utils/Dockerfile .
```

#### Run the Streamlit app

```bash
docker run --rm -p 8501:8501 game-rag:latest \
  python3 -m streamlit run app_streamlit.py \
  --server.address=0.0.0.0 --server.port=8501
```

* `--server.address 0.0.0.0` allows external connections.
* `--server.port 8501` is the port you opened in the firewall rule.

Leave this process running while you use the app.


#### Access the app from your browser

Find the VM’s **External IP** in the VM instances list, then open:

```text
http://<GCP_EXTERNAL_IP>:8501/
```

## Benchmarking and Additional Scripts

The `utils/benchmark.py` script was used to measure performance across different environments, including:

* Local machine
* GCP CPU VM
* HPC GPU compute node

Metrics include latency, throughput, and token-level statistics.

The `utils/rag.py` script contains the original prototype RAG implementation used to generate the FAISS index and test retrieval logic before refactoring it into `rag_core.py`.
