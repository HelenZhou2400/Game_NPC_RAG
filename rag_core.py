# rag_core.py

import os
import time
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# config
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # or your current Qwen model
INDEX_DIR = "universal_game_index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_DEFAULT = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None
vectorstore = None


RAG_ENV = "local"
# RAG_ENV = "gcp cpu vm"



def load_vectorstore() -> FAISS:
    # load FAISS index 
    if not os.path.isdir(INDEX_DIR):
        raise FileNotFoundError(
            f"Index directory '{INDEX_DIR}' not found. "
            f"Need to build your FAISS index first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def init_rag():
    # initialize RAG components
    global vectorstore, model, tokenizer

    if vectorstore is None:
        print("Loading FAISS index...")
        vectorstore = load_vectorstore()
        print("Index loaded.")

    if model is None or tokenizer is None:
        environment = RAG_ENV
        print(
            f"Loading model {MODEL_NAME} on {environment} CPU..."
        )
        model, tokenizer = load_model()
        print("Model loaded.")


def get_answer(
    question: str,
    max_new_tokens: int = 256,
    top_k: int = TOP_K_DEFAULT,
) -> Dict[str, Any]:
    # run RAG cycle and return metrics and runtime info
    global vectorstore, model, tokenizer
    if vectorstore is None or model is None or tokenizer is None:
        init_rag()
        print("RAG not initialized, re-initializing...")

    environment = RAG_ENV
    start_total = time.time()

    # retrieve documents
    t0 = time.time()
    docs = vectorstore.similarity_search(question, k=top_k)
    retrieval_time = time.time() - t0

    context_str = "\n\n".join(
        [f"[Doc {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)]
    )

    # prepare prompt
    messages = [
        {"role": "system", "content": "You are a knowledgeable game librarian NPC. Answer questions using the context provided."},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"}
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # token counting
    input_ids = tokenizer(chat_text, return_tensors="pt").input_ids.to(DEVICE)
    input_tokens = input_ids.shape[-1]

    # generation
    t1 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
        )
    generation_time = time.time() - t1

    gen_ids = generated_ids[0][input_tokens:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    output_tokens = gen_ids.shape[-1]

    total_time = time.time() - start_total

    return {
        "answer": answer,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": total_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "retrieved_docs": docs,
        "environment": environment,
    }
