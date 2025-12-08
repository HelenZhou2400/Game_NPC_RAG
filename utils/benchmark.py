import os
import time
import torch
import csv
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

save_spot = "universal_game_index"
model_name = "Qwen/Qwen3-4B"

# fixed benchmark queries
test_queries = [
    "Who is Geralt?",
    "What is a witcher?",
    "Tell me about the Wild Hunt",
    "What monsters can I find in The Witcher?",
    "How do I fight drowners?",
    "Compare the elves in The Witcher and Dragon Age",
    "Tell me the story of Lara Dorren",
    "What happened at Kaer Morhen?",
    "Who is Jacques de Aldersberg?",
    "Tell me about Geralt's silver sword",
]

print("=" * 60)
print("RAG System Benchmark")
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("=" * 60)

print("\nLoading models...")
start_load = time.time()

# Embeddings + FAISS
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
)

memory = FAISS.load_local(save_spot, embedder, allow_dangerous_deserialization=True)

# LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/hf_models/Qwen3-4B"   # bound scratch path

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)

brain = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True,
)

if device == "cpu":
    brain.to("cpu")

load_time = time.time() - start_load
print(f"Models loaded in {load_time:.2f}s")

OUTPUT_DIR = os.environ.get("RAG_OUTPUT_DIR", ".")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "rag_log.txt")
def append_log(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def build_prompt(query: str):
    """
    Retrieve context from FAISS and build a chat-formatted prompt.
    Returns (prompt_text, embed_time_seconds).
    """
    start_embed = time.time()
    relevant_stuff = memory.similarity_search(query, k=3)
    embed_time = time.time() - start_embed

    context_str = "\n\n".join(
        [f"[{d.metadata.get('game', 'unknown')}] {d.page_content}" for d in relevant_stuff]
    )

    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable game librarian NPC.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {query}",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return text, embed_time

def run_single_query(query: str):
    """Run the single-query benchmark for one question."""
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start_total = time.time()

    text, embed_time = build_prompt(query)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    input_tokens = int((model_inputs.input_ids[0] != tokenizer.pad_token_id).sum().item())

    start_gen = time.time()
    generated_ids = brain.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        do_sample=True,
    )
    gen_time = time.time() - start_gen

    # decode output and log
    decoded_output = tokenizer.decode(
        generated_ids[0][input_tokens:], skip_special_tokens=True
    ).strip()

    append_log("="*80)
    append_log(f"[SINGLE QUERY]")
    append_log(f"Query: {query}")
    append_log(f"Answer:\n{decoded_output}")
    append_log("="*80 + "\n")

    # count generated tokens by length difference
    output_tokens = int(generated_ids[0].shape[0] - input_tokens)

    total_time = time.time() - start_total

    if device == "cuda":
        torch.cuda.synchronize()
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
    else:
        gpu_memory = 0.0

    tps = output_tokens / gen_time if gen_time > 0 else 0.0

    return {
        "query": query,
        "total_time": total_time,
        "embed_time": embed_time,
        "gen_time": gen_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": tps,
        "gpu_memory_gb": gpu_memory,
    }


def run_batch(queries):
    """Run a batched generation to simulate concurrency.

    queries: list[str], length == concurrency level.
    Returns summary stats for the whole batch.
    """
    batch_size = len(queries)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # build prompts + retrieve context
    batch_texts = []
    embed_times = []

    embed_start_all = time.time()
    for q in queries:
        text, embed_time = build_prompt(q)
        batch_texts.append(text)
        embed_times.append(embed_time)
    embed_total_time = time.time() - embed_start_all

    model_inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        truncation=True,
    ).to(device)

    # token stats
    input_lens = [
        int((ids != tokenizer.pad_token_id).sum().item())
        for ids in model_inputs.input_ids
    ]

    # generate for whole batch
    start_gen = time.time()
    generated_ids = brain.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        do_sample=True,
    )
    gen_time = time.time() - start_gen

    if device == "cuda":
        torch.cuda.synchronize()
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
    else:
        gpu_memory = 0.0

    total_output_tokens = 0
    per_query_outputs = []

    for i, q in enumerate(queries):
        seq = generated_ids[i]
        input_len = input_lens[i]
        output_len = int(seq.shape[0] - input_len)
        total_output_tokens += output_len

        decoded_output = tokenizer.decode(
            seq[input_len:], skip_special_tokens=True
        ).strip()

        append_log("="*80)
        append_log(f"[BATCH QUERY]  C={len(queries)}")
        append_log(f"Query {i+1}: {q}")
        append_log(f"Answer:\n{decoded_output}")
        append_log("="*80 + "\n")

        per_query_outputs.append(
            {
                "query": q,
                "embed_time": embed_times[i],
                "input_tokens": input_len,
                "output_tokens": output_len,
            }
        )

    batch_total_time = embed_total_time + gen_time
    batch_tps = total_output_tokens / gen_time if gen_time > 0 else 0.0

    return {
        "queries": queries,
        "per_query": per_query_outputs,
        "batch_total_time": batch_total_time,
        "batch_gen_time": gen_time,
        "batch_tokens": total_output_tokens,
        "batch_tokens_per_second": batch_tps,
        "gpu_memory_gb": gpu_memory,
    }


# =============================================================================================================================
# single-query benchmark for 10 fixed questions
# =============================================================================================================================
results = []

print("\nRunning single-query benchmark...")
for idx, query in enumerate(test_queries):
    print(f"[{idx+1}/{len(test_queries)}] Testing: {query[:50]}...")

    result = run_single_query(query)
    results.append(result)

    print(
        f"   Total: {result['total_time']:.2f}s | "
        f"Gen: {result['gen_time']:.2f}s | "
        f"TPS: {result['tokens_per_second']:.1f}"
    )

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = os.path.join(OUTPUT_DIR,f"benchmark_results_{timestamp}.csv")

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nSingle-query results saved to: {csv_file}")

avg_total = sum(r["total_time"] for r in results) / len(results)
avg_gen = sum(r["gen_time"] for r in results) / len(results)
avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
avg_gpu = sum(r["gpu_memory_gb"] for r in results) / len(results)

print("\n" + "=" * 60)
print("SINGLE-QUERY BENCHMARK SUMMARY")
print("=" * 60)
print(f"Total Queries: {len(results)}")
print(f"Average Total Time: {avg_total:.2f}s")
print(f"Average Generation Time: {avg_gen:.2f}s")
print(f"Average Tokens/Second: {avg_tps:.1f}")
print(f"Average GPU Memory: {avg_gpu:.2f} GB")
print(f"Model Load Time: {load_time:.2f}s")
print("=" * 60)

report_file = os.path.join(OUTPUT_DIR,f"benchmark_report_{timestamp}.txt")
with open(report_file, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("RAG System Performance Report\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Device: {device}\n")
    if device == "cuda":
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n"
        )
    f.write(f"Model: {model_name}\n\n")

    f.write("PERFORMANCE METRICS (Single Query)\n")
    f.write("-" * 60 + "\n")
    f.write(f"Total Queries Tested: {len(results)}\n")
    f.write(f"Model Load Time: {load_time:.2f}s\n")
    f.write(f"Average Total Time per Query: {avg_total:.2f}s\n")
    f.write(
        f"Average Embedding Time: {sum(r['embed_time'] for r in results) / len(results):.3f}s\n"
    )
    f.write(f"Average Generation Time: {avg_gen:.2f}s\n")
    f.write(f"Average Tokens per Second: {avg_tps:.1f}\n")
    f.write(f"Average GPU Memory Usage: {avg_gpu:.2f} GB\n\n")

    f.write("DETAILED RESULTS\n")
    f.write("-" * 60 + "\n")
    for i, r in enumerate(results, 1):
        f.write(f"{i}. {r['query']}\n")
        f.write(
            f"   Total: {r['total_time']:.2f}s | Gen: {r['gen_time']:.2f}s | "
            f"TPS: {r['tokens_per_second']:.1f}\n"
        )
        f.write(
            f"   Tokens: {r['output_tokens']} | GPU: {r['gpu_memory_gb']:.2f}GB\n\n"
        )

print(f"Single-query report saved to: {report_file}")

# =============================================================================================================================
# Multiple query, batch concurrency benchmark
# test multiple concurrency level(# of parallel queries) 
# ===============================================================================================================================

concurrency_levels = [1, 2, 4, 8, 16]
runs_per_level = 3

multi_results = []

print("\nRunning multi-query (concurrency) benchmark...")
for C in concurrency_levels:
    for run_idx in range(runs_per_level):
        # build a batch of C queries
        queries_batch = [
            test_queries[(run_idx * C + j) % len(test_queries)] for j in range(C)
        ]

        stats = run_batch(queries_batch)
        batch_time = stats["batch_total_time"]
        tps = stats["batch_tokens_per_second"]
        gpu_mem = stats["gpu_memory_gb"]

        print(
            f"C={C:2d} | run={run_idx+1} | "
            f"time={batch_time:.2f}s | TPS={tps:.1f} | GPU={gpu_mem:.2f}GB"
        )

        multi_results.append(
            {
                "concurrency": C,
                "run_idx": run_idx,
                "batch_total_time": batch_time,
                "batch_gen_time": stats["batch_gen_time"],
                "batch_tokens": stats["batch_tokens"],
                "batch_tokens_per_second": tps,
                "gpu_memory_gb": gpu_mem,
            }
        )

multi_csv_file = os.path.join(OUTPUT_DIR, f"benchmark_multi_{timestamp}.csv")
with open(multi_csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "concurrency",
            "run_idx",
            "batch_total_time",
            "batch_gen_time",
            "batch_tokens",
            "batch_tokens_per_second",
            "gpu_memory_gb",
        ],
    )
    writer.writeheader()
    writer.writerows(multi_results)

print(f"Multi-query benchmark saved to: {multi_csv_file}")

by_concurrency = {}
for r in multi_results:
    C = r["concurrency"]
    by_concurrency.setdefault(C, []).append(r)

with open(report_file, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("CONCURRENCY BENCHMARK (Batch Inference)\n")
    f.write("=" * 60 + "\n\n")

    f.write("SUMMARY BY CONCURRENCY LEVEL\n")
    f.write("-" * 60 + "\n")

    for C in sorted(by_concurrency.keys()):
        runs = by_concurrency[C]
        avg_batch_time = sum(x["batch_total_time"] for x in runs) / len(runs)
        avg_gen_time = sum(x["batch_gen_time"] for x in runs) / len(runs)
        avg_tokens = sum(x["batch_tokens"] for x in runs) / len(runs)
        avg_tps = sum(x["batch_tokens_per_second"] for x in runs) / len(runs)
        avg_gpu = sum(x["gpu_memory_gb"] for x in runs) / len(runs)

        f.write(f"Concurrency: {C}\n")
        f.write(
            f"   Avg Batch Total Time: {avg_batch_time:.2f}s\n"
            f"   Avg Generation Time:   {avg_gen_time:.2f}s\n"
            f"   Avg Batch Tokens:      {avg_tokens:.1f}\n"
            f"   Avg Tokens/Second:     {avg_tps:.1f}\n"
            f"   Avg GPU Memory:        {avg_gpu:.2f} GB\n\n"
        )

    f.write("DETAILED RUNS\n")
    f.write("-" * 60 + "\n")
    for r in multi_results:
        C = r["concurrency"]
        run_idx = r["run_idx"]
        f.write(
            f"C={C:2d} | run={run_idx+1} | "
            f"batch_total_time={r['batch_total_time']:.2f}s | "
            f"gen_time={r['batch_gen_time']:.2f}s | "
            f"tokens={r['batch_tokens']} | "
            f"TPS={r['batch_tokens_per_second']:.1f} | "
            f"GPU={r['gpu_memory_gb']:.2f} GB\n"
        )

print(f"Concurrency benchmark appended to report: {report_file}")