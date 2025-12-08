import os
import json
import glob
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

raw_data = "./video-game-text-dataset-master/series"
save_spot = "universal_game_index"
model_name = "Qwen/Qwen3-4B"

print("\nStep 1: Loading embedding model...")
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': device}
)

if os.path.exists(save_spot):
    print("Found existing index, loading...")
    memory = FAISS.load_local(save_spot, embedder, allow_dangerous_deserialization=True)
    print(f"Loaded {memory.index.ntotal} documents")
else:
    print("Building new index from game files...")
    docs = []
    folders = [f.path for f in os.scandir(raw_data) if f.is_dir()]
    
    print(f"Found {len(folders)} game folders")
    
    for idx, folder in enumerate(folders):
        game = os.path.basename(folder)
        files = glob.glob(os.path.join(folder, "*.json"))
        print(f"[{idx+1}/{len(folders)}] Processing {game}: {len(files)} files")
        
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as opened_file:
                    content = json.load(opened_file)
                    
                    if 'codexes' in content:
                        for item in content['codexes']:
                            text = item.get('text', '')
                            title = item.get('title', 'Unknown')
                            
                            if text and len(text) > 50:
                                doc_text = f"Game: {game}\nTitle: {title}\n\n{text}"
                                docs.append(Document(
                                    page_content=doc_text,
                                    metadata={"game": game, "source": title}
                                ))
            except Exception as e:
                continue
    
    print(f"\nCollected {len(docs)} documents")
    if len(docs) == 0:
        print("ERROR: No documents found!")
        exit(1)
        
    print("Building FAISS index...")
    memory = FAISS.from_documents(docs, embedder)
    memory.save_local(save_spot)
    print("Saved to disk")

print(f"\nStep 2: Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
brain = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("Ready!\n")
print("="*60)
print("Game Knowledge NPC - Powered by Qwen3-4B")
print("Type 'exit' or 'quit' to leave")
print("="*60)

while True:
    try:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        print("Searching...")
        relevant_stuff = memory.similarity_search(user_input, k=3)
        
        games_found = list(set([d.metadata['game'] for d in relevant_stuff]))
        print(f"Found info from: {', '.join(games_found)}")
        
        context_str = "\n\n".join([f"[{d.metadata['game']}] {d.page_content}" for d in relevant_stuff])

        messages = [
            {"role": "system", "content": "You are a knowledgeable game librarian NPC. Answer questions using the context provided."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_input}"}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        print("Thinking...")
        generated_ids = brain.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            do_sample=True
        )
        
        response = tokenizer.decode(
            generated_ids[0][len(model_inputs.input_ids[0]):],
            skip_special_tokens=True
        ).strip()
        
        print(f"\nNPC: {response}")
        
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
        break
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        continue