# ==========================
#        Imports
# ==========================

import torch

# ==========================
#     Konfigurationen
# ==========================

# Globale Benchmark-Konfiguration für alle Benchmarks
CONFIG = {
    # Liste der zu testenden Language Models (Name von Huggingface)
    "models": ["Qwen/Qwen2.5-3B", "google/gemma-2-2b", "meta-llama/Llama-3.2-3B"],

    # Installationspfad der Language Models
    "cache_dir": "./models_cache",

    # Ausgabepfad der Messergebnisse
    "results_dir": "./results",

    # Gerät: cuda, cpu
    "device": "cuda",

    # Datentyp: torch.float64, torch.float32, torch.float16, torch.bfloat16
    "datatype": torch.float16,

    # interne Implementierung der Attention des Language Models: eager, sdpa, flash_attention_2
    "attention": "sdpa",

    # Test-Prompts für RAM-Messung und Latenzmessung (in diesem Fall entnommen aus den eval-Benchmarks MMLU, HellaSwag, GSM8K, HumanEval und BoolQ)
    "prompts" : [
        "What was GDP per capita in the United States in 1850 when adjusting for inflation and PPP in 2011 prices? About $300, About $3k, About $8k, About $15k",
        "A woman stands at the end of a diving board. She lightly bounces up and down. she",
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "def max_element(l: list): \"\"\"Return maximum element in the list. >>> max_element([1, 2, 3]) 3 >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10]) 123 \"\"\"",
        "Nuclear power in the United States is provided by 99 commercial reactors with a net capacity of 100,350 megawatts (MW)."
    ],

    # Anzahl an Messwiederholungen je Prompt und Modell für RAM-Messung und Latenzmessung
    "repetitions": 5,

    # Maximale Anzahl generierter Tokens für RAM-Messung und Latenzmessung
    "token_limit": 100,

    # Liste der Benchmarks für Antwortqualität-Messungen (mmlu 15min, hellaswag: 35min, gsm8k: 5std, humaneval: 12min (nur auf Linux), boolq: 4min)
    "tasks" : ["mmlu", "hellaswag", "gsm8k", "humaneval", "boolq"],

    # Antwortqualität-Benchmark Konfiguration
    "num_fewshot": 0,
    "batch_size": "auto",
    "datatype_quality": "float16",
    "device_quality": "cuda:0"
}