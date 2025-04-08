# ==========================
#        Imports
# ==========================

# Standardbibliotheken
import csv  # Für CSV-Ausgabe
import gc  # Speicherbereinigung
import io  # String-Ein-/Ausgabe
import os  # Betriebssystem-Funktionen
import time  # Zeitmessungen
import warnings  # Warnungen unterdrücken
from contextlib import redirect_stdout, redirect_stderr  # Ausgabeumleitung
from datetime import datetime  # Zeitstempel erstellen

# Drittanbieter-Bibliotheken
import numpy as np  # Für statistische Berechnungen
import torch  # PyTorch für Modell-Handling
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as transformers_logging  # Transformers

# Lokale Bibliotheken
from system_info import print_system_info # Systeminformationen-Ausgabe

# ==========================
#     Konfigurationen
# ==========================

# Lokale Konfiguration erstellen
local_config = {
    "models" : ["Qwen/Qwen2.5-3B", "google/gemma-2-2b", "meta-llama/Llama-3.2-3B"],
    "prompts" : [
        "What was GDP per capita in the United States in 1850 when adjusting for inflation and PPP in 2011 prices? About $300, About $3k, About $8k, About $15k",
        "A woman stands at the end of a diving board. She lightly bounces up and down. she",
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "def max_element(l: list): \"\"\"Return maximum element in the list. >>> max_element([1, 2, 3]) 3 >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10]) 123 \"\"\"",
        "Nuclear power in the United States is provided by 99 commercial reactors with a net capacity of 100,350 megawatts (MW)."
    ],
    "repetitions" : 2,
    "token_limit" : 100,
    "datatype" : "torch.float16",
    "attention" : "sdpa",
    "device" : "cuda",
    "cache_dir" : "./models_cache",
    "results_dir" : "./results"
}

# Laden der globalen Konfiguration (wenn vorhanden)
try:
    from global_config import CONFIG as config
except ImportError:
    config = local_config

# Setzen der finalen Konfiguration
models = config["models"]
prompts = config["prompts"]
repetitions = config["repetitions"]
token_limit = config["token_limit"]
datatype = config["datatype"]
attention = config["attention"]
device = config["device"]
cache_dir = config["cache_dir"]
results_dir = config["results_dir"]

# Verzeichnisse anlegen, wenn dieses noch nicht existieren
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


# Warnungen und unnötige Protokollierungen unterdrücken
warnings.filterwarnings("ignore", category=UserWarning)
transformers_logging.set_verbosity_error()

# Deaktivieren von torch.compile für Stabilität
if hasattr(torch, 'compile'):
    torch.compile = lambda model, *args, **kwargs: model

# ==========================
#     Hilfsfunktionen
# ==========================

def configuration_info():
    """Gibt Konfigurationsinformationen aus."""
    print("\n--- Konfigurationsinformationen ---")
    print(f"🔁 Wiederholungen pro Messung: {repetitions}")
    print(f"📈 maximales Token-Antwortlimit: {token_limit}")
    print(f"🧮 PyTorch Datentyp: {datatype}")
    print(f"🧠 Attention-Typ: {attention}\n")

def clean_memory():
    """Leert den Speicher zur Vorbereitung auf die nächste Messung."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    time.sleep(2)

def warm_up_model(model_name):
    """Lädt das Modell einmal, um initiale Schwankungen zu minimieren."""
    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            attn_implementation=attention,
            torch_dtype=datatype
        ).to(device)

    del model, tokenizer
    clean_memory()

def measure_cold_start_latency(model_name, prompt):
    """Misst die Latenz inklusive Modell- und Tokenizer-Laden."""
    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
        # CUDA-Events für präzise Zeitmessung
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()

        start_event.record()
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            attn_implementation=attention,
            torch_dtype=datatype
        ).to(device)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        _ = model.generate(**inputs, max_new_tokens=token_limit, pad_token_id=tokenizer.pad_token_id)

        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000  # in Sekunden

        token_count = inputs['input_ids'].numel()

    del model, tokenizer
    clean_memory()

    return round(elapsed_time, 2), token_count

def measure_inference_latency(model_name, prompt):
    """Misst die reine Inferenzlatenz ohne Ladezeiten."""
    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            attn_implementation=attention,
            torch_dtype=datatype
        ).to(device)

        # CUDA-Events für präzise Zeitmessung
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()

        start_event.record()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        _ = model.generate(**inputs, max_new_tokens=token_limit, pad_token_id=tokenizer.pad_token_id)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000  # in Sekunden

        token_count = inputs['input_ids'].numel()

    del model, tokenizer
    clean_memory()

    return round(elapsed_time, 2), token_count

def measure_multiple_latencies(measure_func, model_name, prompt, repetitions):
    """Misst die Latenz mehrmals und gibt Durchschnitt, Standardabweichung und Tokenanzahl zurück."""
    latencies = []
    token_counts = []

    for _ in range(repetitions):
        latency, token_count = measure_func(model_name, prompt)
        latencies.append(latency)
        token_counts.append(token_count)

    avg_latency = round(np.mean(latencies), 2)
    std_latency = round(np.std(latencies), 2)
    avg_tokens = int(np.mean(token_counts))

    return avg_latency, std_latency, avg_tokens

def save_results_to_csv(csv_path, model_name, results_per_prompt, model_summary):
    """Speichert die Einzel- und Durchschnittsergebnisse in einer CSV-Datei."""
    with open(csv_path, "w", encoding="utf-8", newline='') as file:
        writer = csv.writer(file)

        # Kopfzeile
        writer.writerow([
            "Frage Nr.",
            "Prompt",
            "Cold-Start (Ø, s)",
            "Cold-Start (±)",
            "Inferenz (Ø, s)",
            "Inferenz (±)",
            "Tokens (Ø)"
        ])

        # Prompt-Ergebnisse schreiben
        for row in results_per_prompt:
            writer.writerow(row)

        # Leere Zeile zur Trennung
        writer.writerow([])
        writer.writerow([])

        # Zusammenfassung
        writer.writerow([
            "Modell",
            "Cold-Start (Ø, s)",
            "Cold-Start (±)",
            "Inferenz (Ø, s)",
            "Inferenz (±)"
        ])

        writer.writerow([
            model_name,
            round(model_summary["avg_cold"], 2),
            round(model_summary["std_cold"], 2),
            round(model_summary["avg_inf"], 2),
            round(model_summary["std_inf"], 2)
        ])

# ==========================
#         Hauptteil
# ==========================

if __name__ == '__main__':
    # Systeminformationen ausgeben, wenn dies noch nicht passiert ist
    if os.getenv("BENCH_PARENT") != "1":
        print_system_info()
        print("\n" + "=" * 60 + "\n")

    # Art der Analyse anzeigen
    print("--- Latenz-Analyse ---\n")
    configuration_info()
    print("=" * 60)

    for model_name in models:
        # Zeit messen und Startzeit anzeigen
        now = datetime.now()
        start_time = now.strftime("%d.%m.%Y, %H:%M:%S")
        timestamp = now.strftime("%d.%m.%Y_%H-%M-%S")
        print(f"\n🕰️  Startzeit: {start_time}")
        
        print(f"Lade Modell: {model_name}")

        print(f"🔥 Warm-Up für Modell {model_name} ...")
        warm_up_model(model_name)
        print(f"✅ Warm-Up abgeschlossen für Modell {model_name}!\n")
        print("-" * 60)

        cold_start_latencies = []
        inference_latencies = []
        results_per_prompt = []

        for i, prompt in enumerate(prompts, 1):
            # Cold-Start-Latenz mehrfach messen
            cold_avg, cold_std, cold_tokens = measure_multiple_latencies(measure_cold_start_latency, model_name, prompt, repetitions)
            cold_start_latencies.append(cold_avg)

            print(f"\nFrage {i} (Cold-Start):")
            print(f"  🕒 Durchschnittliche Latenz: {cold_avg:.2f}s ± {cold_std:.2f}s")
            print(f"  🔢 Durchschnittliche Tokens: {cold_tokens}")
            print(f"  📦 Batchgröße: 1 \n")

            # Reine Inferenzlatenz mehrfach messen
            inference_avg, inference_std, inference_tokens = measure_multiple_latencies(measure_inference_latency, model_name, prompt, repetitions)
            inference_latencies.append(inference_avg)
            print(f"Frage {i} (Inferenz):")
            print(f"  🕒 Durchschnittliche Latenz: {inference_avg:.2f}s ± {inference_std:.2f}s")
            print(f"  🔢 Durchschnittliche Tokens: {inference_tokens}")
            print(f"  📦 Batchgröße: 1\n")
            print("-" * 60)

            # Ergebnisse für CSV-Datei speichern
            results_per_prompt.append([
                i,
                prompt,
                cold_avg,
                cold_std,
                inference_avg,
                inference_std,
                cold_tokens
            ])

        # Gesamtdurchschnitt und Standardabweichung berechnen
        if cold_start_latencies:
            avg_cold_latency = np.mean(cold_start_latencies)
            std_cold_latency = np.std(cold_start_latencies)
            print(f"\n📊 Durchschnittliche Cold-Start-Latenz für Modell {model_name}: {avg_cold_latency:.2f}s ± {std_cold_latency:.2f}s")

        if inference_latencies:
            avg_inference_latency = np.mean(inference_latencies)
            std_inference_latency = np.std(inference_latencies)
            print(f"📊 Durchschnittliche Inferenz-Latenz für Modell {model_name}: {avg_inference_latency:.2f}s ± {std_inference_latency:.2f}s \n")

        # Ergebnisse für CSV-Datei speichern
        model_summary = {
            "avg_cold": avg_cold_latency,
            "std_cold": std_cold_latency,
            "avg_inf": avg_inference_latency,
            "std_inf": std_inference_latency
        }

        # Speicherpfad für CSV-Datei prüfen und ggf. anlegen
        safe_model_name = model_name.replace('/', '_')
        model_result_dir = os.path.join(results_dir, safe_model_name)
        os.makedirs(model_result_dir, exist_ok=True)

        # CSV-Dateipfad zusammensetzen
        csv_filename = f"latency_{timestamp}.csv"
        csv_path = os.path.join(model_result_dir, csv_filename)

        # CSV-Datei erstellen
        save_results_to_csv(
            csv_path=csv_path,
            model_name=model_name,
            results_per_prompt=results_per_prompt,
            model_summary=model_summary
        )

        print(f"💾 Ergebnisse gespeichert unter: {csv_path} \n")
        print("=" * 60)

    print("\n✅ Latenzmessungen abgeschlossen!")