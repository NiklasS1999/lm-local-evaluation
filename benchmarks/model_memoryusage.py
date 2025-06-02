# ==========================
#        Imports
# ==========================

# Standardbibliotheken
import csv  # CSV-Datei ausgeben
import gc  # Speicherbereinigung
import io  # String-Ein-/Ausgabe
import multiprocessing  # Parallele Prozessverwaltung
import os  # Betriebssystem-Funktionen
import subprocess  # Ausführen von Systembefehlen
import time  # Zeitmessungen
import warnings  # Warnungen unterdrücken
from contextlib import redirect_stdout, redirect_stderr  # Ausgabeumleitung
from datetime import datetime  # Zeitstempel erstellen

# Drittanbieter-Bibliotheken
import numpy as np  # Für statistische Berechnungen
import psutil  # System- und Prozessinformationen
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
os.environ.update({
    "TRANSFORMERS_VERBOSITY": "error",
    "TRANSFORMERS_NO_PROGRESS_BAR": "1",
    "DISABLE_PROGRESS_BAR": "1"
})

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

def measure_process_memory_usage():
    """Misst den aktuellen RAM-Verbrauch des Prozesses."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e6  # in MB

def get_gpu_memory():
    """Misst den aktuellen GPU-Speicherverbrauch."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e6  # in MB

def get_gpu_temperature():
    """Liest die aktuelle GPU-Temperatur aus."""
    try:
        return subprocess.getoutput("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader").strip()
    except:
        return "Unbekannt"
    
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

def ensure_model_is_available(model_name, cache_dir, device):
    """Prüft, ob das Modell vollständig lokal verfügbar ist – lädt es bei Bedarf herunter."""
    try:
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
            AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True).to(device)
        print(f"\n📁 Modell bereits lokal vorhanden: {model_name}")
    except Exception as e:
        print(f"\n⚠️  Modell nicht vollständig oder beschädigt. Es wird heruntergeladen: {model_name}")
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        print(f"✅ Modell erfolgreich heruntergeladen: {model_name}")
    clean_memory()

def measure_additional_metrics(model_name, prompt):
    """Misst zusätzliche Metriken: GPU-Nutzung, GPU-Temperatur, Startzeit."""
    torch.cuda.synchronize()
    clean_memory()

    gpu_mem_before = get_gpu_memory()
    gpu_temp_before = get_gpu_temperature()
    
    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            attn_implementation=attention,
            torch_dtype=datatype
        ).to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    _ = model.generate(**inputs, max_new_tokens=token_limit, pad_token_id=tokenizer.pad_token_id)

    gpu_mem_after = get_gpu_memory()
    gpu_temp_after = get_gpu_temperature()

    torch.cuda.synchronize()

    metrics = {
        'gpu_mem_usage': max(0, gpu_mem_after - gpu_mem_before),
        'gpu_temp_before': gpu_temp_before,
        'gpu_temp_after': gpu_temp_after
    }

    del model, tokenizer
    clean_memory()

    return metrics

def run_multiple_inference_measurements(model_name, prompt, repetitions):
    """Führt mehrere Inferenzdurchläufe mit multiprocessing durch und berechnet Durchschnitt + Standardabweichung."""

    ram_loads = []
    ram_inferences = []
    ram_totals = []
    token_counts = []

    for _ in range(repetitions):
        # Neuen Prozess mit Shared Dictionary starten
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        process = multiprocessing.Process(
            target=run_inference,
            args=(model_name, prompt, return_dict)
        )

        process.start()
        process.join()
        time.sleep(2)

        if ('ram_load_usage' not in return_dict or
            'ram_inference_usage' not in return_dict or
            'ram_total_usage' not in return_dict or
            'token_count' not in return_dict):
            print("⚠️  Ein Durchlauf konnte nicht ausgewertet werden – wird übersprungen.")
            continue

        # Ergebnisse sammeln
        ram_loads.append(return_dict['ram_load_usage'])
        ram_inferences.append(return_dict['ram_inference_usage'])
        ram_totals.append(return_dict['ram_total_usage'])
        token_counts.append(return_dict['token_count'])

    # Falls keine erfolgreichen Durchläufe, abbrechen
    if not ram_totals:
        return None

    # Durchschnitt & Standardabweichung berechnen
    ram_load_avg = round(np.mean(ram_loads), 2)
    ram_load_std = round(np.std(ram_loads), 2)
    ram_inf_avg = round(np.mean(ram_inferences), 2)
    ram_inf_std = round(np.std(ram_inferences), 2)
    ram_total_avg = round(np.mean(ram_totals), 2)
    ram_total_std = round(np.std(ram_totals), 2)
    avg_tokens = int(np.mean(token_counts))

    return {
        "ram_load_avg": ram_load_avg,
        "ram_load_std": ram_load_std,
        "ram_inference_avg": ram_inf_avg,
        "ram_inference_std": ram_inf_std,
        "ram_total_avg": ram_total_avg,
        "ram_total_std": ram_total_std,
        "avg_tokens": avg_tokens
    }

def save_results_to_csv(csv_path, model_name, results_per_prompt, model_summary):
    """Speichert die Einzel- und Durchschnittsergebnisse in einer CSV-Datei."""
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Abschnitt 1: Ergebnisse pro Prompt
        writer.writerow(["Ergebnisse je Prompt"])
        writer.writerow([
            "Modell", "Prompt Nr.", "Prompt",
            "RAM Laden Ø [MB]", "RAM Laden Std. [MB]",
            "RAM Inferenz Ø [MB]", "RAM Inferenz Std. [MB]",
            "RAM Gesamt Ø [MB]", "RAM Gesamt Std. [MB]",
            "GPU Verbrauch [MB]", "Tokens generiert"
        ])

        for row in results_per_prompt:
            writer.writerow([
                model_name, *row
            ])

        # Leere Zeile zur Trennung
        writer.writerow([])
        writer.writerow([])

        # Abschnitt 2: Durchschnittsergebnisse je Modell
        writer.writerow(["Gesamtdurchschnitt pro Modell"])
        writer.writerow([
            "Modell",
            "RAM Laden Ø [MB]", "RAM Laden Std. [MB]",
            "RAM Inferenz Ø [MB]", "RAM Inferenz Std. [MB]",
            "RAM Gesamt Ø [MB]", "RAM Gesamt Std. [MB]",
            "GPU Verbrauch Ø [MB]"
        ])

        writer.writerow([
            model_name,
            f"{model_summary['avg_ram_load']:.2f}", f"{model_summary['std_ram_load']:.2f}",
            f"{model_summary['avg_ram_inf']:.2f}", f"{model_summary['std_ram_inf']:.2f}",
            f"{model_summary['avg_ram_total']:.2f}", f"{model_summary['std_ram_total']:.2f}",
            f"{model_summary['avg_gpu']:.2f}",
        ])

# ==========================
#     Inferenz-Funktion
# ==========================

def run_inference(model_name, prompt, return_dict):
    """Führt die Inferenz durch und misst den RAM-Verbrauch."""
    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
        torch.cuda.synchronize()
        clean_memory()

        # Schritt 1: RAM vor dem Laden messen
        mem_before_load = measure_process_memory_usage()

        # Tokenizer und Modell laden
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            attn_implementation=attention,
            torch_dtype=datatype
        ).to(device)

        # Schritt 2: RAM nach dem Laden messen
        mem_after_load = measure_process_memory_usage()

        # Prompt-Inferenz
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=token_limit, pad_token_id=tokenizer.pad_token_id)

        # Schritt 3: RAM nach der Inferenz messen
        mem_after_inference = measure_process_memory_usage()

        # Ergebnisse berechnen
        ram_load_usage = max(0, mem_after_load - mem_before_load)
        ram_inference_usage = max(0, mem_after_inference - mem_after_load)
        ram_total_usage = max(0, mem_after_inference - mem_before_load)
        num_generated_tokens = outputs.shape[-1] - inputs['input_ids'].shape[-1]

        torch.cuda.synchronize()
        del model, tokenizer, outputs, inputs
        clean_memory()

        return_dict['ram_load_usage'] = ram_load_usage
        return_dict['ram_inference_usage'] = ram_inference_usage
        return_dict['ram_total_usage'] = ram_total_usage
        return_dict['token_count'] = num_generated_tokens

# ==========================
#         Hauptteil
# ==========================

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # Systeminformationen ausgeben, wenn dies noch nicht passiert ist
    if os.getenv("BENCH_PARENT") != "1":
        print_system_info()
        print("\n" + "=" * 60 + "\n")

    # Art der Analyse anzeigen
    print("--- Speichernutzung-Analyse ---\n")
    configuration_info()
    print("=" * 60)

    for model_name in models:
        # Prüfen ob das Modell bereits lokal vorhanden ist
        ensure_model_is_available(model_name, cache_dir, device)
        
        # Zeit messen und Startzeit anzeigen
        now = datetime.now()
        start_time = now.strftime("%d.%m.%Y, %H:%M:%S")
        timestamp = now.strftime("%d.%m.%Y_%H-%M-%S")
        print(f"\n🕰️  Startzeit: {start_time}")

        print(f"Lade Modell: {model_name}")

        print(f"🔥 Warm-Up für Modell {model_name} ...")
        warm_up_model(model_name)
        print(f"✅ Warm-Up abgeschlossen für Modell {model_name}!\n")

        ram_load_usages = []
        ram_inference_usages = []
        ram_total_usages = []
        token_counts = []
        gpu_usages = []
        results_per_prompt = []

        for i, prompt in enumerate(prompts, 1):
            # RAM-Messungen mit Wiederholungen via multiprocessing
            inference_results = run_multiple_inference_measurements(model_name, prompt, repetitions)
            if inference_results is None:
                print(f"\n❌ Fehler bei Modell {model_name}, Frage {i}: Inferenzmessung fehlgeschlagen.\n")
                continue

            avg_load = inference_results['ram_load_avg']
            std_load = inference_results['ram_load_std']
            avg_inf = inference_results['ram_inference_avg']
            std_inf = inference_results['ram_inference_std']
            avg_total = inference_results['ram_total_avg']
            std_total = inference_results['ram_total_std']
            avg_tokens = inference_results['avg_tokens']

            # Zusätzliche Metriken separat messen
            additional_metrics = measure_additional_metrics(model_name, prompt)
            gpu_usage = additional_metrics['gpu_mem_usage']

            # Sammeln für Gesamtstatistik
            ram_load_usages.append(avg_load)
            ram_inference_usages.append(avg_inf)
            ram_total_usages.append(avg_total)
            token_counts.append(avg_tokens)
            gpu_usages.append(gpu_usage)

            # Ausgabe der Durchschnittswerte für den Prompt
            print(f"\nFrage {i}:")
            print(f"  📦 RAM Verbrauch (∅ aus {repetitions} Läufen)")
            print(f"    - Laden       : {avg_load:.2f} MB (± {std_load:.2f})")
            print(f"    - Inferenz    : {avg_inf:.2f} MB  (± {std_inf:.2f})")
            print(f"    - Gesamt      : {avg_total:.2f} MB (± {std_total:.2f})")
            print(f"  🔢 Tokens       : {avg_tokens} tokens")
            print(f"  🎮 GPU Verbrauch: {gpu_usage:.2f} MB")
            print(f"  🌡️  Temperatur   : {additional_metrics['gpu_temp_before']}°C → {additional_metrics['gpu_temp_after']}°C\n")

            # Ergebnisse für CSV-Datei speichern
            results_per_prompt.append([
                i,
                prompt,
                avg_load, std_load,
                avg_inf, std_inf,
                avg_total, std_total,
                gpu_usage,
                avg_tokens
            ])

        # Gesamtauswertung für das Modell
        if ram_load_usages:
            print(f"\n📊 Gesamtdurchschnitt für Modell {model_name}:")
            print(f"  - Modell- & Tokenizer-Laden  : {np.mean(ram_load_usages):.2f} MB (± {np.std(ram_load_usages):.2f})")
            print(f"  - Prompt-Inferenz            : {np.mean(ram_inference_usages):.2f} MB (± {np.std(ram_inference_usages):.2f})")
            print(f"  - Gesamt                     : {np.mean(ram_total_usages):.2f} MB (± {np.std(ram_total_usages):.2f})")
            print(f"  - Tokens ∅                   : {int(np.mean(token_counts))}")
            print(f"  - GPU-Nutzung                : {np.mean(gpu_usages):.2f} MB (± {np.std(gpu_usages):.2f})\n")

        # Ergebnisse für CSV-Datei speichern
        model_summary = {
            "avg_ram_load": np.mean(ram_load_usages),
            "std_ram_load": np.std(ram_load_usages),
            "avg_ram_inf": np.mean(ram_inference_usages),
            "std_ram_inf": np.std(ram_inference_usages),
            "avg_ram_total": np.mean(ram_total_usages),
            "std_ram_total": np.std(ram_total_usages),
            "avg_gpu": np.mean(gpu_usages)
        }

        # Speicherpfad für CSV-Datei prüfen und ggf. anlegen
        safe_model_name = model_name.replace('/', '_')
        model_result_dir = os.path.join(results_dir, safe_model_name)
        os.makedirs(model_result_dir, exist_ok=True)

        # CSV-Dateipfad zusammensetzen
        csv_filename = f"memory_usage_{timestamp}.csv"
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

    print("\n✅ Speichernutzungsmessungen abgeschlossen!")