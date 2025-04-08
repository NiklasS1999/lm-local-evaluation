# ==========================
#        Imports
# ==========================

# Standardbibliotheken
import csv  # Für CSV-Dateiausgabe
import gc  # Speicherbereinigung
import io  # Für StringIO
import os  # Betriebssystem-Funktionen
import warnings  # Warnungen unterdrücken
from contextlib import redirect_stdout, redirect_stderr  # Für das Umleiten von stdout und stderr
from datetime import datetime  # Zeitstempel erstellen

# Drittanbieter-Bibliotheken
import torch  # PyTorch für Modell-Handling
from transformers import AutoModelForCausalLM, logging as transformers_logging  # Transformers

# ==========================
#     Konfigurationen
# ==========================

# Lokale Konfiguration erstellen
local_config = {
    "models" : ["Qwen/Qwen2.5-3B", "google/gemma-2-2b", "meta-llama/Llama-3.2-3B"],
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

def clean_memory():
    """Leert den Speicher zur Vorbereitung auf die nächste Messung."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def get_model_param_size(model_name):
    """Berechnet die Anzahl der Modellparameter in Milliarden und Rohwert."""
    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            attn_implementation=attention,
            torch_dtype=datatype
        ).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        param_size_billion = param_count / 1e9

    del model
    clean_memory()
    return param_size_billion, param_count

def save_results_to_csv(csv_path, model_name, param_size, param_count):
    """Speichert die Modellparametergrößen in eine CSV-Datei."""
    with open(csv_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Modellname", "Parametergröße (Mrd.)", "Anzahl Parameter (roh)"])
        writer.writerow([model_name, f"{param_size:.2f}", param_count])

# ==========================
#         Hauptteil
# ==========================

if __name__ == '__main__':
    print("--- Parametergröße-Analyse ---\n")
    print("=" * 60)

    for model_name in models:
        # Zeit messen und Startzeit anzeigen
        now = datetime.now()
        start_time = now.strftime("%d.%m.%Y, %H:%M:%S")
        timestamp = now.strftime("%d.%m.%Y_%H-%M-%S")
        print(f"\n🕰️  Startzeit: {start_time}")

        print(f"Lade Modell: {model_name}")
        param_size, param_count = get_model_param_size(model_name)
        print(f"📏 Modell-Parametergröße: {param_size:.2f} Milliarden Parameter\n")

        # Speicherpfad für CSV-Datei prüfen und ggf. anlegen
        safe_model_name = model_name.replace('/', '_')
        model_result_dir = os.path.join(results_dir, safe_model_name)
        os.makedirs(model_result_dir, exist_ok=True)

        # CSV-Dateipfad zusammensetzen
        csv_filename = f"parameter_size_{timestamp}.csv"
        csv_path = os.path.join(model_result_dir, csv_filename)

        # CSV-Datei erstellen
        save_results_to_csv(csv_path, model_name, param_size, param_count)

        print(f"💾 Ergebnisse gespeichert unter: {csv_path} \n")
        print("=" * 60)

    print("\n✅ Parametergrößenanalyse abgeschlossen!")