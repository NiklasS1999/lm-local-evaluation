# ==========================
#        Imports
# ==========================

# Standardbibliotheken
import io  # Für StringIO
import json  # Zum Speichern der Ergebnisse im JSON-Format
import os  # Für Dateipfade und Verzeichnisoperationen
from contextlib import redirect_stdout, redirect_stderr  # Für das Umleiten von stdout und stderr
from datetime import datetime  # Zeitstempel erstellen

# Drittanbieter-Bibliotheken
import datasets  # Umgebungsvariable für ein dataset setzen
from transformers import AutoModelForCausalLM, AutoTokenizer # Transformers

# Lokale Bibliotheken
import lm_eval  # Evaluierungs-Framework für Sprachmodelle
from lm_eval.models.huggingface import HFLM  # Hugging Face Modell-Ladefunktion für lm_eval
from system_info import print_system_info # Systeminformationen-Ausgabe

# ==========================
#     Konfigurationen
# ==========================

# Lokale Konfiguration erstellen
local_config = {
    "models" : ["Qwen/Qwen2.5-3B", "google/gemma-2-2b", "meta-llama/Llama-3.2-3B"],
    "tasks" : ["mmlu", "hellaswag", "gsm8k", "humaneval", "boolq"],
    "num_fewshot" : 0,
    "batch_size" : "auto",
    "datatype_quality" : "float16",
    "device_quality" : "cuda:0",
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
tasks = config["tasks"]
num_fewshot = config["num_fewshot"]
batch_size = config["batch_size"]
datatype = config["datatype_quality"]
device = config["device_quality"]
cache_dir = config["cache_dir"]
results_dir = config["results_dir"]

# Verzeichnisse anlegen, wenn dieses noch nicht existieren
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# ==========================
#   Umgebungsvariablen
# ==========================

# Umgebungsvariablen zum Ausführen mancher Benchmarks setzen
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

# ==========================
#     Hilfsfunktionen
# ==========================

def get_local_model_path(model_name):
    """Bestimmt den lokalen Pfad eines Modells im Cache-Verzeichnis."""
    model_folder_name = f"models--{model_name.replace('/', '--')}"
    local_path = os.path.join(cache_dir, model_folder_name)

    # Überprüfen, ob das Modellverzeichnis existiert
    if os.path.exists(local_path):
        snapshots_path = os.path.join(local_path, "snapshots")

        # Falls Snapshots existieren, wähle den neuesten
        if os.path.exists(snapshots_path) and os.listdir(snapshots_path):
            snapshot_folders = sorted(os.listdir(snapshots_path))  # Sortiere Snapshots
            snapshot_path = os.path.join(snapshots_path, snapshot_folders[-1])  # Neuester Snapshot
            return snapshot_path

    # Falls kein Modell vorhanden ist, gebe None zurück
    return None

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

def serialize_results(obj):
    """Konvertiert nicht-serialisierbare Objekte in Strings, damit sie als JSON gespeichert werden können."""
    if isinstance(obj, dict):
        return {k: serialize_results(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_results(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_results(v) for v in obj)
    elif hasattr(obj, '__dict__'):  # Falls es sich um ein komplexes Objekt handelt
        return str(obj)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)  # Fallback für unbekannte Objekte

def configuration_info():
    """Gibt Konfigurationsinformationen aus."""
    print("\n--- Konfigurationsinformationen ---")
    print(f"🧮 PyTorch Datentyp: {datatype}")
    print(f"📚 Anzahl an Beispiel-Fragen: {num_fewshot}")
    print(f"📦 Batchgröße: {batch_size}\n")

# ==========================
#         Hauptteil
# ==========================

if __name__ == '__main__':
    # Systeminformationen ausgeben, wenn dies noch nicht passiert ist
    if os.getenv("BENCH_PARENT") != "1":
        print_system_info()
        print("\n" + "=" * 60 + "\n")

    # Art der Analyse anzeigen
    print("--- Antwortqualität-Analyse ---\n")
    configuration_info()
    print("-" * 60)

    for model_name in models:
        # Prüfen ob das Modell bereits lokal vorhanden ist
        ensure_model_is_available(model_name, cache_dir, device)
        
        # Speicherpfad für JSON-Datei prüfen und ggf. anlegen
        safe_model_name = model_name.replace('/', '_')
        model_result_dir = os.path.join(results_dir, safe_model_name)
        os.makedirs(model_result_dir, exist_ok=True)

        # Setzt den Pfad für das zu untersuchende Modell
        pretrained_path = get_local_model_path(model_name)

        # Modell laden
        model = HFLM(
            pretrained=pretrained_path,
            trust_remote_code=True,
            dtype=datatype,
            device=device,
            cache_dir=cache_dir
        )

        # Durchlaufe alle Benchmarks und teste das Modell darauf
        for task in tasks:
            print(f"\n🔍 Evaluierung von {model_name} für Benchmark {task}...")

            # Zeit messen und Startzeit anzeigen
            now = datetime.now()
            start_time = now.strftime("%d.%m.%Y, %H:%M:%S")
            timestamp = now.strftime("%d.%m.%Y_%H-%M-%S")
            print(f"\n🕰️  Startzeit: {start_time}")

            # Einstellungen für die Evaluierung setzen
            evaluate_kwargs = {
                "model": model,
                "tasks": [task],
                "num_fewshot": num_fewshot,
                "batch_size": batch_size,
                "device": device
            }

            # Nur für humaneval: Codeausführung erlauben
            if task == "humaneval":
                print("\n⚠️  HumanEval: Codeausführung aktiviert!\n")
                evaluate_kwargs["confirm_run_unsafe_code"] = True

            # Evaluierung für eine Aufgabe durchführen
            results = lm_eval.simple_evaluate(**evaluate_kwargs)

            # JSON-Dateipfad zusammensetzen
            json_filename = f"{task}_{timestamp}.json"
            output_path = os.path.join(model_result_dir, json_filename)

            # Konvertiere Ergebnisse in ein JSON-kompatibles Format
            results_serializable = serialize_results(results)

            # Ergebnisse in JSON-Datei speichern
            with open(output_path, "w") as f:
                json.dump(results_serializable, f, indent=4)

            print(f"\n✅ Evaluierung abgeschlossen für {model_name} - {task}")
            print(f"💾 Ergebnisse gespeichert unter: {output_path} \n")
            print("-" * 60)

    print("\n✅ Antwortqualitätsmessungen abgeschlossen!")