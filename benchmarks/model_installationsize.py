# ==========================
#        Imports
# ==========================

# Standardbibliotheken
import csv  # Für CSV-Ausgabe
import io  # Für String-basierte Umleitung von stdout/stderr
import os  # Für Dateipfad- und Verzeichnisoperationen
import warnings  # Zum Unterdrücken von Warnmeldungen
from contextlib import redirect_stdout, redirect_stderr  # Umleitung von Ausgabe und Fehlermeldungen
from datetime import datetime  # Zeitstempel erstellen

# Drittanbieter-Bibliotheken
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as transformers_logging  # Transformers

# ==========================
#     Konfigurationen
# ==========================

# Lokale Konfiguration erstellen
local_config = {
    "models" : ["Qwen/Qwen2.5-3B", "google/gemma-2-2b", "meta-llama/Llama-3.2-3B"],
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
device = config["device"]
cache_dir = config["cache_dir"]
results_dir = config["results_dir"]

# Verzeichnis anlegen, wenn dieses noch nicht existiert
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Warnungen und unnötige Protokollierungen unterdrücken
warnings.filterwarnings("ignore", category=UserWarning)
transformers_logging.set_verbosity_error()

# ==========================
#     Hilfsfunktionen
# ==========================

def get_directory_size(directory):
    """Berechnet die Größe eines Verzeichnisses in Bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size

def format_size(size_in_bytes):
    """Formatiert die Größe in ein lesbares Format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024

def get_model_directory(cache_dir, model_name):
    """Gibt den Pfad des Modell-Speicherorts zurück."""
    # HuggingFace speichert Modelle in diesem Format ab
    return os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")

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

def save_results_to_csv(csv_path, model_name, formatted_size, size_in_bytes):
    """Speichert die Ergebnisse der Speicherverbrauchsanalyse in eine CSV-Datei."""
    with open(csv_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Modellname", "Installationsgröße (lesbar)", "Installationsgröße (Bytes)"])
        writer.writerow([model_name, formatted_size, size_in_bytes])

# ==========================
#         Hauptteil
# ==========================

if __name__ == '__main__':
    # Art der Analyse anzeigen
    print("--- Festplatten-Speicherverbrauch Analyse ---\n")
    print("=" * 60)

    for model_name in models:
        # Prüfen ob das Modell bereits lokal vorhanden ist
        ensure_model_is_available(model_name, cache_dir, device)

        # Zeit messen und Startzeit anzeigen
        now = datetime.now()
        start_time = now.strftime("%d.%m.%Y, %H:%M:%S")
        timestamp = now.strftime("%d.%m.%Y_%H-%M-%S")
        print(f"\n🕰️  Startzeit: {start_time}")

        # Model-Speicherpfad bestimmen
        model_dir = get_model_directory(cache_dir, model_name)

        # Speicherverbrauch berechnen
        size_in_bytes = get_directory_size(model_dir)
        formatted_size = format_size(size_in_bytes)
    
        print(f"{model_name}: Gesamter Speicherverbrauch: {formatted_size} \n")

        # Speicherpfad für CSV-Datei prüfen und ggf. anlegen
        safe_model_name = model_name.replace('/', '_')
        model_result_dir = os.path.join(results_dir, safe_model_name)
        os.makedirs(model_result_dir, exist_ok=True)

        # CSV-Dateipfad zusammensetzen
        csv_filename = f"installation_size_{timestamp}.csv"
        csv_path = os.path.join(model_result_dir, csv_filename)

        # CSV-Datei erstellen
        save_results_to_csv(csv_path, model_name, formatted_size, size_in_bytes)

        print(f"💾 Ergebnisse gespeichert unter: {csv_path} \n")
        print("=" * 60)

    print("\n✅ Festplatten-Speicherverbrauchsmessung abgeschlossen!")