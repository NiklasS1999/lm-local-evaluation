# ==========================
#        Imports
# ==========================

# Standardbibliotheken
import os
import csv
import json
import re
from datetime import datetime
from pathlib import Path

# Drittanbieter-Bibliotheken
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
#     Konfigurationen
# ==========================

# Lokale Konfiguration erstellen
local_config = {
    "models" : ["Qwen/Qwen2.5-3B", "google/gemma-2-2b", "meta-llama/Llama-3.2-3B"],
    "results_dir" : "./models_cache"
}

# Laden der globalen Konfiguration (wenn vorhanden)
try:
    from global_config import CONFIG as config
except ImportError:
    config = local_config

# Setzen der finalen Konfiguration
models = config["models"]
results_dir = config["results_dir"]

# Verzeichnis anlegen, wenn dieses noch nicht existiert
os.makedirs(results_dir, exist_ok=True)

# --- weitere Konfigurationen ---
# Übersicht aller Benchmarks mit den jeweils relevanten Metrik-Feldern
BENCHMARKS = {
    "installation_size": ["Installationsgröße (lesbar)"],
    "latency": ["Cold-Start (Ø, s)", "Cold-Start (±)", "Inferenz (Ø, s)", "Inferenz (±)"],
    "memory_usage": [
        "RAM Laden Ø [MB]", "RAM Laden Std. [MB]",
        "RAM Inferenz Ø [MB]", "RAM Inferenz Std. [MB]",
        "RAM Gesamt Ø [MB]", "RAM Gesamt Std. [MB]",
        "GPU Verbrauch Ø [MB]"
    ],
    "parameter_size": ["Parametergröße (Mrd.)"],
    "mmlu": ["acc,none", "acc_stderr,none"],
    "hellaswag": ["acc,none", "acc_stderr,none", "acc_norm,none", "acc_norm_stderr,none"],
    "gsm8k": ["exact_match,strict-match", "exact_match_stderr,strict-match", "exact_match,flexible-extract", "exact_match_stderr,flexible-extract"],
    "humaneval": ["pass@1,create_test", "pass@1_stderr,create_test"],
    "boolq": ["acc,none", "acc_stderr,none"]
}

# Anzeige-Name und Einheit pro Feld zur besseren Lesbarkeit in CSV/Plots
METRIC_DISPLAY_AND_UNIT = {
    # Installationsgröße, Latenz, Parametergröße
    "Installationsgröße (lesbar)": ("Installationsgröße", "GB"),
    "Cold-Start (Ø, s)": ("Gesamtdauer (Modellstart + Antwort)", "Sekunden"),
    "Cold-Start (±)": ("Gesamtdauer – Streuung", "Sekunden"),
    "Inferenz (Ø, s)": ("Antwortdauer (ohne Modellstart)", "Sekunden"),
    "Inferenz (±)": ("Antwortdauer – Streuung", "Sekunden"),
    "Parametergröße (Mrd.)": ("Parametergröße", "Mrd."),

    # Speicherverbrauch
    "RAM Laden Ø [MB]": ("RAM-Belegung Modellstart", "MB"),
    "RAM Laden Std. [MB]": ("RAM-Belegung Modellstart (Streuung)", "MB"),
    "RAM Inferenz Ø [MB]": ("RAM-Belegung Antwort", "MB"),
    "RAM Inferenz Std. [MB]": ("RAM-Belegung Antwort (Streuung)", "MB"),
    "RAM Gesamt Ø [MB]": ("RAM-Belegung gesamt", "MB"),
    "RAM Gesamt Std. [MB]": ("RAM-Belegung gesamt (Streuung)", "MB"),
    "GPU Verbrauch Ø [MB]": ("GPU RAM-Belegung", "MB"),

    # Antwortqualität – Eval Benchmarks
    "acc,none": ("Genauigkeit", "Prozent"),
    "acc_stderr,none": ("Genauigkeit – Streuung", "Prozent"),
    "acc_norm,none": ("Normierte Genauigkeit", "Prozent"),
    "acc_norm_stderr,none": ("Normierte Genauigkeit – Streuung", "Prozent"),
    "exact_match,strict-match": ("Exakte Übereinstimmung", "Prozent"),
    "exact_match_stderr,strict-match": ("Exakte Übereinstimmung – Streuung", "Prozent"),
    "exact_match,flexible-extract": ("Exakte Übereinstimmung – flexibel", "Prozent"),
    "exact_match_stderr,flexible-extract": ("Exakte Übereinstimmung – flexibel – Streuung", "Prozent"),
    "pass@1,create_test": ("Lösungsrate", "Prozent"),
    "pass@1_stderr,create_test": ("Lösungsrate – Streuung", "Prozent"),
}

# Überschriften in Konsolen- oder CSV-Ausgabe pro Benchmark-Typ
SECTION_TITLES = {
    "installation_size": "Installationsgröße",
    "latency": "Latenz",
    "memory_usage": "Speicherverbrauch",
    "parameter_size": "Modellgröße",
    "mmlu": "Allgemeinwissen & akademisches Wissen (MMLU)",
    "hellaswag": "Alltagslogik & Szenarien-Verständnis (HellaSwag)",
    "gsm8k": "Mathematische Denkaufgaben (GSM8K)",
    "humaneval": "Programmierfähigkeiten (HumanEval)",
    "boolq": "Ja-/Nein-Fragen zu kurzen Texten (BoolQ)",
}

# Hauptmetrik je Benchmark für Vergleichsplots (Eval Benchmarks)
EVAL_PLOTS_METRICS = {
    "acc,none": "Genauigkeit",
    "exact_match,strict-match": "Exakte Übereinstimmung",
    "pass@1,create_test": "Lösungsrate"
}

# Kürzel für eval-Benchmarks
BENCHMARK_LABELS = {
    "mmlu": "MMLU",
    "hellaswag": "HellaSwag",
    "gsm8k": "GSM8K",
    "humaneval": "HumanEval",
    "boolq": "BoolQ"
}

# Automatische Ableitungen
DISPLAY_NAME_MAP = {k: v[0] for k, v in METRIC_DISPLAY_AND_UNIT.items()}
UNIT_SUFFIX_MAP = {v[0]: v[1] for k, v in METRIC_DISPLAY_AND_UNIT.items()}

# Metriken, die gerundet (4 Nachkommastellen) ausgegeben werden sollen
ROUND_FLOAT_FOR = {
    "Genauigkeit",
    "Genauigkeit – Streuung",
    "Normierte Genauigkeit",
    "Normierte Genauigkeit – Streuung",
    "Exakte Übereinstimmung",
    "Exakte Übereinstimmung – Streuung",
    "Exakte Übereinstimmung (flexibel)",
    "Exakte Übereinstimmung – Streuung (flexibel)",
    "Lösungsrate",
    "Lösungsrate – Streuung"
}

# Felder, die als Prozent ausgegeben werden sollen (× 100 + "%")
SHOW_AS_PERCENT = {
    "acc,none",
    "acc_stderr,none",
    "acc_norm,none",
    "acc_norm_stderr,none",
    "exact_match,strict-match",
    "exact_match_stderr,strict-match",
    "exact_match,flexible-extract",
    "exact_match_stderr,flexible-extract",
    "pass@1,create_test",
    "pass@1_stderr,create_test"
}

# Hauptmetriken nach Benchmark (für Zuordnung von Spaltennamen)
EVAL_METRIC_TO_BENCHMARK = {
    "acc,none": ["mmlu", "hellaswag", "boolq"],
    "exact_match,strict-match": ["gsm8k"],
    "exact_match,flexible-extract": ["gsm8k"],
    "pass@1,create_test": ["humaneval"]
}

# ==========================
#     Hilfsfunktionen
# ==========================

def find_latest_file(directory, prefix):
    """Finde die neueste Datei mit gegebenem Prefix (.csv oder .json) im Verzeichnis."""
    files = list(directory.glob(f"{prefix}_*.csv")) + list(directory.glob(f"{prefix}_*.json"))
    if not files:
        return None

    def extract_timestamp(f):
        match = re.search(r"_(\d{2}\.\d{2}\.\d{4})_(\d{2}-\d{2}-\d{2})", f.name)
        if match:
            date_str = match.group(1) + match.group(2)
            return datetime.strptime(date_str, "%d.%m.%Y%H-%M-%S")
        return datetime.min

    return max(files, key=extract_timestamp)

def read_csv_data(filepath, fields):
    """Lese relevante Felder aus CSV-Datei. Für Speicherverbrauch spezielle Behandlung."""
    if "memory_usage" in filepath.name:
        return read_memory_usage_summary(filepath, fields)
    if "latency" in filepath.name:
        return read_latency_summary(filepath, fields)

    with open(filepath, newline='', encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        if not reader:
            return {}

        last_row = reader[-1]
        result = {}
        for k in fields:
            val = last_row.get(k, "")
            try:
                result[k] = float(val.replace(",", "."))
            except (ValueError, AttributeError):
                result[k] = val
        return result

def read_latency_summary(filepath, fields):
    """Extrahiere Latenzwerte aus CSV mit zwei Blöcken – nimm den zweiten Header + letzte Zeile."""
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()

        # Nur Zeilen aus dem unteren Block behalten (ab Zeile mit 'Modell')
        summary_lines = [line.strip() for line in lines if line.strip().startswith("Modell") or line.strip().startswith(tuple(local_config["models"]))]

        if len(summary_lines) < 2:
            return {}

        import io
        reader = csv.DictReader(io.StringIO("\n".join(summary_lines)))
        rows = list(reader)

        if not rows:
            return {}

        last_row = rows[-1]
        result = {}
        for k in fields:
            val = last_row.get(k, "").strip()
            try:
                result[k] = float(val.replace(",", "."))
            except (ValueError, AttributeError):
                result[k] = val

        return result

    except Exception as e:
        print(f"⚠️ Fehler beim Parsen der Latenz aus {filepath.name}: {e}")
        return {}

def read_memory_usage_summary(filepath, fields):
    """Extrahiere Speicherverbrauch aus spezieller CSV-Ausgabe mit Durchschnittszeile."""
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            if "Gesamtdurchschnitt pro Modell" in line:
                header_line = lines[idx + 1].strip()
                value_line = lines[idx + 2].strip()
                break
        else:
            return {}

        headers = header_line.split(",")
        values = value_line.split(",")

        if len(headers) != len(values):
            return {}

        row = dict(zip(headers, values))
        result = {}
        for k in fields:
            val = row.get(k, "").strip()
            try:
                result[k] = float(val.replace(",", "."))
            except (ValueError, AttributeError):
                result[k] = val

        return result

    except Exception as e:
        print(f"⚠️ Fehler beim Parsen von Speicherverbrauch: {filepath.name}: {e}")
        return {}

def read_json_data(filepath, benchmark_key, keys):
    """Lese relevante Felder aus JSON-Ausgabe des Benchmark-Tools."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    try:
        section = data["results"][benchmark_key]
        return {k: section.get(k, "n/a") for k in keys}
    except KeyError:
        return {}

def collect_model_data(model_dir):
    """Durchlaufe alle Benchmarks und sammle Metriken für ein Modellverzeichnis."""
    model_data = {}
    for benchmark, fields in BENCHMARKS.items():
        file = find_latest_file(model_dir, benchmark)
        if not file:
            continue
        if file.suffix == ".csv":
            data = read_csv_data(file, fields)
        elif file.suffix == ".json":
            data = read_json_data(file, benchmark, fields)
        else:
            data = {}
        model_data[benchmark] = data
    return model_data

def format_console_output(model, data):
    """Gibt die Benchmark-Ergebnisse formatiert in der Konsole aus."""
    print(f"\n{'='*60}")
    print(f"Modell: {model}")
    print(f"{'='*60}")

    all_benchmark_keys = list(BENCHMARKS.keys())
    max_label_len = max(
        (len(DISPLAY_NAME_MAP.get(label_key, label_key))
         for b_key in all_benchmark_keys
         for label_key in data.get(b_key, {})),
        default=0
    )

    for b_key in all_benchmark_keys:
        title = SECTION_TITLES.get(b_key, b_key).upper()
        print(f"\n[{title}]")
        benchmark_data = data.get(b_key, {})

        if benchmark_data:
            for key, val in benchmark_data.items():
                label = DISPLAY_NAME_MAP.get(key, key)
                suffix = UNIT_SUFFIX_MAP.get(label, "")

                if isinstance(val, float):
                    if key in SHOW_AS_PERCENT:
                        val_str = f"{round(val * 100, 2):.2f} %"
                    elif key in ROUND_FLOAT_FOR:
                        val_str = f"{val:.4f}"
                    else:
                        val_str = f"{val:.2f}"
                        if suffix:
                            val_str += f" {suffix}"
                else:
                    val_str = str(val)

                print(f"{label:<{max_label_len}} : {val_str}")
        else:
            print("– keine Daten verfügbar –")

    print()

def write_csv_summary(all_data,output_file="results/benchmark_overview.csv"):
    """Schreibt die gesammelten Benchmark-Daten in eine CSV-Datei."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    eval_main_metrics = {
        "mmlu": "acc,none",
        "hellaswag": "acc,none",
        "gsm8k": "exact_match,strict-match",
        "humaneval": "pass@1,create_test",
        "boolq": "acc,none",
    }

    eval_metric_to_benchmark = {}
    for b_key, metric in eval_main_metrics.items():
        eval_metric_to_benchmark.setdefault(metric, []).append(b_key)

    ordered_metrics = []
    for b in ["installation_size", "latency", "memory_usage", "parameter_size"]:
        ordered_metrics.extend(BENCHMARKS[b])

    ordered_eval_metrics = [(b_key, k) for b_key, k in eval_main_metrics.items()]

    fieldnames = ["Modell"]
    full_label_map = {}

    for k in ordered_metrics:
        label = DISPLAY_NAME_MAP.get(k, k)
        full_label_map[(k, "")] = label
        fieldnames.append(label)

    for b_key, k in ordered_eval_metrics:
        label = DISPLAY_NAME_MAP.get(k, k)
        suffix = f" ({BENCHMARK_LABELS[b_key]})"
        full_label = label + suffix
        full_label_map[(k, b_key)] = full_label
        fieldnames.append(full_label)
    
    flex_label = DISPLAY_NAME_MAP["exact_match,flexible-extract"]
    flex_field = f"{flex_label} ({BENCHMARK_LABELS['gsm8k']})"
    full_label_map[("exact_match,flexible-extract", "gsm8k")] = flex_field
    gsm_index = fieldnames.index("Exakte Übereinstimmung (GSM8K)")
    fieldnames.insert(gsm_index + 1, flex_field)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model, benchmarks in all_data.items():
            row = {"Modell": model}

            for b_key, b_data in benchmarks.items():
                for k, v in b_data.items():
                    if (k, "") in full_label_map:
                        full_label = full_label_map[(k, "")]
                    elif (k, b_key) in full_label_map:
                        full_label = full_label_map[(k, b_key)]
                    else:
                        continue

                    label = DISPLAY_NAME_MAP.get(k, k)
                    suffix = UNIT_SUFFIX_MAP.get(label, "")
                    value_str = ""

                    if isinstance(v, float):
                        if k in SHOW_AS_PERCENT:
                            value_str = f"{round(v * 100, 2):.2f} %"
                        elif k in ROUND_FLOAT_FOR:
                            value_str = f"{v:.4f}"
                        else:
                            value_str = f"{v:.2f}"

                        if suffix and "Prozent" not in value_str and "%" not in value_str:
                            value_str = f"{value_str} {suffix}"
                    else:
                        value_str = str(v)

                    row[full_label] = value_str

            writer.writerow(row)

def generate_comparison_plots(csv_path="results/benchmark_overview.csv",plot_dir="results/result_plots"):
    """Erstellt Balkendiagramme für ausgewählte technische und Evaluationsmetriken."""
    try:
        os.makedirs(plot_dir, exist_ok=True)
        df = pd.read_csv(csv_path)
        df.set_index("Modell", inplace=True)

        tech_cols = [
            DISPLAY_NAME_MAP["Inferenz (Ø, s)"],
            DISPLAY_NAME_MAP["Cold-Start (Ø, s)"],
            DISPLAY_NAME_MAP["Installationsgröße (lesbar)"],
            DISPLAY_NAME_MAP["RAM Gesamt Ø [MB]"],
            DISPLAY_NAME_MAP["GPU Verbrauch Ø [MB]"],
            DISPLAY_NAME_MAP["Parametergröße (Mrd.)"]
        ]

        unit_labels = {
            "Sekunden": "Sekunden",
            "GB": "Gigabyte",
            "MB": "Megabyte",
            "Prozent": "Prozent",
            "Mrd.": "Milliarden"
        }

        for col in tech_cols:
            if col not in df.columns:
                print(f"⚠️ Spalte '{col}' fehlt – übersprungen.")
                continue

            cleaned = df[col].replace(["–", "n/a", "", None], pd.NA).astype(str)
            numeric = pd.to_numeric(
                cleaned.str.extract(r"([\d.,]+)")[0].str.replace(",", "."),
                errors="coerce"
            )
            numeric.index = df.index

            if numeric.isna().all():
                print(f"⚠️ Keine Daten für '{col}' – kein Plot.")
                continue

            df[col] = numeric
            raw_unit = UNIT_SUFFIX_MAP.get(col, "")
            unit = unit_labels.get(raw_unit, "")

            ax = df[col].plot(kind="bar", figsize=(8, 5), title=col, rot=45)

            for idx, val in enumerate(df[col]):
                if pd.isna(val):
                    dummy = max(1, df[col].max() * 0.05)
                    ax.bar(idx, dummy, color="lightgray", alpha=0.5)
                    ax.text(idx, dummy / 2, "Keine Daten",
                            ha="center", va="center", fontsize=9, color="gray")
                else:
                    ax.text(idx, val / 2, f"{val:.2f}",
                            ha="center", va="center", fontsize=9, color="white")

            ax.set_ylabel(unit)
            ax.set_xlabel("Modell")
            plt.tight_layout()

            safe_name = col.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "").replace("+", "plus")
            plt.savefig(os.path.join(plot_dir, f"plot_{safe_name}.png"))
            plt.close()

        eval_metrics = {
            "mmlu": ["acc,none"],
            "hellaswag": ["acc,none"],
            "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract"],
            "humaneval": ["pass@1,create_test"],
            "boolq": ["acc,none"]
        }

        for key, metrics in eval_metrics.items():
            for metric in metrics:
                name = DISPLAY_NAME_MAP.get(metric, metric)
                col = f"{name} ({BENCHMARK_LABELS[key]})"

                if col not in df.columns:
                    print(f"⚠️ Spalte '{col}' fehlt – übersprungen.")
                    continue

                cleaned = df[col].replace(["–", "n/a", "", None], pd.NA).astype(str)
                numeric = pd.to_numeric(
                    cleaned.str.extract(r"([\d.,]+)")[0].str.replace(",", "."),
                    errors="coerce"
                )
                numeric.index = df.index

                if numeric.isna().all():
                    print(f"⚠️ Keine Daten für '{col}' – kein Plot.")
                    continue

                df[col] = numeric
                ax = df[col].plot(kind="bar", figsize=(8, 5), title=col, rot=45)

                for idx, val in enumerate(df[col]):
                    if pd.isna(val):
                        dummy = max(1, df[col].max() * 0.05)
                        ax.bar(idx, dummy, color="lightgray", alpha=0.5)
                        ax.text(idx, dummy / 2, "Keine Daten",
                                ha="center", va="center", fontsize=9, color="gray")
                    else:
                        ax.text(idx, val / 2, f"{val:.2f}%",
                                ha="center", va="center", fontsize=9, color="white")

                ax.set_ylabel("Prozent")
                ax.set_xlabel("Modell")
                plt.tight_layout()

                # 🔁 Neues Dateinamensschema für GSM8K
                if key == "gsm8k":
                    suffix = "strict" if "strict" in metric else "flexible"
                    filename = f"plot_gsm8k_{suffix}.png"
                else:
                    filename = f"plot_{key}.png"

                plt.savefig(os.path.join(plot_dir, filename))
                plt.close()

        print(f"✅ Einzelplots gespeichert unter: '{plot_dir}'")

    except Exception as e:
        print(f"⚠️ Fehler bei Diagrammerstellung: {e}")

def generate_combined_technical_plot(csv_path="results/benchmark_overview.csv",output_path="results/result_plots/plot_technical_combined.png"):
    """Erstellt einen kombinierten Balkenplot für technische Metriken."""
    try:
        df = pd.read_csv(csv_path)
        df.set_index("Modell", inplace=True)

        columns_to_plot = [
            DISPLAY_NAME_MAP["Inferenz (Ø, s)"],
            DISPLAY_NAME_MAP["Cold-Start (Ø, s)"],
            DISPLAY_NAME_MAP["Installationsgröße (lesbar)"],
            DISPLAY_NAME_MAP["RAM Gesamt Ø [MB]"],
            DISPLAY_NAME_MAP["GPU Verbrauch Ø [MB]"],
            DISPLAY_NAME_MAP["Parametergröße (Mrd.)"]
        ]

        unit_labels = {
            "Sekunden": "Sekunden",
            "GB": "Gigabyte",
            "MB": "Megabyte",
            "Prozent": "Prozent",
            "Mrd.": "Milliarden"
        }

        available_columns = [col for col in columns_to_plot if col in df.columns]
        if not available_columns:
            print("⚠️ Keine technischen Metriken vorhanden – gemeinsamer Plot wird übersprungen.")
            return

        rows, cols = 2, 3
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.flatten()

        for i, col in enumerate(available_columns):
            cleaned = df[col].replace(["–", "n/a", "", None], pd.NA).astype(str)
            numeric = pd.to_numeric(
                cleaned.str.extract(r"([\d.,]+)")[0].str.replace(",", "."),
                errors="coerce"
            )
            numeric.index = df.index

            ax = axes[i]
            numeric.plot(kind="bar", ax=ax, color="skyblue", rot=45)
            ax.set_title(col)
            ax.set_xlabel("")

            raw_unit = UNIT_SUFFIX_MAP.get(col, "")
            unit_label = unit_labels.get(raw_unit, "")
            ax.set_ylabel(unit_label)

            max_val = numeric.max()
            only_dummy = numeric.isna().all()
            dummy_height = 1 if only_dummy else max_val * 0.05
            text_offset = 0.2 if only_dummy else max_val * 0.02

            for idx, (model_name, val) in enumerate(numeric.items()):
                if pd.isna(val):
                    ax.bar(idx, dummy_height, color="lightgray", alpha=0.5)
                    ax.text(
                        idx,
                        dummy_height + text_offset,
                        "Keine Daten",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="gray"
                    )
                else:
                    ax.text(
                        idx,
                        val / 2,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white"
                    )

            if only_dummy:
                ax.set_ylim(0, 10)

        # Leere Subplots entfernen
        for j in range(len(available_columns), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print("✅ Gemeinsamer technischer Plot gespeichert unter:", output_path)

    except Exception as e:
        print(f"⚠️ Fehler beim Erstellen des kombinierten technischen Plots: {e}")

def generate_combined_eval_plot(csv_path="results/benchmark_overview.csv",output_path="results/result_plots/plot_eval_quality_combined.png"):
    """Erstellt einen kombinierten Balkenplot für die Evaluationsmetriken."""
    try:
        df = pd.read_csv(csv_path)
        df.set_index("Modell", inplace=True)

        eval_columns = [
            f"{DISPLAY_NAME_MAP['acc,none']} ({BENCHMARK_LABELS['mmlu']})",
            f"{DISPLAY_NAME_MAP['acc,none']} ({BENCHMARK_LABELS['hellaswag']})",
            f"{DISPLAY_NAME_MAP['exact_match,strict-match']} ({BENCHMARK_LABELS['gsm8k']})",
            f"{DISPLAY_NAME_MAP['exact_match,flexible-extract']} ({BENCHMARK_LABELS['gsm8k']})",
            f"{DISPLAY_NAME_MAP['pass@1,create_test']} ({BENCHMARK_LABELS['humaneval']})",
            f"{DISPLAY_NAME_MAP['acc,none']} ({BENCHMARK_LABELS['boolq']})"
        ]

        available_columns = [col for col in eval_columns if col in df.columns]
        if not available_columns:
            print("⚠️ Keine Eval-Metriken vorhanden – gemeinsamer Vergleichsplot wird übersprungen.")
            return

        fig, axes = plt.subplots(
            1, len(available_columns),
            figsize=(5 * len(available_columns), 5),
            sharey=False
        )

        if len(available_columns) == 1:
            axes = [axes]

        for i, col in enumerate(available_columns):
            cleaned_series = df[col].replace(["–", "n/a", "", None], pd.NA).astype(str)
            numeric_series = pd.to_numeric(
                cleaned_series.str.extract(r"([\d.,]+)")[0].str.replace(",", "."),
                errors="coerce"
            )
            numeric_series.index = df.index

            ax = axes[i]
            numeric_series.plot(kind="bar", ax=ax, color="skyblue", rot=45)
            ax.set_title(col)
            ax.set_xlabel("")
            ax.set_ylabel("Prozent")
            ax.yaxis.set_label_position("left")

            max_value = numeric_series.max()
            only_dummy = numeric_series.isna().all()
            dummy_height = 1 if only_dummy else max_value * 0.05
            text_offset = 0.2 if only_dummy else max_value * 0.02

            for idx, (model_name, val) in enumerate(numeric_series.items()):
                if pd.isna(val):
                    ax.bar(idx, dummy_height, color="lightgray", alpha=0.5)
                    ax.text(
                        idx,
                        dummy_height + text_offset,
                        "Keine Daten",
                        ha="center",
                        va="bottom",
                        color="gray",
                        fontsize=9
                    )
                else:
                    ax.text(
                        idx,
                        val / 2,
                        f"{val:.2f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white"
                    )

            if only_dummy:
                ax.set_ylim(0, 10)

        for ax in axes:
            ax.set_ylabel("Prozent")
            ax.yaxis.set_label_position("left")
            ax.get_yaxis().set_visible(True)
            ax.tick_params(labelleft=True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print("✅ Gemeinsamer Eval-Plot gespeichert unter:", output_path)

    except Exception as e:
        print(f"⚠️ Fehler beim Erstellen des kombinierten Eval-Plots: {e}")

def evaluate_model_advantages(csv_path="results/benchmark_overview.csv",output_path="results/model_advantages.txt"):
    """Analysiert Vorteile und Nachteile je Modell anhand der Metriken."""
    try:
        df = pd.read_csv(csv_path)
        df.set_index("Modell", inplace=True)

        lower_is_better = {
            "Installationsgröße",
            "Gesamtdauer (Modellstart + Antwort)",
            "Antwortdauer (ohne Modellstart)",
            "RAM-Belegung Antwort",
            "RAM-Belegung gesamt",
            "GPU RAM-Belegung",
            "Parametergröße"
        }

        higher_is_better = {
            "Genauigkeit (MMLU)",
            "Genauigkeit (HellaSwag)",
            "Exakte Übereinstimmung (GSM8K)",
            "Exakte Übereinstimmung – flexibel (GSM8K)",
            "Lösungsrate (HumanEval)",
            "Genauigkeit (BoolQ)"
        }

        metrics = list(lower_is_better | higher_is_better)
        available_metrics = [m for m in metrics if m in df.columns]
        model_evaluation = {
            model: {"Vorteile": [], "Nachteile": []}
            for model in df.index
        }

        for metric in available_metrics:
            cleaned = df[metric].replace(["–", "n/a", "", None], pd.NA).astype(str)
            numeric = pd.to_numeric(
                cleaned.str.extract(r"([\d.,]+)")[0].str.replace(",", "."),
                errors="coerce"
            )
            numeric.index = df.index

            if numeric.isna().any():
                continue

            ascending = metric in lower_is_better
            sorted_series = numeric.sort_values(ascending=ascending)
            values = sorted_series.values

            best_value = values[0]
            second_best = values[1] if len(values) > 1 else best_value
            worst_value = values[-1]
            second_worst = values[-2] if len(values) > 1 else worst_value

            for model in df.index:
                model_value = numeric[model]

                if model_value == best_value:
                    if second_best == 0:
                        annotation = "bester Wert"
                    else:
                        diff = (
                            (second_best - best_value) / second_best
                            if ascending else
                            (best_value - second_best) / second_best
                        )
                        pct_str = f"{diff * 100:.0f}%"
                        sign = "-" if ascending else "+"
                        annotation = f"{sign}{pct_str}"
                    model_evaluation[model]["Vorteile"].append(f"{metric} ({annotation})")

                elif model_value == worst_value:
                    if second_worst == 0:
                        annotation = "schlechtester Wert"
                    else:
                        diff = (
                            (model_value - second_worst) / second_worst
                            if ascending else
                            (second_worst - model_value) / second_worst
                        )
                        pct_str = f"{diff * 100:.0f}%"
                        sign = "+" if ascending else "-"
                        annotation = f"{sign}{pct_str}"
                    model_evaluation[model]["Nachteile"].append(f"{metric} ({annotation})")

        # Konsolenausgabe
        print("\n" + "=" * 60 + "\n")
        print("📊 Vergleich: Vorteile & Nachteile je Modell (mit prozentualem Unterschied)\n")
        print("-" * 60 + "\n")

        for model, result in model_evaluation.items():
            print(f"Modell: {model}\n")
            print("  Vorteile:")
            if result["Vorteile"]:
                for v in result["Vorteile"]:
                    print(f"    ✔ {v}")
            else:
                print("    – keine")

            print("  Nachteile:")
            if result["Nachteile"]:
                for n in result["Nachteile"]:
                    print(f"    ✘ {n}")
            else:
                print("    – keine")
            print("\n" + "-" * 60 + "\n")

        # Dateiausgabe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("📊 Vergleich: Vorteile & Nachteile je Modell (mit prozentualem Unterschied)\n")
            f.write("=" * 60 + "\n\n")

            for model, result in model_evaluation.items():
                f.write(f"Modell: {model}\n\n")
                f.write("  Vorteile:\n")
                if result["Vorteile"]:
                    for v in result["Vorteile"]:
                        f.write(f"    ✔ {v}\n")
                else:
                    f.write("    – keine\n")

                f.write("  Nachteile:\n")
                if result["Nachteile"]:
                    for n in result["Nachteile"]:
                        f.write(f"    ✘ {n}\n")
                else:
                    f.write("    – keine\n")

                f.write("-" * 60 + "\n\n")

        print(f"📄 Vorteile & Nachteile gespeichert unter: {output_path}")
        print("\n" + "=" * 60 + "\n")

    except Exception as e:
        print(f"⚠️ Fehler bei der Modellauswertung: {e}")

# ==========================
#         Hauptteil
# ==========================

def main():
    """Hauptfunktion zur Erstellung der Benchmark-Übersicht."""
    print("📊 Erstelle Benchmark-Übersicht...")

    all_model_data = {}

    for model in models:
        model_dir = Path(results_dir) / model.replace("/", "_")

        if model_dir.exists():
            data = collect_model_data(model_dir)
            all_model_data[model] = data
            format_console_output(model, data)
        else:
            print(f"⚠️  Verzeichnis nicht gefunden: {model_dir}")

    write_csv_summary(all_model_data)
    generate_comparison_plots()
    generate_combined_technical_plot()
    generate_combined_eval_plot()
    evaluate_model_advantages()

    print("✅ Fertig! Ergebnis gespeichert in 'results/benchmark_overview.csv'.\n")


if __name__ == "__main__":
    main()