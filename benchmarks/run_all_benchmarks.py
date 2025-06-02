import subprocess
import os
import sys
from system_info import print_system_info

# Setze Umgebungsvariable im aktuellen Prozess
os.environ["BENCH_PARENT"] = "1"

def run_benchmark_scripts():
    scripts = [
        "model_parametersize.py",
        "model_installationsize.py",
        "model_memoryusage.py",
        "model_latency.py",
        "model_responsequality.py",
        "benchmark_overview.py"
    ]

    for script in scripts:
        print(f"\n--- Starte: {script} ---")
        try:
            subprocess.run(
                [sys.executable, script],
                check=True,
                env={**os.environ}
            )
            print(f"--- Fertig: {script} ---\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Fehler beim Ausführen von {script}: {e}")
            break

if __name__ == "__main__":
    print_system_info()
    run_benchmark_scripts()