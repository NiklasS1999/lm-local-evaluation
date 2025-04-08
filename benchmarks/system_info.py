# ==========================
#        Imports
# ==========================

# Standardbibliotheken
import platform
import shutil
import subprocess

# Drittanbieter-Bibliotheken
import psutil
import torch

# ==========================
#     Hilfsfunktionen
# ==========================

def _print_windows_info():
    """Gibt die Systeminformationen für einen Windows-Betriebssystem aus."""
    print("\n--- Systeminformationen (Windows) ---")
    os_edition = subprocess.getoutput('powershell -Command "(Get-WmiObject -Class Win32_OperatingSystem).Caption"').strip()
    cpu_name = subprocess.getoutput('powershell -Command "Get-CimInstance -ClassName Win32_Processor | Select-Object -ExpandProperty Name"').strip()
    total_ram = psutil.virtual_memory().total / (1024 ** 3)

    print(f"Betriebssystem: {os_edition}")
    print(f"CPU: {cpu_name}")
    print(f"CPU-Kerne: {psutil.cpu_count(logical=False)}, Threads: {psutil.cpu_count(logical=True)}")
    print(f"Gesamter RAM (gerundet): {round(total_ram)} GB")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Treiber-Version: {subprocess.getoutput('nvidia-smi --query-gpu=driver_version --format=csv,noheader')}")
    print(f"GPU Power Limit: {subprocess.getoutput('nvidia-smi --query-gpu=power.limit --format=csv,noheader')}")
    print(f"CUDA-Version: {torch.version.cuda}")
    print(f"cuDNN-Version: {torch.backends.cudnn.version()}")
    print(f"PyTorch-Version: {torch.__version__}")
    print(f"Python-Version: {platform.python_version()}")

def _print_linux_info():
    """Gibt die Systeminformationen für ein Linux-Betriebssystem aus."""
    print("\n--- Systeminformationen (Linux) ---")
    os_info = platform.platform()
    try:
        cpu_info = subprocess.getoutput("lscpu | grep 'Model name' | awk -F: '{print $2}'").strip()
    except Exception:
        cpu_info = platform.processor()
    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    gpu_name = "Nicht verfügbar"
    driver_version = "Nicht verfügbar"
    power_limit = "Nicht verfügbar"

    if shutil.which("nvidia-smi"):
        gpu_name = subprocess.getoutput("nvidia-smi --query-gpu=name --format=csv,noheader").strip()
        driver_version = subprocess.getoutput("nvidia-smi --query-gpu=driver_version --format=csv,noheader").strip()
        power_limit = subprocess.getoutput("nvidia-smi --query-gpu=power.limit --format=csv,noheader").strip()

    print(f"Betriebssystem: {os_info}")
    print(f"CPU: {cpu_info}")
    print(f"CPU-Kerne: {psutil.cpu_count(logical=False)}, Threads: {psutil.cpu_count(logical=True)}")
    print(f"Gesamter RAM (gerundet): {round(total_ram)} GB")
    print(f"GPU: {gpu_name}")
    print(f"GPU Treiber-Version: {driver_version}")
    print(f"GPU Power Limit: {power_limit}")
    print(f"CUDA-Version: {torch.version.cuda}")
    print(f"cuDNN-Version: {torch.backends.cudnn.version()}")
    print(f"PyTorch-Version: {torch.__version__}")
    print(f"Python-Version: {platform.python_version()}")

# ==========================
#         Hauptteil
# ==========================

def print_system_info():
    """Gibt die Systeminformationen je nach Betriebssystem in der Konsole aus."""
    if platform.system() == "Windows":
        _print_windows_info()
    elif platform.system() == "Linux":
        _print_linux_info()