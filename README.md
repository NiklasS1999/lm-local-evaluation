# Allgemeines

Dieses Repository basiert auf einem Fork von [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) und enthält wichtige Fixes und Erweiterungen, um aktuelle Language Models (z. B. Qwen, Gemma, LLaMA 3) unter Linux oder Windows 11 mit CUDA 12.1–12.8 einerseits in Bezug auf die Antwortqualität und andererseits in Bezug auf technische Metriken benchmarken zu können.

---

## 🔧 Voraussetzungen

- ✅ Python 3.11
- ✅ Git + GitHub-Zugang
- ✅ Windows 11 oder Linux
- ✅ NVIDIA GPU mit CUDA 12.x Unterstützung (z.B. GeForce RTX 3060)
- ✅ NVIDIA CUDA-Toolkit
- ✅ NVIDIA cuDNN-Bibliothek
- ✅ Huggingface-Zugang
- ✅ GitHub lm-local-evaluation Repository
- ✅ Virtual Environment (empfohlen)
- ✅ Nano-Texteditor (empfohlen)

---

## 📖🪟 Anleitung - Installation der Voraussetzungen (neues Windows 11 System)

1. Python 3.11.0 herunterladen und installieren
    - Python 3.11.0 von der [offiziellen Webseite](https://www.python.org/downloads/release/python-3110/) herunterladen und installieren
    - mit "python --info" in der Konsole prüfen, ob die Installation geklappt hat
2. Git herunterladen und installieren
    - Git von der [offiziellen Webseite](https://git-scm.com/downloads/win) herunterladen und installieren
    - mit "git --version" in der Konsole prüfen, ob die Installation geklappt hat
3. Git Umgebungsvariable setzen
    - Windows+S Taste drücken: “Umgebungsvariable bearbeiten” suchen
    - Wähle “Umgebungsvariablen”
    - unter “Systemvariablen”, zu “Path” gehen und “Bearbeiten” auswählen
    - “Neu” anklicken und den Installationspfad von Git hinzufügen (z.B. C:\Program Files\Git\bin)
    - auf “OK” klicken und alle Fenster schließen
    - Terminal neu starten und mit “git --version” testen, ob git erkannt wird
4. CUDA installieren
    - überprüfen, ob die eigene [Nvidia GPU CUDA fähig](https://developer.nvidia.com/cuda-gpus) ist
    - neueste Nvidia Grafikkarten-Softwareversion über z.B. Geforce Experience installieren
    - CUDA Toolkit für Windows [herunterladen](https://developer.nvidia.com/cuda-downloads) und installieren
    - Computer neustarten
5. cuDNN-Bibliothek installieren
    - cuDNN-Bibliothek [herunterladen](https://developer.nvidia.com/cudnn-downloads) (z.B. Windows - x86_64 - Tarball - 12)
    - heruntergeladene Dateien entpacken
    - einzelne Dateien (bin, include, lib) in die Verzeichnisse des CUDA-Installationsordners kopieren (z.B. Default: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8)
6. CUDA Umgebungsvariable setzen
    - Windows+S Taste drücken: “Umgebungsvariable bearbeiten” suchen
    - Wähle “Umgebungsvariablen”
    - unter “Systemvariablen”, zu “Path” gehen und “Bearbeiten” auswählen
    - “Neu” anklicken und den Installationspfad von CUDA hinzufügen (z.B. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\)
    - auf “OK” klicken und alle Fenster schließen
    - Terminal neu starten und mit “nvcc --version” testen, ob CUDA erkannt wird
7. Virtual Environment anlegen und starten
    - Order erstellen, in welchem die Scripte und Modelle gespeichert werden sollen (z.B. E:\Daten\LM_Benchmarks)
    - im Windows-Terminal in den neuen Ordner wechseln (z.B. “cd /d E:\Daten\LM_Benchmarks”)
    - Virtuelle Umgebung erstellen (z.B. mit “py -3.11 -m venv venv”)
    - Windows Ausführungsrechte für den aktuellen Benutzer setzen (z.B. über PowerShell als Administrator: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser)
    - Virtuelle Umgebung starten (z.B. mit “./venv/Scripts/activate”)
8. GitHub Benchmark Repository klonen
    - ???
9. Abhängigkeiten installieren
    - alle benötigten Abhängigkeiten in dem neuen Ordner mit “pip install -r requirements.txt“ installieren
10. Huggingface Login und Konfiguration
    - Account unter [Huggingface anlegen](https://huggingface.co/join)
    - einen Access-Token [unter den Accounteinstellungen](https://huggingface.co/settings/tokens) anlegen und kopieren
    - in der Windows Konsole “huggingface-cli login” eingeben
    - Token versteckt (man sieht die Eingabe nicht!) mit einem Rechtsklick einfügen und Enter drücken
    - bei der Abfrage Y eingeben und Enter drücken
    - Huggingface Model-Download Ordner setzen (z.B. über die Konsole mit “setx HF_HOME "E:\Daten\LM_Benchmarks\models_cache"”)


## 📖🐧 Anleitung - Installation der Voraussetzungen (neues Windows 11 WSL2 Linux System)

1. Im BIOS des Mainboards den SVM Mode aktivieren (damit eine Virtualisierung möglich ist)
2. WSL2 (Ubuntu) installieren und konfigurieren
    - in der Konsole folgendes eingeben: “wsl --install -d Ubuntu”
    - in der Konsole folgendes eingeben: “wsl.exe --update”
    - Computer neustarten
    - Windows+S Taste drücken: “Ubuntu” suchen und öffnen
    - Benutzernamen und Passwort setzen
    - Linux Version aktualisieren (z.B. mit “sudo apt update && sudo apt upgrade -y”)
    - Paketinstaller aktualisieren (z.B. mit “sudo apt update”)
    - Python 3.11, Venv, Distutils, Git, Essentials, Nano installieren
        - “sudo apt install -y software-properties-common”
        - “sudo add-apt-repository ppa:deadsnakes/ppa”
        - “sudo apt install -y python3.11 python3.11-venv python3.11-distutils python3.11-dev git build-essential nano”
3. CUDA installieren
    - wie in der Windows Anleitung unter Punkt 4. beschrieben vorgehen (für WSL2 Ubuntu)
    - CUDA-Toolkit direkt unter Ubuntu installieren (z.B. mit “sudo apt install -y nvidia-cuda-toolkit”)
4. Virtual Environment anlegen und starten
    - Order erstellen, in welchem die Scripte und Modelle gespeichert werden sollen
    - im Ubuntu-Terminal in den neuen Ordner wechseln (z.B. “cd ./Daten/LM_Benchmarks”)
    - VENV anlegen mit: “python3.11 -m venv venv”
    - VENV aktivieren mit: “source venv/bin/activate”
5. GitHub Benchmark Repository klonen
    - ???
6. Abhängigkeiten installieren
    - Pip aktualisieren mit “pip install --upgrade pip”
    - alle benötigten Abhängigkeiten in dem neuen Ordner mit “pip install -r requirements.txt“ installieren
7. Huggingface Login und Konfiguration
    - Account unter [Huggingface anlegen](https://huggingface.co/join)
    - einen Access-Token [unter den Accounteinstellungen](https://huggingface.co/settings/tokens) anlegen und kopieren
    - in der Ubuntu Konsole “huggingface-cli login” eingeben
    - Token versteckt (man sieht die Eingabe nicht!) mit einem Rechtsklick einfügen und Enter drücken
    - bei der Abfrage Y eingeben und Enter drücken

---

## ⚙️ Installation (einmalig)

```bash
# 1. Ubuntu Packages installieren
sudo apt update && sudo apt upgrade -y

sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.11 python3.11-venv python3.11-distutils python3.11-dev git build-essential nano
sudo apt install -y nvidia-cuda-toolkit

# 2. Repository klonen
git clone https://github.com/NiklasS1999/lm-evaluation-harness.git
cd lm-evaluation-harness

# 3. Python-Umgebung erstellen
python3.11 -m venv venv
source venv/bin/activate

# 4. Abhängigkeiten installieren
pip install -r requirements_cuda128.txt
pip install -r requirements_freeze.txt
