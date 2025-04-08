# 🧠 LM-Local-Evaluation

Mithilfe dieses Repositories können Language Models wie **Qwen**, **Gemma** oder **LLaMA 3** evaluiert sowie untereinander verglichen werden.

Dazu werden je Modell folgende Kategorien gemessen, ausgewertet und grafisch dargestellt:
- Installationsgröße
- Latenz
- RAM-Speichernutzung
- GPU-Speichernutzung
- Parametergröße
- Antwortqualität

Dies dient dem Vergleich von verschiedenen Language Models zur lokalen Nutzung als persönlichen Assistenten.<br>
Dadurch kann herausgefunden werden, ob der Einsatz eines bestimmten Language Models für eine bestimmte Hardware-Konfiguration, beispielsweise ein mobiles Endgeräts, sinnvoll ist.<br>
Je nach Benutzer-Präferenz können hierbei andere Schwerpunkte gesetzt werden und entsprechend eine Auswahl getroffen werden. Dabei soll dieses Repository unterstützen.<br>

Dieses Repository enthält eine lokal integrierte und angepasste Version von [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) für die Ermittlung der Antwortqualität, daher muss kein separates GitHub-Repo geklont werden. Alle Änderungen sind enthalten und einsatzbereit.

---

## 🔧 Voraussetzungen

- Python 3.11
- Git
- Windows 11 oder Linux
- NVIDIA GPU mit CUDA 12.x (z.B. GeForce RTX 3060)
- NVIDIA CUDA-Toolkit + cuDNN-Bibliothek
- Huggingface Account & Access Token
- Virtual Environment (empfohlen)

---

## 🪟 Anleitung Installation (neues Windows 11 System)

1. Python 3.11.0 herunterladen und installieren
    - Python 3.11.0 von der [offiziellen Webseite](https://www.python.org/downloads/release/python-3110/) herunterladen und installieren
    - Überprüfen, ob die Installation geklappt hat:
```bash
python --version
```
2. Git herunterladen und installieren
    - Git von der [offiziellen Webseite](https://git-scm.com/downloads/win) herunterladen und installieren
    - Überprüfen, ob die Installation geklappt hat:
```bash
git --version
```
3. Git Umgebungsvariable setzen
    - Windows+S Taste drücken: “Umgebungsvariable bearbeiten” suchen
    - Wähle “Umgebungsvariablen”
    - unter “Systemvariablen”, zu “Path” gehen und “Bearbeiten” auswählen
    - “Neu” anklicken und den Installationspfad von Git hinzufügen (z.B. C:\Program Files\Git\bin)
    - auf “OK” klicken und alle Fenster schließen
    - Terminal neu starten und testen, ob git erkannt wird:
```bash
git --version
```
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
    - Terminal neu starten und testen, ob CUDA erkannt wird:
```bash
nvcc --version
```
7. lm-local-evaluation Repository klonen
    - Order erstellen, in welchem die Scripte und Modelle gespeichert werden sollen:
    ```bash
    mkdir E:\Daten\LM_Benchmarks
    ```
    
    - im Windows-Terminal in den neuen Ordner wechseln:
    ```bash
    cd /d E:\Daten\LM_Benchmarks
    ```

    - GitHub-Repository klonen:
    ```bash
    git clone https://github.com/NiklasS1999/lm-local-evaluation.git
    ```

    - Repository öffnen:
    ```bash
    cd lm-local-evaluation
    ```
8. Virtual Environment anlegen und starten
    - Virtuelle Umgebung erstellen (z.B. mit “python -m venv venv”)
    - Windows Ausführungsrechte für den aktuellen Benutzer setzen (z.B. über PowerShell als Administrator: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser)
    - Virtuelle Umgebung starten (z.B. mit “venv/Scripts/activate”)
9. Abhängigkeiten installieren
    - Pip aktualisieren mit "python.exe -m pip install --upgrade pip"
    - alle benötigten Abhängigkeiten mit “pip install -r requirements.txt“ installieren
11. Huggingface Login und Konfiguration
    - Account unter [Huggingface anlegen](https://huggingface.co/join)
    - einen Access-Token [unter den Accounteinstellungen](https://huggingface.co/settings/tokens) anlegen und kopieren
    - in der Windows Konsole “huggingface-cli login” eingeben
    - Token versteckt (man sieht die Eingabe nicht!) mit einem Rechtsklick einfügen und Enter drücken
    - bei der Abfrage Y eingeben und Enter drücken
    - Huggingface Model-Download Ordner setzen (z.B. über die Konsole mit “setx HF_HOME "E:\Daten\LM_Benchmarks\models_cache"”)


## 🐧 Anleitung Installation (neues Windows 11 WSL2 Linux System)

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
4. lm-local-evaluation Repository klonen
    - Order erstellen, in welchem die Scripte und Modelle gespeichert werden sollen (z.B. "mkdir LM_Benchmarks")
    - im Ubuntu-Terminal in den neuen Ordner wechseln (z.B. “cd ./Daten/LM_Benchmarks”)
    - Repository klonen mit: “git clone https://github.com/NiklasS1999/lm-local-evaluation.git”
    - Repository öffnen mit: “cd lm-local-evaluation”
6. Virtual Environment anlegen und starten
    - VENV anlegen mit: “python3.11 -m venv venv”
    - VENV aktivieren mit: “source venv/bin/activate”
7. Abhängigkeiten installieren
    - Pip aktualisieren mit "python.exe -m pip install --upgrade pip"
    - alle benötigten Abhängigkeiten mit “pip install -r requirements.txt“ installieren
8. Huggingface Login und Konfiguration
    - Account unter [Huggingface anlegen](https://huggingface.co/join)
    - einen Access-Token [unter den Accounteinstellungen](https://huggingface.co/settings/tokens) anlegen und kopieren
    - in der Ubuntu Konsole “huggingface-cli login” eingeben
    - Token versteckt (man sieht die Eingabe nicht!) mit einem Rechtsklick einfügen und Enter drücken
    - bei der Abfrage Y eingeben und Enter drücken

---

## ⚙️ Konfiguration

Die Benchmarks können mithilfe von verschiedenen Einstellungen ausgeführt werden. Diese wirken sich drastisch auf die Ergebnisqualität sowie die Ausführungszeit aus.<br>
Die Konfiguration kann mithilfe der Datei global_config.py angepasst werden, diese ist ausführlich kommentiert und zeigt mögliche Anpassungen auf.<br>

Neben der globalen Konfiguration kann auch je Benchmark eine eigene Konfiguration gesetzt werden.<br>
Dies geschieht in den einzelnen Benchmark-Scripten.<br>

Bearbeiten der Globalen Konfiguration:
```bash
cd ./benchmarks # in den Benchmark-Ordner wechseln
nano global_config.py # nur unter Linux
```
Bearbeiten einer lokalen Konfiguration:
```bash
cd ./benchmarks # in den Benchmark-Ordner wechseln
nano model_latency.py # nur unter Linux
```

---

## 🚀 Ausführung

Es können entweder einzelne Benchmarks oder alle automatisiert nacheinander ausgeführt werden.

Liste an vorhandenen Benchmarks:

| Benchmark-Skript          | Beschreibung                                                                                      |
|------------------------------|-------------------------------------------------------------------------------------------------------|
| `model_installationsize.py`  | Misst die **Installationsgröße** des Modells.                                                        |
| `model_latency.py`           | Misst die **Latenz**: Tokenizer-, Modell-, Antwort- und Gesamtlatenz.                                |
| `model_memoryusage.py`       | Misst die **RAM- und GPU-Speichernutzung**: Cold-Start, reine Inferenz, Gesamtverbrauch.             |
| `model_parametersize.py`     | Misst die **Größe der Modellparameter** – als Anhaltspunkt für den geschätzten Rechenaufwand.        |
| `model_responsequality.py`   | Bewertet die **Antwortqualität** anhand vordefinierter Benchmarks (z. B. MMLU, HellaSwag, GSM8K).    |

Ausführung eines einzelnen Benchmarks:
```bash
cd ./benchmarks # in den Benchmark-Ordner wechseln
python model_installationsize.py
```
Ausführung aller Benchmarks inkl. Auswertung:
```bash
cd ./benchmarks # in den Benchmark-Ordner wechseln
python run_all_benchmarks.py
```

---

## 📊 Ergebnisse

Während der Ausführung eines Benchmarks werden die Ergebnisse in der Konsole ausgegeben. Außerdem werden die Ergebnisse am Ende des Benchmarks entweder in eine .csv oder eine .json Datei mit dem Namen des Benchmarks sowie einem Zeitstempel geschrieben und unter ./results/Modell_Name abgespeichert.<br>

Wenn alle Benchmarks mithilfe von run_all_benchmarks.py ausgeführt werden, wird am Ende eine Auswertung durchgeführt. Durch diese wird eine Übersicht der wichtigsten Ergebnisse aller durchgeführten Benchmarks in der Konsole ausgegeben, sowie die wichtigsten Ergebnisse unter ./results/benchmark_overview.csv abgespeichert.<br>

Außerdem werden im Rahmen der Auswertung verschiedene Plots (Balkendiagramme) zum Vergleich der evaluierten Language Models generiert und unter ./results/result_plots abgespeichert.<br>

Des Weiteren wird eine Auswertung bezüglich der jeweiligen Vor- und Nachteile eines Modells in Bezug auf den Vergleich zu den anderen Modellen in der Konsole ausgegeben. Diese Ergebnisse werden außerdem in einer Textdatei unter ./results/model_advantages.txt abgespeichert.<br>

Die Auswertung kann auch manuell über den folgenden Befehl erstellt werden:
```bash
python benchmark_overview.py
```
Es werden dann jedoch nur die Ergebnisse dargestellt, welche auch vorhanden sind (wo bisher Benchmarks durchgeführt wurden).

---

## 💡 Tipps

Manche Antwortqualität-Benchmarks sind nur unter einem Linux-Betriebssystem ausführbar und werfen unter der Ausführung in Windows einen Fehler (z.B. HumanEval).<br>

Da das Projekt vollständig mit Python umgesetzt wurde, ist der Code betriebssystemunabhängig, jedoch wurde der Code nur unter Windows 11 und WSL2-Linux (Ubuntu) getestet.<br>

Die Ausführungszeit ist je nach Konfiguration sehr unterscheidlich, in der von mir bereitgestellten Konfiguration dauerte die vollständige Auswertung insgesamt 19 Stunden mit einer GeForce RTX 3060.<br>
Dies ist vorallem auf den Antwortqualität-Benchmark HumanEval zurückzuführen. Dieser dauerte alleine rund 15 Stunden für alle drei Language Modelle.<br>

Je nach Language Modell kann es sein, das in der Konsole Fehler auftreten. Besonders bei den Antwortqualität-Benchmarks kann dies aufgrund von Inkompatibilität vorkommen.<br>
Getestet wurden lediglich die drei in der Konfiguration hinterlegten Modelle.<br>

Wenn ein anderes Modell getestet werden soll, muss in der Konfiguration unter "models": der Huggingface-Pfad zum Language Model angegeben werden.<br>
Dementsprechend können nur Language Models gebenchmarkt werden, die auch in Huggignface öffentlich zugänglich sind.<br>

Bei der ersten Ausführung eines Benchmarks werden die entsprechend in der Globalen konfiguration hinterlegten Modelle von Huggingface heruntergeladen.<br>
Dies kann je nach Modellgröße und Internetverbindung mehrere Stunden in Anspruch nehmen.<br>


