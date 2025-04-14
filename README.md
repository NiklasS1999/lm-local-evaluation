# 🧠 LM-Local-Evaluation

Mithilfe dieses Repositories können Language Models wie **Qwen**, **Gemma** oder **LLaMA 3** evaluiert sowie untereinander verglichen werden.

Dazu werden je Modell folgende Kategorien gemessen, ausgewertet und grafisch dargestellt:
- Installationsgröße
- Parametergröße
- Latenz
- RAM-Speichernutzung
- GPU-Speichernutzung
- Antwortqualität

Dies dient dem Vergleich von verschiedenen Language Models zur lokalen Nutzung als persönlichen Assistenten.<br>
Dadurch kann herausgefunden werden, ob der Einsatz eines bestimmten Language Models für eine bestimmte Hardware-Konfiguration, beispielsweise ein mobiles Endgeräts, sinnvoll ist.<br>
Je nach Benutzer-Präferenz können hierbei andere Schwerpunkte gesetzt werden und entsprechend eine Auswahl getroffen werden.<br>

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

1. **Python 3.11.0 herunterladen und installieren**
    - Python 3.11.0 von der [offiziellen Webseite](https://www.python.org/downloads/release/python-3110/) herunterladen und installieren
    - Überprüfen, ob die Installation geklappt hat:
    ```bash
    python --version
    ```
2. **Git herunterladen und installieren**
    - Git von der [offiziellen Webseite](https://git-scm.com/downloads/win) herunterladen und installieren
    - Überprüfen, ob die Installation geklappt hat
    ```bash
    git --version
    ```
3. **Git Umgebungsvariable setzen**
    - Windows+S Taste drücken: “Umgebungsvariable bearbeiten” suchen
    - Wähle “Umgebungsvariablen”
    - Unter “Systemvariablen”, zu “Path” gehen und “Bearbeiten” auswählen
    - “Neu” anklicken und den Installationspfad von Git hinzufügen (z.B. C:\Program Files\Git\bin)
    - Auf “OK” klicken und alle Fenster schließen
    - Terminal neu starten und testen, ob git erkannt wird
    ```bash
    git --version
    ```
4. **CUDA installieren**
    - Überprüfen, ob die eigene [Nvidia GPU CUDA fähig](https://developer.nvidia.com/cuda-gpus) ist
    - Neueste Nvidia Grafikkarten-Softwareversion über z.B. Geforce Experience installieren
    - CUDA Toolkit für Windows [herunterladen](https://developer.nvidia.com/cuda-downloads) und installieren
    - Computer neustarten
5. **cuDNN-Bibliothek installieren**
    - cuDNN-Bibliothek [herunterladen](https://developer.nvidia.com/cudnn-downloads) (z.B. Windows - x86_64 - Tarball - 12)
    - Heruntergeladene Dateien entpacken
    - Einzelne Dateien (bin, include, lib) in die Verzeichnisse des CUDA-Installationsordners kopieren (z.B. Default: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8)
6. **CUDA Umgebungsvariable setzen**
    - Windows+S Taste drücken: “Umgebungsvariable bearbeiten” suchen
    - Wähle “Umgebungsvariablen”
    - Unter “Systemvariablen”, zu “Path” gehen und “Bearbeiten” auswählen
    - “Neu” anklicken und den Installationspfad von CUDA hinzufügen (z.B. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\)
    - Auf “OK” klicken und alle Fenster schließen
    - Terminal neu starten und testen, ob CUDA erkannt wird
    ```bash
    nvcc --version
    ```
7. **lm-local-evaluation Repository klonen**
    - Order erstellen, in welchem die Scripte und Modelle gespeichert werden sollen
    ```bash
    mkdir LM_Benchmarks
    ```

    - In den neuen Ordner wechseln
    ```bash
    cd .\LM_Benchmarks
    ```

    - GitHub-Repository klonen
    ```bash
    git clone https://github.com/NiklasS1999/lm-local-evaluation.git
    ```

    - GitHub-Repository öffnen
    ```bash
    cd lm-local-evaluation
    ```
8. **Virtual Environment anlegen und starten**
    - Virtuelle Umgebung erstellen
    ```bash
    python -m venv venv
    ```

    - Windows Ausführungsrechte für den aktuellen Benutzer setzen (über PowerShell als Administrator)
    ```bash
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```

    - Virtuelle Umgebung starten
    ```bash
    .\venv\Scripts\activate
    ```
9. **Modulabhängigkeiten installieren**
    - Pip aktualisieren
    ```bash
    python.exe -m pip install --upgrade pip
    ```

    - Benötigte Abhängigkeiten installieren
    ```bash
    pip install -r requirements.txt
    ```
10. **Huggingface Login und Konfiguration**
    - Account unter [Huggingface anlegen](https://huggingface.co/join)
    - Access-Token [unter den Accounteinstellungen](https://huggingface.co/settings/tokens) anlegen und kopieren
    - Über Windows-Konsole einloggen
    ```bash
    huggingface-cli login
    ```

    - Token versteckt (man sieht die Eingabe nicht!) mit einem Rechtsklick einfügen und Enter drücken
    - Bei der Rückfrage Y eingeben und Enter drücken
    - Huggingface Model-Download Ordner setzen
    ```bash
    setx HF_HOME "E:\Daten\LM_Benchmarks\models_cache"
    ```


## 🐧 Anleitung Installation (neues Windows 11 WSL2 Linux System)

1. **SVM-Mode aktivieren**
    - Computer neustarten
    - BIOS-Zugriffstaste während des Bootvorgangs klicken (meist F1 / F2 / Entf)
    - Im BIOS den SVM-Mode aktivieren (damit eine Virtualisierung möglich ist)
2. **WSL2 installieren und konfigurieren**
    - WSL (Ubuntu) installieren
    ```bash
    wsl --install -d Ubuntu
    ```

    - WSL (Ubuntu) updaten
    ```bash
    wsl.exe --update
    ```

    - Computer neustarten
    - Windows+S Taste drücken: “Ubuntu” suchen und öffnen
    - Benutzernamen und Passwort setzen
    - Linux Version aktualisieren
    ```bash
    sudo apt update && sudo apt upgrade -y
    ```

    - Paketinstaller aktualisieren
    ```bash
    sudo apt update
    ```

    - Python 3.11, Venv, Distutils, Git, Essentials, Nano installieren
    ```bash
    sudo apt install -y software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install -y python3.11 python3.11-venv python3.11-distutils python3.11-dev git build-essential nano
    ```
3. **CUDA installieren**
    - Wie in der oberen Windows Anleitung unter Punkt 4. beschrieben vorgehen (für WSL2 Ubuntu)
    - CUDA-Toolkit direkt unter Ubuntu installieren
    ```bash
    sudo apt install -y nvidia-cuda-toolkit
    ```
4. **lm-local-evaluation Repository klonen**
    - Order erstellen, in welchem die Scripte und Modelle gespeichert werden sollen
    ```bash
    mkdir LM_Benchmarks
    ```

    - In den neuen Ordner wechseln
    ```bash
    cd ./LM_Benchmarks
    ```

    - GitHub-Repository klonen
    ```bash
    git clone https://github.com/NiklasS1999/lm-local-evaluation.git
    ```

    - GitHub-Repository öffnen
    ```bash
    cd lm-local-evaluation
    ```
5. **Virtual Environment anlegen und starten**
    - Virtuelle Umgebung erstellen
    ```bash
    python3.11 -m venv venv
    ```

    - Virtuelle Umgebung starten
    ```bash
    source venv/bin/activate
    ```
6. **Abhängigkeiten installieren**
    - Pip aktualisieren
    ```bash
    pip install --upgrade pip
    ```

    - Benötigte Abhängigkeiten installieren
    ```bash
    pip install -r requirements.txt
    ```
7. **Huggingface Login und Konfiguration**
    - Account unter [Huggingface anlegen](https://huggingface.co/join)
    - Access-Token [unter den Accounteinstellungen](https://huggingface.co/settings/tokens) anlegen und kopieren
    - Über Ubuntu-Konsole einloggen
    ```bash
    huggingface-cli login
    ```

    - Token versteckt (man sieht die Eingabe nicht!) mit einem Rechtsklick einfügen und Enter drücken
    - Bei der Abfrage Y eingeben und Enter drücken

---

## ⚙️ Konfiguration

Die Benchmarks können mithilfe von verschiedenen Einstellungen ausgeführt werden. Diese wirken sich drastisch auf die Ergebnisqualität sowie die Ausführungszeit aus.<br>
Die Konfiguration kann mithilfe der Datei global_config.py angepasst werden, diese ist ausführlich kommentiert und zeigt mögliche Anpassungen auf.<br>

Neben der globalen Konfiguration kann auch je Benchmark eine eigene Konfiguration gesetzt werden.<br>
Dies geschieht in den einzelnen Benchmark-Scripten.<br>

Bearbeiten der Globalen Konfiguration:
```bash
cd ./benchmarks # in den Benchmark-Ordner wechseln, wenn noch nicht getan
nano global_config.py # nur unter Linux
notepad global_config.py # nur unter Windows
```

Bearbeiten einer lokalen Konfiguration:
```bash
cd ./benchmarks # in den Benchmark-Ordner wechseln, wenn noch nicht getan
nano model_latency.py # nur unter Linux
notepad model_latency.py # nur unter Windows
```

---

## 🚀 Ausführung

Es können entweder einzelne Benchmarks oder alle automatisiert nacheinander ausgeführt werden.

Liste an vorhandenen Benchmarks:

| Benchmark-Skript          | Beschreibung                                                                                      |
|------------------------------|-------------------------------------------------------------------------------------------------------|
| `model_installationsize.py`  | Misst die **Installationsgröße** des Modells.                                                        |
| `model_parametersize.py`     | Misst die **Größe der Modellparameter** – als Anhaltspunkt für den geschätzten Rechenaufwand.        |
| `model_latency.py`           | Misst die **Latenz**: Tokenizer-, Modell-, Antwort- und Gesamtlatenz.                                |
| `model_memoryusage.py`       | Misst die **RAM- und GPU-Speichernutzung**: Cold-Start, reine Inferenz, Gesamtverbrauch.             |
| `model_responsequality.py`   | Bewertet die **Antwortqualität** anhand vordefinierter Benchmarks (z. B. MMLU, HellaSwag, GSM8K).    |

Ausführung eines einzelnen Benchmarks:
```bash
cd ./benchmarks # in den Benchmark-Ordner wechseln, wenn noch nicht getan
python model_installationsize.py
```

Ausführung aller Benchmarks inkl. Auswertung:
```bash
cd ./benchmarks # in den Benchmark-Ordner wechseln, wenn noch nicht getan
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
cd ./benchmarks # in den Benchmark-Ordner wechseln, wenn noch nicht getan
python benchmark_overview.py
```
Es werden dann jedoch nur die Ergebnisse dargestellt, welche auch vorhanden sind (wo bisher Benchmarks durchgeführt wurden).

---

## 💡 Tipps

Manche Antwortqualität-Benchmarks sind nur unter einem Linux-Betriebssystem ausführbar und werfen unter der Ausführung in Windows einen Fehler (z.B. HumanEval).<br>

Da das Projekt vollständig mit Python umgesetzt wurde, ist der Code betriebssystemunabhängig, jedoch wurde der Code nur unter Windows 11 und WSL2-Linux (Ubuntu) getestet.<br>

Die Ausführungszeit ist je nach Konfiguration sehr unterschiedlich, in der von mir bereitgestellten Konfiguration dauerte die vollständige Auswertung insgesamt 19 Stunden mit einer GeForce RTX 3060.<br>
Dies ist vorallem auf den Antwortqualität-Benchmark HumanEval zurückzuführen. Dieser dauerte alleine insgesamt rund 15 Stunden für alle drei Language Modelle.<br>

Je nach Language Modell kann es sein, das in der Konsole Fehler auftreten. Besonders bei den Antwortqualität-Benchmarks kann dies aufgrund von Inkompatibilität vorkommen.<br>
Getestet wurden lediglich die drei in der Konfiguration hinterlegten Modelle.<br>

Wenn ein anderes Modell getestet werden soll, muss in der Konfiguration unter "models": der Huggingface-Pfad zum Language Model angegeben werden.<br>
Dementsprechend können nur Language Models gebenchmarkt werden, die auch in Huggignface öffentlich zugänglich sind.<br>

Bei der ersten Ausführung eines Benchmarks werden die entsprechend in der Globalen Konfiguration hinterlegten Modelle von Huggingface heruntergeladen.<br>
Dies kann je nach Modellgröße und Internetverbindung mehrere Stunden in Anspruch nehmen.<br>


