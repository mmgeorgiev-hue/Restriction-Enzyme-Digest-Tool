
### S.L.I.C.E. (Streamlit)

Interactive tool to rank restriction enzymes against a FASTA genome, with T-DNA border checks and simulated insertions.

## Pre-reqs

- **Python 3.10 or newer** (recommended: 3.11+)
- A terminal (`Terminal` on macOS, `PowerShell` or `cmd` on Windows)

## Run it locally

Repository on GitHub: [mmgeorgiev-hue/Restriction-Enzyme-Digest-Tool](https://github.com/mmgeorgiev-hue/Restriction-Enzyme-Digest-Tool).

**Important:** `pip install` and `streamlit run` must be **two separate commands**. Also run them **from inside this project folder** (where `requirements.txt` and `app.py` live), not from your home directory (`~`).

### 1. Get the code

**Clone from GitHub (recommended):**

```bash
git clone https://github.com/mmgeorgiev-hue/Restriction-Enzyme-Digest-Tool.git
cd Restriction-Enzyme-Digest-Tool
```

**Or** download ZIP from GitHub (**Code** ▶ **Download ZIP**), unzip it, then:

```bash
cd path/to/Restriction-Enzyme-Digest-Tool
```

(On macOS the folder name must match wherever you extracted it.)

### 2. (Recommended) Use a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (cmd)
```

### 3. Install dependencies

Run **one** of these (use the same `python` you use for the app):

```bash
python3 -m pip install -r requirements.txt
```

### 4. Start the app

This must be a **separate** command after install (not combined with `pip`):

```bash
streamlit run app.py
```

Your browser should open at `http://localhost:8501`. If not, copy the URL printed in the terminal.

Stop the server with `Ctrl+C`.


