# MathGrade Bundle Generator (Web)

This project provides a small web app to upload:
1) the official **exam + solution PDF** and  
2) a student's **LaTeX results (.tex)**

The Python backend generates a **bundled PDF** (questions + solution + student answers) and returns it for download.

---

## Requirements

### 1) Python
- Python 3.10+ recommended
- Create and use a virtual environment (`.venv`)

### 2) LaTeX (XeLaTeX)
The backend compiles LaTeX using `xelatex`, so you must have:

- **MiKTeX** (Windows) or **TeX Live** (Linux/macOS)
- `xelatex` available in your PATH

Quick check:
```bash
xelatex --version
```

If that command fails, install MiKTeX/TeX Live and ensure the binaries are in PATH.

> Tip (MiKTeX): use MiKTeX Console to enable on-the-fly package install or install required packages ahead of time.

### 3) Python dependencies
Backend uses:
- `fastapi`
- `uvicorn`
- `python-multipart` (for file uploads)

Install them:
```bash
pip install fastapi uvicorn python-multipart
```

---

## Project Structure (important files)

- `server.py` — FastAPI backend
- `web/index.html` — frontend upload page (vanilla HTML/JS)
- `grader/` — core grading / bundling logic
- `build/` — local build output (optional; server uses temp dirs per request)

---

## Run Locally (Development)

### Step 1 — Create venv and install deps
From the project root:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn python-multipart
```

**Windows (CMD):**
```bat
python -m venv .venv
.\.venv\Scripts\activate
pip install fastapi uvicorn python-multipart
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn python-multipart
```

### Step 2 — Start the backend server
From the project root (same folder as `server.py`):
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Verify it’s running:
- Open: `http://localhost:8000/health`
- You should see: `{"ok": true}`

### Step 3 — Start the frontend page
In a **second terminal**, serve the `web/` folder.

Option A (recommended):
```bash
cd web
python -m http.server 5173
```

Then open:
- `http://localhost:5173/`

Option B (serve web folder from project root):
```bash
python -m http.server 5173 --directory web
```

Then open:
- `http://localhost:5173/`

### Step 4 — Generate a bundle
1. Choose the official reference PDF (exam+solution)
2. Choose the student `.tex`
3. Click **Generate PDF**
4. Your browser should download `qa_bundle.pdf`

---

## API

### POST `/api/generate`
**Form-data fields**
- `reference_pdf` (file, .pdf)
- `student_tex` (file, .tex or .txt)

**Response**
- On success: `application/pdf` download
- On failure: `application/json` with fields `{ "error": "...", "detail": "..." }`

---

## Common Issues / Troubleshooting

### 1) “Failed to fetch” in the browser
Usually means the backend returned an error.
- Check the backend console (uvicorn terminal) for the traceback
- Confirm `http://localhost:8000/health` works

### 2) `xelatex` not found
Install MiKTeX / TeX Live and ensure `xelatex` is in PATH:
```bash
xelatex --version
```

### 3) Missing LaTeX packages / font errors
If XeLaTeX fails due to missing packages, install them via MiKTeX Console or TeX Live package manager.
If you see a font-related issue, ensure Arial exists or adjust the hardcoded font in `server.py`.

### 4) Windows Defender / permissions / temp folder issues
The server writes to a temp folder for each request (e.g. `%TEMP%\mathgrade_*`).
If you suspect permissions issues, try running the terminal as admin once to confirm.

---

## Notes (Security)
This service compiles user-supplied LaTeX. For real deployments:
- run in a sandbox/container
- keep shell-escape disabled (default)
- consider file size limits + request timeouts
- restrict CORS to your domain
