# 🌿 VitalizeHealth — AI Nutrition Planner

A full-stack health web app combining a **React-style frontend**, a **Flask backend**, and a **scikit-learn Random Forest ML engine** — all in one VS Code project.

---

## 📁 Project Structure

```
vitalizehealth/
│
├── run.py                        ← ✅ START HERE — main entry point
├── requirements.txt              ← Python dependencies
├── .env                          ← API key config (edit this)
│
├── backend/
│   └── app.py                    ← Flask server + REST API routes
│
├── ml_engine/
│   ├── __init__.py
│   └── bmi_diet_model.py         ← 🤖 ML engine (ported from notebook)
│                                    Steps: load data → clean → train RF → predict
│
├── frontend/
│   └── index.html                ← Full frontend (served by Flask)
│
└── static/
    └── charts/                   ← EDA charts auto-generated on startup
        ├── eda_overview.png
        ├── confusion_matrix.png
        └── feature_importance.png
```

---

## ⚡ Setup & Run (Step by Step)

### 1. Open the folder in VS Code

```
File → Open Folder → select the `vitalizehealth` folder
```

---

### 2. Create a virtual environment

Open the **VS Code Terminal** (`Ctrl + `` ` ``) and run:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. (Optional) Add your Anthropic API Key

Open `.env` and replace the placeholder:

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxx
```

> Without a key, the app still works fully using the ML offline plan generator.

---

### 5. Run the app

```bash
python run.py
```

You will see:

```
🔄 Training ML model on startup...
✅ BMI Dataset loaded  — shape: (741, 5)
✅ BMI Dataset cleaned — shape: (741, 9)
✅ EDA chart saved → static/charts/eda_overview.png
✅ Model trained  — Accuracy: 99.xx%
✅ Confusion matrix saved → static/charts/confusion_matrix.png
✅ Feature importance saved → static/charts/feature_importance.png
🚀 Model ready — accuracy: 99.xx%

🚀  VitalizeHealth running at → http://localhost:5000
    Press Ctrl+C to stop
```

---

### 6. Open in browser

```
http://localhost:5000
```

---

## 🔌 API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Serves the frontend |
| POST | `/api/predict` | ML prediction only (BMI → category) |
| POST | `/api/generate-plan` | Full plan (ML + AI if key is set) |
| GET | `/api/charts` | Lists available EDA charts |
| GET | `/api/health` | Server health check |

### Example: `/api/predict`

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 25, "height_cm": 165, "weight_kg": 72}'
```

Response:
```json
{
  "success": true,
  "data": {
    "bmi": 26.45,
    "category": "Overweight",
    "goal": "Lose Weight – Low Calorie & High Protein",
    "focus": ["Vegetables", "Fruits", "Lean Protein", "Dal", "Salad"],
    "avoid": ["Fried Food", "Sugary Drinks", "White Rice"],
    "daily_calories": "1500–2000",
    "tip": "Create a 400–500 kcal daily deficit...",
    "model_accuracy": 99.33
  }
}
```

---

## 🤖 ML Engine Details

File: `ml_engine/bmi_diet_model.py`

| Notebook Step | What it does | Python function |
|---|---|---|
| Step 2 | Generate BMI dataset (741 rows) | `_generate_bmi_dataset()` |
| Step 3 | Clean columns, remove duplicates | `_clean_and_calc()` |
| Step 4 | Recalculate BMI from height/weight | `_clean_and_calc()` |
| Step 5 | EDA visualisations (3 charts) | `_save_eda_charts()` |
| Step 6 | Diet recommendation labels | `DIET_RECOMMENDATIONS` dict |
| Step 7 | Train Random Forest Classifier | `BMIDietModel.train()` |
| Step 8 | Predict diet plan for new person | `BMIDietModel.predict()` |

**BMI Categories (6-class):**
- BMI < 18.5 → Underweight
- 18.5–24.9 → Normal Weight
- 25–29.9   → Overweight
- 30–34.9   → Obese Class 1
- 35–39.9   → Obese Class 2
- ≥ 40      → Obese Class 3

---

## 🛠 Troubleshooting

**`ModuleNotFoundError: No module named 'flask'`**
→ Make sure your virtual environment is activated (`venv\Scripts\activate` on Windows)

**Port 5000 already in use:**
```bash
# Change port in .env:
PORT=8080
# Then re-run: python run.py
```

**App runs but no AI plan (just ML plan):**
→ Add your `ANTHROPIC_API_KEY` to `.env` and restart

---

## 🧪 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, Tailwind CSS, Vanilla JS |
| Backend | Python 3.10+, Flask 3.x |
| ML Engine | scikit-learn (Random Forest), pandas, numpy |
| Charts | matplotlib, seaborn |
| AI | Anthropic Claude API (optional) |
