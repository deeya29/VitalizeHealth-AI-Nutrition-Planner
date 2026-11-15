"""
backend/app.py
==============
VitalizeHealth — Flask Backend
Serves the frontend HTML and exposes REST API endpoints that call the ML engine.

Routes:
  GET  /                  → serves frontend/index.html
  POST /api/predict       → ML prediction (BMI category + diet plan)
  POST /api/generate-plan → Full AI diet plan (via Anthropic API)
  GET  /api/charts        → list of EDA chart filenames
  GET  /api/health        → health check
"""

import os
import sys
import json

# ── add project root to path so ml_engine is importable ─────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, jsonify, request, send_from_directory, render_template_string
from flask_cors import CORS

from ml_engine.bmi_diet_model import get_model, DIET_RECOMMENDATIONS

# ── Anthropic SDK (optional — gracefully degrades if not installed) ─
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.join(ROOT, "static"),
    template_folder=os.path.join(ROOT, "frontend"),
)
CORS(app)  # allow frontend on any port to call backend

FRONTEND_DIR = os.path.join(ROOT, "frontend")
CHARTS_DIR   = os.path.join(ROOT, "static", "charts")

# ─────────────────────────────────────────────────────────────────
# STARTUP — train the ML model once when Flask starts
# ─────────────────────────────────────────────────────────────────
model = get_model()


# ─────────────────────────────────────────────────────────────────
# ROUTE: Serve Frontend
# ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    html_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


# ─────────────────────────────────────────────────────────────────
# ROUTE: ML Prediction
# POST /api/predict
# Body: { age, height_cm, weight_kg }
# ─────────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Validate inputs
    try:
        age       = int(data["age"])
        height_cm = float(data["height_cm"])
        weight_kg = float(data["weight_kg"])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    if not (10 <= age <= 100):
        return jsonify({"error": "Age must be between 10 and 100"}), 400
    if not (100 <= height_cm <= 250):
        return jsonify({"error": "Height must be between 100 and 250 cm"}), 400
    if not (20 <= weight_kg <= 300):
        return jsonify({"error": "Weight must be between 20 and 300 kg"}), 400

    result = model.predict(age, height_cm, weight_kg)
    return jsonify({"success": True, "data": result})


# ─────────────────────────────────────────────────────────────────
# ROUTE: Generate Full AI Plan
# POST /api/generate-plan
# Body: { name, age, gender, height_cm, weight_kg, activity,
#         goal, diet, conditions }
# ─────────────────────────────────────────────────────────────────
@app.route("/api/generate-plan", methods=["POST"])
def generate_plan():
    data = request.get_json(force=True)

    # Required fields
    required = ["name", "age", "gender", "height_cm", "weight_kg", "activity", "goal", "diet"]
    for field in required:
        if not data.get(field):
            return jsonify({"error": f"Missing required field: {field}"}), 400

    try:
        age       = int(data["age"])
        height_cm = float(data["height_cm"])
        weight_kg = float(data["weight_kg"])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid numeric input: {e}"}), 400

    name       = data["name"].strip()
    gender     = data["gender"]
    activity   = data["activity"]
    goal       = data["goal"]
    diet       = data["diet"]
    conditions = data.get("conditions", "").strip()

    # ── Step 1: ML Prediction ─────────────────────────────────────
    ml_result = model.predict(age, height_cm, weight_kg)
    bmi           = ml_result["bmi"]
    bmi_category  = ml_result["category"]
    ml_plan       = DIET_RECOMMENDATIONS[bmi_category]

    # ── Step 2: TDEE Calculation ──────────────────────────────────
    if gender == "Female":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5

    activity_mult = {
        "Sedentary":   1.2,
        "Lightly":     1.375,
        "Moderately":  1.55,
        "Very":        1.725,
        "Extra":       1.9,
    }
    mult_key = next((k for k in activity_mult if activity.startswith(k)), "Lightly")
    tdee = int(bmr * activity_mult[mult_key])

    if goal == "Lose Weight" or "Obese" in bmi_category or bmi_category == "Overweight":
        tdee -= 400
    if goal == "Gain Muscle" or bmi_category == "Underweight":
        tdee += 300

    # Clamp to ML range
    cal_str = ml_result["daily_calories"].replace(" (Doctor Only)", "").replace(" kcal", "")
    if "–" in cal_str:
        cal_min, cal_max = map(int, cal_str.split("–"))
        tdee = max(cal_min, min(tdee, cal_max))

    protein = int(weight_kg * (2.0 if goal == "Gain Muscle" else 1.6))
    water   = round(weight_kg * 0.033, 1)

    # ── Step 3: Build Claude prompt using ML output ───────────────
    prompt = f"""You are a certified clinical nutritionist and dietitian AI.
A BMI-based ML model (Random Forest, 6-class) has already classified this patient.
Your job is to elaborate it into a beautifully detailed, practical meal plan.

ML Model Output:
- BMI: {bmi} → Category: "{bmi_category}"
- ML Goal: {ml_result['goal']}
- ML Focus Foods: {', '.join(ml_result['focus'])}
- ML Avoid Foods: {', '.join(ml_result['avoid'])}
- ML Calorie Range: {ml_result['daily_calories']} kcal/day
- Model Accuracy: {ml_result['model_accuracy']}%

Patient Profile:
- Name: {name} | Age: {age} | Gender: {gender}
- Height: {height_cm}cm | Weight: {weight_kg}kg
- Activity: {activity} | Goal: {goal}
- Dietary Preference: {diet}
- Health Conditions: {conditions or 'None'}
- TDEE Target: {tdee} kcal/day
- Protein Target: {protein}g/day
- Water Target: {water}L/day

Provide:
1. PERSONALIZED ASSESSMENT (2–3 sentences using ML BMI classification)
2. DAILY MEAL SCHEDULE (honour ML focus/avoid foods, India-specific):
   🌅 Breakfast (7:00–8:00 AM)
   🌞 Mid-Morning Snack (10:30 AM)
   ☀️ Lunch (1:00–2:00 PM)
   🌆 Evening Snack (4:30 PM)
   🌙 Dinner (7:30–8:30 PM)
3. MACRO BREAKDOWN (carbs/protein/fat in grams)
4. TOP 5 RECOMMENDED FOODS (from ML focus list with reasons)
5. FOODS TO AVOID (from ML avoid list with reasons)
6. KEY LIFESTYLE TIPS (3 tips for "{bmi_category}" category)

Be practical, India-specific, encouraging. Use emojis and clear section headers."""

    # ── Step 4: Call Anthropic API ────────────────────────────────
    api_key  = os.environ.get("ANTHROPIC_API_KEY", "")
    ai_plan  = None
    ai_error = None

    if api_key and ANTHROPIC_AVAILABLE:
        try:
            client   = anthropic.Anthropic(api_key=api_key)
            message  = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1600,
                messages=[{"role": "user", "content": prompt}],
            )
            ai_plan = message.content[0].text
        except Exception as e:
            ai_error = str(e)
            ai_plan  = None

    # ── Fallback: generate plan locally using ML recommendations ──
    if not ai_plan:
        ai_plan = _local_plan(
            name, age, gender, bmi, bmi_category, ml_plan,
            goal, diet, tdee, protein, water, conditions
        )

    return jsonify({
        "success":   True,
        "ai_plan":   ai_plan,
        "ml_result": ml_result,
        "calories":  tdee,
        "protein":   protein,
        "water":     water,
        "ai_used":   bool(api_key and ANTHROPIC_AVAILABLE and not ai_error),
        "ai_error":  ai_error,
    })


# ─────────────────────────────────────────────────────────────────
# ROUTE: List EDA Charts
# GET /api/charts
# ─────────────────────────────────────────────────────────────────
@app.route("/api/charts")
def list_charts():
    charts = []
    if os.path.isdir(CHARTS_DIR):
        charts = [f for f in os.listdir(CHARTS_DIR) if f.endswith(".png")]
    return jsonify({"charts": charts})


# ─────────────────────────────────────────────────────────────────
# ROUTE: Health Check
# GET /api/health
# ─────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({
        "status":           "ok",
        "model_trained":    model.trained,
        "model_accuracy":   model.accuracy,
        "anthropic_sdk":    ANTHROPIC_AVAILABLE,
        "api_key_set":      bool(os.environ.get("ANTHROPIC_API_KEY")),
    })


# ─────────────────────────────────────────────────────────────────
# Local Plan Generator — ML-powered offline fallback
# ─────────────────────────────────────────────────────────────────
def _local_plan(name, age, gender, bmi, bmi_category, ml_plan,
                goal, diet, calories, protein, water, conditions):
    is_veg   = "veg" in diet.lower() or "vegan" in diet.lower()
    is_vegan = "vegan" in diet.lower()

    breakfast = (
        "Moong Dal Chilla (2 pcs) + Coconut Chutney + Black Coffee" if is_vegan else
        "Oats Upma + Paneer Bhurji + Green Tea"                       if is_veg  else
        "Scrambled Eggs (2) + Whole Wheat Toast + Green Tea"
    )
    lunch = (
        "Rajma/Chole + Brown Rice + Cucumber-Tomato Salad"              if is_vegan else
        "Paneer Bhurji + 2 Wheat Rotis + Dal + Salad"                   if is_veg  else
        "Grilled Chicken/Fish (150g) + Brown Rice + Dal + Salad"
    )
    dinner = (
        "Dal Tadka + 2 Rotis + Stir-fried Vegetables"                   if is_vegan else
        "Palak Paneer + 1 Roti + Raita + Salad"                         if is_veg  else
        "Baked Fish/Egg Curry + Salad + 1 Small Roti"
    )
    snack = (
        "Roasted Chana (30g) + 1 Fruit"                                  if is_vegan else
        "Greek Yogurt (150g) + Mixed Nuts (15g)"                         if is_veg  else
        "Boiled Eggs (2) + 1 Fruit"
    )

    avoid_lines = "\n".join(
        f"{i+1}. 🚫 {item}" for i, item in enumerate(ml_plan["avoid"])
    )
    focus_lines = "\n".join(
        f"{i+1}. ✅ {item}" for i, item in enumerate(ml_plan["focus"][:5])
    )

    return f"""🤖 ML MODEL OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━
BMI: {bmi}  →  Category: {bmi_category}  (6-class Random Forest)
ML Goal: {ml_plan['goal']}
Calorie Range: {ml_plan['daily_calories']} kcal/day
━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ PERSONALIZED ASSESSMENT

{name}, your BMI of {bmi} classifies you as "{bmi_category}". {ml_plan['tip']} Your plan is calibrated to {calories} kcal/day with {protein}g protein and {water}L water.{' Note: ' + conditions + ' considered.' if conditions else ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━

🌅 DAILY MEAL SCHEDULE

🌅 Breakfast (7:00–8:00 AM)
• {breakfast}
• Approx: {round(calories * 0.22)} kcal

🌞 Mid-Morning Snack (10:30 AM)
• {snack}
• Approx: {round(calories * 0.10)} kcal

☀️ Lunch (1:00–2:00 PM)
• {lunch}
• Approx: {round(calories * 0.35)} kcal

🌆 Evening Snack (4:30 PM)
• Roasted Makhana (1 cup) or Sprouts Chaat
• Approx: {round(calories * 0.10)} kcal

🌙 Dinner (7:30–8:30 PM)
• {dinner}
• Approx: {round(calories * 0.23)} kcal

━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 MACRO BREAKDOWN (Daily · {calories} kcal)
• 🍚 Carbohydrates: {round(calories * 0.45 / 4)}g (~45%)
• 🥩 Protein: {protein}g (~{round(protein * 4 / calories * 100)}%)
• 🥑 Healthy Fats: {round(calories * 0.25 / 9)}g (~25%)

━━━━━━━━━━━━━━━━━━━━━━━━━━

⭐ ML-RECOMMENDED FOODS
{focus_lines}

━━━━━━━━━━━━━━━━━━━━━━━━━━

🚫 FOODS TO AVOID (ML Model)
{avoid_lines}

━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 KEY LIFESTYLE TIPS for {bmi_category}
1. ⏰ Eat within a 10-hour window — front-load calories before 2 PM.
2. 💧 Drink {water}L daily — 500ml warm water every morning.
3. 🚶 Walk 10–15 min after lunch and dinner — stabilises blood sugar.

━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 Generated by ML Diet Engine (offline fallback)."""


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 VitalizeHealth server starting on http://localhost:{port}")
    print("   Press Ctrl+C to stop\n")
    app.run(debug=True, host="0.0.0.0", port=port)
