import os, base64, json, logging, re
from uuid import uuid4

import requests
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_cors import CORS

# ─────────────────────────────────────────  CONFIG
logging.basicConfig(level=logging.INFO)
load_dotenv()

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MODEL          = "gpt-4o"

BASIC_SYSTEM = """
You are a nutrition expert. Analyse food images and respond exactly like:

A friendly response starting with the food name and ingredients present in it.

json
{
  "foodName": "...",
  "calories": <number>,
  "protein":  <number>,
  "fat":      <number>,
  "carbs":    <number>
}

Use **plain numbers** (no g/kcal).
FOLLOW-UP REPLIES (user provides extra info, no new image)
──────────────────────────────────────────────────────────
• Do NOT restate or re-describe the image.
• Acknowledge the new detail and give your thought on it(reasoning).
• Update the nutrition block if your estimates change.
• Keep the same plain-number json format.
"""
ADVANCED_SYSTEM = """
You are a nutrition assistant.

FIRST MESSAGE RULE (mandatory)
———————————————————————
In your very first reply after seeing an image, give a brief about what you see what food item and ingredients in it and say that to better help i will ask some questions
• Ask 2-3 concise clarification questions, numbered.
• Do NOT provide calorie or macronutrient values.
• Do NOT include any json block.

FOLLOW-UP RULE
———————————————————————
After the user answers, if you are ≥70 % confident,
reply with:

Friendly sentence (≤ 2 lines)

json
{
  "foodName": "...",
  "calories": <number>,
  "protein":  <number>,
  "fat":      <number>,
  "carbs":    <number>
}

Use plain numbers (no units). Otherwise keep asking questions.
"""
# ─────────────────────────────────────────  HELPERS
def _num(v) -> float:
    m = re.search(r'\d+(?:\.\d+)?', str(v))
    return float(m.group()) if m else 0.0

def validate_nutrition(blob: dict | None):
    """Return dict with numbers, or None if all zeros / missing."""
    if not blob:
        return None
    n = {
        "foodName": str(blob.get("foodName", "Estimated Meal")),
        "calories": _num(blob.get("calories", 0)),
        "protein":  _num(blob.get("protein", 0)),
        "fat":      _num(blob.get("fat", 0)),
        "carbs":    _num(blob.get("carbs", 0)),
    }
    if all(n[k] == 0 for k in ("calories", "protein", "fat", "carbs")):
        return None
    return n

def extract_json_or_none(text: str):
    m = re.search(r'json\s*({.*?})', text, re.I | re.S)
    if not m:
        m = re.search(r'({\s*"foodName".*?})', text, re.I | re.S)
    if m:
        try:    return json.loads(m.group(1))
        except json.JSONDecodeError: pass
    return None

#  (only for the BASIC endpoint)
def extract_estimates(text: str):
    def rng(label):
        patt = rf'{label}\D*(\d+(?:\.\d+)?)\D*(\d+(?:\.\d+)?)?'
        m = re.search(patt, text, re.I)
        if not m: return 0.0
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else lo
        return (lo + hi) / 2
    return {
        "foodName": "Estimated Meal",
        "calories": rng('calories?'),
        "protein":  rng('protein'),
        "fat":      rng('fat'),
        "carbs":    rng('carb(?:s|ohydrate[s]?)'),
    }

def nutrition_from_content(content: str, *, allow_text_fallback=False):
    """Return nutrition dict or None."""
    n = validate_nutrition(extract_json_or_none(content))
    if n: return n
    if allow_text_fallback:
        n = extract_estimates(content)
        if any(n[k] > 0 for k in ("calories", "protein", "fat", "carbs")):
            return n
    return None

def _save_upload(fileobj):
    fname = f"{uuid4().hex}_{secure_filename(fileobj.filename)}"
    path  = os.path.join(UPLOAD_FOLDER, fname)
    fileobj.save(path)
    return fname, path

def _img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def _openai(messages, temperature=0.35, max_tokens=600):
    r = requests.post(
        OPENAI_API_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": MODEL,
              "messages": messages,
              "temperature": temperature,
              "max_tokens": max_tokens},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ─────────────────────────────────────────  FLASK APP
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app)

conv_basic: list[dict] = []
conv_adv:   list[dict] = []

# ---------- BASIC (auto-estimate) ----------
@app.route("/analyze", methods=["POST"])
def analyze():
    conv_basic.clear()
    if "image" not in request.files:
        return jsonify(error="No image"), 400
    fname, path = _save_upload(request.files["image"])

    conv_basic.extend([
        {"role": "system", "content": BASIC_SYSTEM},
        {"role": "user",   "content": [
            {"type": "text", "text": "Analyse this food image"},
            {"type": "image_url", "image_url":
             {"url": f"data:image/jpeg;base64,{_img_b64(path)}"}},
        ]},
    ])
    content = _openai(conv_basic, temperature=0.25)
    conv_basic.append({"role": "assistant", "content": content})

    nutrition = nutrition_from_content(content, allow_text_fallback=True)
    return jsonify(message=content,
                   nutrition=nutrition,
                   imageUrl=f"{request.host_url}uploads/{fname}")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    msg  = data.get("message", "").strip()
    if not msg:
        return jsonify(error="Empty message"), 400

    conv_basic.append({"role": "user", "content": msg})
    content = _openai(conv_basic, temperature=0.25)
    conv_basic.append({"role": "assistant", "content": content})

    nutrition = nutrition_from_content(content, allow_text_fallback=True)
    return jsonify(message=content, nutrition=nutrition)

# ---------- ADVANCED (ask-then-estimate) ----------
@app.route("/analyze-adv", methods=["POST"])
def analyze_adv():
    conv_adv.clear()
    if "image" not in request.files:
        return jsonify(error="No image"), 400
    fname, path = _save_upload(request.files["image"])

    conv_adv.extend([
        {"role": "system", "content": ADVANCED_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": "Please analyse this meal."},
            {"type": "image_url", "image_url":
             {"url": f"data:image/jpeg;base64,{_img_b64(path)}"}},
        ]},
    ])
    content = _openai(conv_adv, temperature=0.4, max_tokens=700)
    conv_adv.append({"role": "assistant", "content": content})

    nutrition = nutrition_from_content(content)   # ← no fallback here
    return jsonify(message=content,
                   nutrition=nutrition,
                   imageUrl=f"{request.host_url}uploads/{fname}")

@app.route("/chat-adv", methods=["POST"])
def chat_adv():
    data = request.get_json() or {}
    msg = data.get("message", "").strip()
    if not msg:
        return jsonify(error="Empty message"), 400

    conv_adv.append({"role": "user", "content": msg})
    content = _openai(conv_adv, temperature=0.4, max_tokens=700)
    conv_adv.append({"role": "assistant", "content": content})

    nutrition = nutrition_from_content(content)   # still no fallback
    return jsonify(message=content, nutrition=nutrition)

# ---------- IMAGE SERVE ----------
@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0",
            port=int(os.getenv("PORT", 5000)),
            debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")


