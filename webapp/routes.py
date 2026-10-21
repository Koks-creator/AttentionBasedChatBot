import threading

from flask import jsonify, render_template, request

from webapp import app, models

# One instance for the whole app, so we have to do some locking
_locks = {name: threading.Lock() for name in models}


@app.route("/")
def home():
    return render_template("home.html", models=list(models))


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    name = data.get("model")
    message = str(data.get("message", "")).strip()
    history = data.get("history") or []

    if name not in models:
        return jsonify(error=f"unknown model: {name}"), 400
    if not message:
        return jsonify(error="empty message"), 400
    if not isinstance(history, list) or not all(isinstance(t, str) for t in history):
        return jsonify(error="history must be a list of strings"), 400
    try:
        temperature = float(data.get("temperature", 0.7))
        top_k = int(data.get("top_k", 20))
    except (TypeError, ValueError):
        return jsonify(error="temperature must be a number and top_k an integer"), 400

    bot = models[name]
    temperature = min(max(temperature, 0.1), 1.0)
    top_k = min(max(top_k, 1), bot.tokenizer.vocabulary_size())

    with _locks[name]:
        # History comes from the browser, so each tab has it's own
        bot.history = [bot.clean(t) for t in history[-bot.context_turns:]]
        reply = bot.respond(message, temperature=temperature, top_k=top_k)
    return jsonify(reply=reply)