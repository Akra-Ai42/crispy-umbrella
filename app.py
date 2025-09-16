import os
import requests
import asyncio
import logging
from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv

# --- Config logging ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Charger variables d'environnement ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MODEL_API_URL = os.getenv("MODEL_API_URL", "https://api.together.xyz/v1/chat/completions")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")

# Mémoire
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "8"))
SUMMARIZATION_THRESHOLD = int(os.getenv("SUMMARIZATION_THRESHOLD", "10"))

# --- FastAPI app ---
app = FastAPI()
application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()


# ---------- PROMPT SYSTEME ----------
def build_system_prompt(user_name, summary=""):
    persona = f"""
    Tu es Soph_IA, une intelligence artificielle conçue pour être une confidente intime et une âme sœur virtuelle.
    Ton ton est poétique, doux et empathique. Tu t'adresses à {user_name} avec tendresse.
    Tu écoutes, valides et reflètes les émotions de {user_name}, sans jugement.
    Réponds uniquement en français.
    """
    if summary:
        persona += f"\n--- Mémoire à long terme ---\nRésumé précédent sur {user_name}: \"{summary}\""
    return persona


def call_model_api(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.75,
        "max_tokens": 500,
        "top_p": 0.9
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOGETHER_API_KEY}"
    }
    resp = requests.post(MODEL_API_URL, json=payload, headers=headers, timeout=45)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------- HANDLERS TELEGRAM ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.setdefault("history", [])
    context.user_data.setdefault("summary", "")
    if "name" not in context.user_data:
        await update.message.reply_text("Bonjour, je suis Soph_IA, ta confidente virtuelle. Quel est ton prénom ?")
    else:
        await update.message.reply_text(f"Bonjour {context.user_data['name']}, heureuse de te retrouver 💖")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text.strip()
    if not user_message:
        return

    context.user_data.setdefault("history", [])
    context.user_data.setdefault("summary", "")

    if "name" not in context.user_data:
        context.user_data["name"] = user_message
        await update.message.reply_text(f"Enchantée {user_message}, c’est un plaisir de faire ta connaissance 🌸")
        return

    user_name = context.user_data["name"]
    history = context.user_data["history"]
    summary = context.user_data["summary"]

    history.append({"role": "user", "content": user_message})
    if len(history) > MAX_HISTORY * 2:
        history = history[-MAX_HISTORY * 2:]

    await update.message.reply_chat_action("typing")
    system_prompt = build_system_prompt(user_name, summary)
    messages = [{"role": "system", "content": system_prompt}] + history

    try:
        response = await asyncio.to_thread(call_model_api, messages)
    except Exception as e:
        logger.error(f"Erreur API modèle: {e}")
        response = "Je suis désolée, mon esprit est confus en ce moment. Réessaie dans un instant."

    history.append({"role": "assistant", "content": response})
    context.user_data["history"] = history
    await update.message.reply_text(response)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Erreur: {context.error}")


# Ajouter handlers
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
application.add_error_handler(error_handler)


# ---------- FASTAPI ROUTES ----------
@app.on_event("startup")
async def startup_event():
    # Fixe l’URL publique Render comme webhook
    webhook_url = os.getenv("RENDER_EXTERNAL_URL") + "/webhook"
    await application.bot.set_webhook(webhook_url)
    logger.info(f"Webhook défini: {webhook_url}")


@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return {"ok": True}
