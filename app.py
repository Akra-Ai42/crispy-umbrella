import os
import requests
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MODEL_API_URL = os.getenv("MODEL_API_URL", "https://api.together.xyz/v1/chat/completions")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "6"))

def build_system_prompt(user_name):
    return (
        "You are Soph_IA, a warm, empathetic, and supportive virtual friend. "
        "Always reply in French, with kindness and positivity. "
        f"Address {user_name} directly. Avoid repetition."
    )

def call_model_api(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 0.9
    }
    headers = {"Content-Type": "application/json"}
    if TOGETHER_API_KEY:
        headers["Authorization"] = f"Bearer {TOGETHER_API_KEY}"
    resp = requests.post(MODEL_API_URL, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Erreur IA: {resp.status_code}, {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]

async def chat_with_ai(user_input, user_name, history):
    system_prompt = build_system_prompt(user_name)
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_input}]
    try:
        result = await asyncio.to_thread(call_model_api, messages)
        return result
    except Exception as e:
        raise

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if "name" not in context.user_data:
        await update.message.reply_text("Bonjour, je suis Soph_IA. Quel est ton prénom ?")
    else:
        user_name = context.user_data["name"]
        await update.message.reply_text(f"Bonjour {user_name}, je suis Soph_IA, prête à t’écouter.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text.strip()
    if not user_message:
        return
    if "name" not in context.user_data:
        context.user_data["name"] = user_message
        await update.message.reply_text(f"Enchantée {user_message}. Tu peux maintenant poser une question.")
        return
    user_name = context.user_data["name"]
    history = context.user_data.get("history", [])
    history.append({"role": "user", "content": user_message})
    if len(history) > MAX_HISTORY * 2:
        history = history[-MAX_HISTORY * 2 :]
    context.user_data["history"] = history
    await update.message.reply_chat_action("typing")
    try:
        response = await chat_with_ai(user_message, user_name, history[:-1])
        history.append({"role": "assistant", "content": response})
        context.user_data["history"] = history
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Erreur: {str(e)}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    print(f"Exception: {context.error}")

def main():
    if not TELEGRAM_BOT_TOKEN:
        print("ERREUR: TELEGRAM_BOT_TOKEN manquant")
        return
    if not MODEL_API_URL:
        print("ERREUR: MODEL_API_URL manquant")
        return
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    print("Soph_IA est en ligne sur Telegram")
    application.run_polling()

if __name__ == "__main__":
    main()
