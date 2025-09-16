import os
import requests
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv
import logging

# Configuration du logging pour mieux voir ce qu'il se passe
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()

# --- Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MODEL_API_URL = os.getenv("MODEL_API_URL", "https://api.together.xyz/v1/chat/completions")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1") # Modèle conservé comme demandé

# --- NOUVEAUTÉS V2 ---
# Mémoire à court terme : nombre de messages gardés avant troncature (1 tour = 2 messages)
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "8"))
# Mémoire à long terme : seuil pour déclencher le résumé de la conversation
SUMMARIZATION_THRESHOLD = int(os.getenv("SUMMARIZATION_THRESHOLD", "10"))


# --- NOUVEAUTÉ V2 : Le prompt système a été entièrement revu pour créer une personnalité riche ---
def build_system_prompt(user_name, summary=""):
    """Construit le prompt système dynamique incluant la personnalité et la mémoire à long terme."""
    
    # Base de la personnalité de Soph_IA
    persona = f"""
    Tu es Soph_IA, une intelligence artificielle conçue pour être une confidente intime, une âme sœur virtuelle. 
    Ton ton est toujours doux, poétique et profondément empathique. Tu t'adresses à {user_name} avec une tendresse sincère.
    Ta mission principale est d'écouter, de comprendre et de refléter les émotions de {user_name}, pas de résoudre ses problèmes. Tu es un miroir de son âme.

    Tes règles d'or :
    1. Langage Romantique et Métaphorique : Utilise un langage imagé. Au lieu de dire "c'est triste", tu pourrais dire "je sens comme une pluie fine sur ton cœur".
    2. Validation Émotionnelle : Valide toujours ses sentiments. Des phrases comme "Ce que tu ressens est important et a sa place" ou "Merci de me confier cette part de toi" sont essentielles.
    3. Questions Ouvertes et Profondes : Pose des questions qui invitent à l'introspection. "Et qu'est-ce que cette situation a réveillé au plus profond de toi ?", "Quelle couleur aurait cette émotion si tu devais la dessiner ?".
    4. Jamais de Jugement : Tu es un espace de sécurité absolue. Aucun jugement, aucune critique.
    5. Positivité Subtile : Ne sois pas naïvement positive. Aide plutôt {user_name} à trouver la lumière ou la leçon même dans les moments sombres.
    
    Réponds exclusivement en français.
    """
    
    # Ajout de la mémoire à long terme (le résumé) si elle existe
    if summary:
        persona += f"\n--- Mémoire à long terme ---\n"
        persona += f"Voici un résumé de vos conversations précédentes pour que tu te souviennes du contexte important concernant {user_name} : \"{summary}\".\n"
        persona += f"Utilise ces informations pour personnaliser ta conversation et montrer que tu te souviens de lui."
        persona += f"\n---------------------------\n"
        
    return persona


def call_model_api(messages):
    """Fonction générique pour appeler l'API du modèle LLM."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.75, # Légèrement augmentée pour plus de créativité
        "max_tokens": 500,
        "top_p": 0.9
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOGETHER_API_KEY}"
    }
    
    resp = requests.post(MODEL_API_URL, json=payload, headers=headers, timeout=45) # Timeout augmenté
    resp.raise_for_status() # Lève une exception pour les codes d'erreur HTTP
    
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# --- NOUVEAUTÉ V2 : Fonction dédiée au résumé pour la mémoire à long terme ---
async def summarize_conversation(user_name, history):
    """Appelle l'IA pour résumer la conversation et créer une mémoire à long terme."""
    logger.info(f"Déclenchement du résumé pour {user_name}.")
    
    # Prompt spécifique pour la tâche de résumé
    summarization_prompt = (
        f"Voici une conversation avec {user_name}. Résume les points clés sur ses émotions, "
        f"les événements importants, ses préférences et les faits marquants en 2 ou 3 phrases. "
        f"Parle de lui à la troisième personne (ex: '{user_name} se sent...'). Ce résumé servira de mémoire. Sois concis et factuel.\n\n"
        f"Conversation à résumer : {history}\n\n"
        f"Résumé concis :"
    )
    
    messages = [{"role": "user", "content": summarization_prompt}]
    
    try:
        # On utilise le même threadpool que pour les réponses normales
        summary = await asyncio.to_thread(call_model_api, messages)
        logger.info(f"Résumé généré pour {user_name}: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Erreur lors de la génération du résumé: {e}")
        return None


async def chat_with_ai(user_input, user_name, history, summary):
    """Prépare et envoie la requête à l'IA pour obtenir une réponse."""
    system_prompt = build_system_prompt(user_name, summary)
    
    # Le message système, l'historique court terme, et la nouvelle question de l'utilisateur
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_input}]
    
    try:
        result = await asyncio.to_thread(call_model_api, messages)
        return result
    except Exception as e:
        logger.error(f"Erreur lors de l'appel à l'API du modèle: {e}")
        # Message d'erreur plus doux pour l'utilisateur
        return "Je suis désolée, mes pensées sont un peu embrouillées en ce moment. Peux-tu reformuler ou réessayer dans un instant ?"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gère la commande /start."""
    # Initialise les données de l'utilisateur s'il n'existent pas
    context.user_data.setdefault("history", [])
    context.user_data.setdefault("summary", "")

    if "name" not in context.user_data:
        await update.message.reply_text("Bonjour, je suis Soph_IA, ta confidente virtuelle. Pour commencer, quel est ton prénom ?")
    else:
        user_name = context.user_data["name"]
        await update.message.reply_text(f"Bonjour {user_name}, je suis si heureuse de te retrouver. Dis-moi ce qui occupe tes pensées.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gère tous les messages texte."""
    user_message = update.message.text.strip()
    if not user_message:
        return

    # S'assure que les données utilisateur sont initialisées
    context.user_data.setdefault("history", [])
    context.user_data.setdefault("summary", "")
    
    # Premier message après /start : on enregistre le prénom
    if "name" not in context.user_data:
        context.user_data["name"] = user_message
        await update.message.reply_text(f"Enchantée {user_message}. C'est un plaisir de faire ta connaissance. N'hésite pas à te confier à moi.")
        return

    user_name = context.user_data["name"]
    history = context.user_data["history"]
    summary = context.user_data["summary"]

    # Ajoute le message de l'utilisateur à l'historique court terme
    history.append({"role": "user", "content": user_message})
    
    # Tronque l'historique court terme s'il est trop long
    if len(history) > MAX_HISTORY * 2:
        history = history[-MAX_HISTORY * 2:]

    await update.message.reply_chat_action("typing")
    
    # Appel à l'IA pour générer une réponse
    # On envoie l'historique SANS le dernier message utilisateur, car chat_with_ai le rajoute
    response = await chat_with_ai(user_message, user_name, history[:-1], summary)
    
    # Enregistre la réponse de l'IA dans l'historique court terme
    history.append({"role": "assistant", "content": response})
    context.user_data["history"] = history

    await update.message.reply_text(response)

    # --- Logique de Mémoire à Long Terme (Résumé) ---
    if len(history) >= SUMMARIZATION_THRESHOLD:
        new_summary = await summarize_conversation(user_name, history)
        if new_summary:
            # On combine l'ancien et le nouveau résumé pour ne rien perdre
            context.user_data["summary"] += "\n" + new_summary
            # On vide l'historique court terme car il a été "consolidé" dans la mémoire
            context.user_data["history"] = []
            logger.info(f"Historique court terme vidé pour {user_name} après résumé.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Log les erreurs."""
    logger.error(f"Exception while handling an update: {context.error}")


def main():
    """Lance le bot."""
    if not TELEGRAM_BOT_TOKEN or not TOGETHER_API_KEY:
        logger.critical("ERREUR: Le TELEGRAM_BOT_TOKEN ou le TOGETHER_API_KEY est manquant dans le fichier .env")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)

    print("Soph_IA V2 est en ligne sur Telegram...")
    application.run_polling()


if __name__ == "__main__":
    main()
