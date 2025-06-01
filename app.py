from dotenv import load_dotenv
load_dotenv()
import os
from telegram import Update, InputFile, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, Dispatcher
from PyPDF2 import PdfReader
from transformers import pipeline
from gtts import gTTS
from flask import Flask, request, jsonify
import asyncio
from collections import defaultdict

# PDF text extraction
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Text summarization
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
def summarize_text(text):
    # Add trader context prompt
    prompt = (
        "As an expert oil market trader, summarize the following document. "
        "Focus on key trading insights, market trends, and actionable information. "
        "If possible, mention any sources or references found in the document.\n\n"
    )
    full_text = prompt + text
    max_chunk = 1024
    if len(full_text) > max_chunk:
        full_text = full_text[:max_chunk]
    summary = summarizer(full_text, max_length=150, min_length=50, do_sample=False)
    # Add a header to the summary for clarity
    return "Oil Market Trader Summary:\n" + summary[0]['summary_text']

# Text-to-speech
def text_to_speech(text, output_path):
    tts = gTTS(text)
    tts.save(output_path)
    return output_path

# Per-user PDF queues and timers
user_pdf_queues = defaultdict(list)  # user_id -> list of file paths
user_timers = {}  # user_id -> asyncio.Task

async def process_user_pdfs(user_id, chat_id, context):
    pdf_paths = user_pdf_queues[user_id]
    summaries = []
    for file_path in pdf_paths:
        text = extract_text_from_pdf(file_path)
        if text.strip():
            summary = summarize_text(text)
            summaries.append(summary)
        os.remove(file_path)
    user_pdf_queues[user_id] = []
    if summaries:
        combined_summary = '\n'.join(summaries)
        audio_path = f"./tmp/{user_id}_summary.mp3"
        text_to_speech(combined_summary, audio_path)
        with open(audio_path, 'rb') as audio_file:
            await context.bot.send_voice(chat_id=chat_id, voice=audio_file)
        os.remove(audio_path)
    else:
        await context.bot.send_message(chat_id=chat_id, text="Could not extract text from the PDFs.")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    document = update.message.document
    if document.mime_type != 'application/pdf':
        await update.message.reply_text("Please send a PDF document.")
        return
    file = await context.bot.get_file(document.file_id)
    file_path = f"./tmp/{user_id}_{document.file_name}"
    await file.download_to_drive(file_path)
    user_pdf_queues[user_id].append(file_path)
    # Cancel previous timer if exists
    if user_id in user_timers:
        user_timers[user_id].cancel()
    # Start a new timer
    async def timer():
        try:
            await asyncio.sleep(3)
            await process_user_pdfs(user_id, chat_id, context)
        except asyncio.CancelledError:
            pass
    user_timers[user_id] = asyncio.create_task(timer())

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send me one or more PDFs. After 3 seconds of no new PDFs, I'll summarize them all as a single audio message!")

# Flask app for webhook
app = Flask(__name__)

# Telegram bot setup
TOKEN = "7649581953:AAEDP-aBEtjDsDkPGedA7Fz_rGr37eDS-cE"
WEBHOOK_URL = "https://test-summary-bot.onrender.com"

application = ApplicationBuilder().token(TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.Document.PDF, handle_document))

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# Webhook endpoint
@app.route("/webhook", methods=["POST"])
def webhook():
    if request.method == "POST":
        update = Update.de_json(request.get_json(force=True), application.bot)
        asyncio.run(application.process_update(update))
        return "ok"
    return "not allowed", 405

# Deployment instructions (comments):
# 1. Set TELEGRAM_BOT_TOKEN and WEBHOOK_URL as environment variables.
# 2. Deploy this app to a public server (Heroku, Render, Railway, etc.).
# 3. Register the webhook with Telegram:
#    Use the following code once (can be run in a Python shell):
#    from telegram import Bot
#    Bot(TOKEN).set_webhook(url=WEBHOOK_URL + '/webhook')
# 4. Ensure ./tmp directory exists for file storage.

if __name__ == "__main__":
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
