import os
import re
import logging
import asyncio
from typing import List, Dict
from collections import defaultdict
from dotenv import load_dotenv
from telegram import Update, InputFile, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from PyPDF2 import PdfReader
from transformers import pipeline
from gtts import gTTS
from flask import Flask, request, jsonify

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

if not TOKEN or not WEBHOOK_URL:
    raise RuntimeError("TELEGRAM_BOT_TOKEN and WEBHOOK_URL must be set in environment variables or .env file.")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PDF text extraction
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
    return text

# Chunk text for summarization
def chunk_text(text: str, max_chunk: int = 1024) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_chunk:
            chunks.append(current)
            current = sentence
        else:
            current += (" " if current else "") + sentence
    if current:
        chunks.append(current)
    return chunks

# Text summarization
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
def summarize_text(text: str) -> str:
    prompt = (
        "As an expert oil market trader, summarize the following document. "
        "Focus on key trading insights, market trends, and actionable information. "
        "If possible, mention any sources or references found in the document.\n\n"
    )
    full_text = prompt + text
    chunks = chunk_text(full_text)
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
    return "Oil Market Trader Summary:\n" + "\n".join(summaries)

# Text-to-speech
def text_to_speech(text: str, output_path: str) -> str:
    try:
        tts = gTTS(text)
        tts.save(output_path)
    except Exception as e:
        logger.error(f"TTS failed: {e}")
    return output_path

# File name sanitization
def sanitize_filename(filename: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

# Per-user PDF queues and timers
user_pdf_queues: Dict[int, List[str]] = defaultdict(list)
user_timers: Dict[int, asyncio.Task] = {}
MAX_PDFS_PER_USER = 5
MAX_FILE_SIZE_MB = 10

async def process_user_pdfs(user_id: int, chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    pdf_paths = user_pdf_queues[user_id]
    summaries = []
    try:
        print(f"[PROCESS] Starting processing for user {user_id} with {len(pdf_paths)} PDFs.")
        await context.bot.send_message(chat_id=chat_id, text=f"⏳ Processing {len(pdf_paths)} PDF(s)...")
        for idx, file_path in enumerate(pdf_paths, 1):
            print(f"[PROCESS] Extracting text from PDF {idx}/{len(pdf_paths)}: {file_path}")
            await context.bot.send_message(chat_id=chat_id, text=f"Extracting text from PDF {idx}/{len(pdf_paths)}...")
            text = extract_text_from_pdf(file_path)
            if text.strip():
                print(f"[PROCESS] Summarizing PDF {idx}/{len(pdf_paths)}.")
                await context.bot.send_message(chat_id=chat_id, text=f"Summarizing PDF {idx}/{len(pdf_paths)}...")
                summary = summarize_text(text)
                summaries.append(summary)
            else:
                logger.warning(f"No text extracted from {file_path}")
                await context.bot.send_message(chat_id=chat_id, text=f"No text could be extracted from PDF {idx}.")
    finally:
        for file_path in pdf_paths:
            try:
                os.remove(file_path)
                print(f"[CLEANUP] Deleted file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
        user_pdf_queues[user_id] = []
        user_timers.pop(user_id, None)
    if summaries:
        combined_summary = '\n'.join(summaries)
        audio_path = f"./tmp/{user_id}_summary.mp3"
        print(f"[PROCESS] Generating audio summary for user {user_id}.")
        await context.bot.send_message(chat_id=chat_id, text="Generating audio summary...")
        text_to_speech(combined_summary, audio_path)
        try:
            with open(audio_path, 'rb') as audio_file:
                print(f"[SEND] Sending audio summary to user {user_id}.")
                await context.bot.send_voice(chat_id=chat_id, voice=audio_file)
                await context.bot.send_message(chat_id=chat_id, text="✅ Summary audio sent!")
        finally:
            try:
                os.remove(audio_path)
                print(f"[CLEANUP] Deleted audio file: {audio_path}")
            except Exception as e:
                logger.error(f"Failed to delete {audio_path}: {e}")
    else:
        await context.bot.send_message(chat_id=chat_id, text="Could not extract text from the PDFs.")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    document = update.message.document
    print(f"[RECEIVE] Received document from user {user_id}: {document.file_name}")
    # Security: check file type and size
    if document.mime_type != 'application/pdf' or not document.file_name.lower().endswith('.pdf'):
        await update.message.reply_text("Please send a valid PDF document.")
        return
    if document.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        await update.message.reply_text(f"PDF too large. Max size is {MAX_FILE_SIZE_MB}MB.")
        return
    if len(user_pdf_queues[user_id]) >= MAX_PDFS_PER_USER:
        await update.message.reply_text(f"You can only send up to {MAX_PDFS_PER_USER} PDFs at a time.")
        return
    file = await context.bot.get_file(document.file_id)
    safe_name = sanitize_filename(f"{user_id}_{document.file_name}")
    file_path = f"./tmp/{safe_name}"
    await file.download_to_drive(file_path)
    user_pdf_queues[user_id].append(file_path)
    print(f"[QUEUE] Added PDF to queue for user {user_id}. Queue length: {len(user_pdf_queues[user_id])}")
    await update.message.reply_text(f"PDF received! Waiting for more... (Send more PDFs or wait 3 seconds to process)")
    # Cancel previous timer if exists
    if user_id in user_timers:
        user_timers[user_id].cancel()
        print(f"[TIMER] Cancelled previous timer for user {user_id}.")
    # Start a new timer
    async def timer():
        try:
            print(f"[TIMER] Started 3-second timer for user {user_id}.")
            await asyncio.sleep(3)
            await process_user_pdfs(user_id, chat_id, context)
        except asyncio.CancelledError:
            print(f"[TIMER] Timer cancelled for user {user_id}.")
    user_timers[user_id] = asyncio.create_task(timer())

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to the Oil Market PDF Summarizer Bot!\n\n"
        "Send me one or more PDF documents related to the oil market.\n"
        "After you stop sending PDFs for 3 seconds, I will summarize all of them in the context of an oil market trader, focusing on key trading insights, market trends, and actionable information.\n\n"
        "You will receive a single audio message with the combined summary.\n\n"
        "Just send your PDFs to get started!"
    )

# Flask app for webhook
app = Flask(__name__)

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
