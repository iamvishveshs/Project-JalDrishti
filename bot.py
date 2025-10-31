import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import sys

# --- Configuration ---
BOT_TOKEN = "8228755868:AAF5TKV6s15AIBU6LFMVju3-vjQ-aocgGrE" # Your Bot Token

# --- Map DISTRICT Names to their Invite Links ---
# You get these from the private channel settings
DISTRICT_INVITE_LINKS = {
    "Mandi District": "t.me/+XxuJ6tkWIUBhMTk9",     # e.g., "t.me/+ABC123XYZ"
    "Kullu District": "t.me/+KOSTCa8LaihiYzNl",
    "Shimla District": "t.me/+mvjGO9a5EpIyYTI9",
    "Bilaspur District": "t.me/+W9epomxWzEozNmQ9",
    "Chamba District": "t.me/+TDuH5VtDPoE0MzY1",
    "Hamirpur District": "t.me/+aYxdGzKxMG1kZmNl",
    "Kangra District": "t.me/+dAprlhj-J5NiMDc1",
    "Kinnaur District": "t.me/+fuKsqSi61bNjOWJl",
    "Lahaul and Spiti District": "t.me/+Fu3T94X4lrUyODg1",
    "Sirmaur District": "t.me/+lbzbvwku55VlMDE1",
    "Solan District": "t.me/+i7wQf9HNDT5lYWJl",
    "Una District": "t.me/+dy5NxKccJV5kYjI1"
    # Add all districts you monitor and their invite links
}
# --- End Configuration ---

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Command Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message."""
    user = update.effective_user
    await update.message.reply_html(
        f"""👋 Hi {user.mention_html()}!
<b>Welcome to the Himachal Pradesh Flood Alerts Bot!</b>
Get timely flood warnings for districts across Himachal Pradesh.
This bot helps you subscribe to private alert channels for specific districts. You'll receive notifications when potential floods are detected near monitored locations within that district.<b>To get started:</b>
Use the <b>/subscribe</b> command or tap the button below to choose a district.
Stay safe! 🙏""",
        reply_markup=start_keyboard(),
    )

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the district selection keyboard."""
    text = "Please choose a district to subscribe to for flood alerts:"
    if update.message:
        await update.message.reply_text(text, reply_markup=district_keyboard())
    elif update.callback_query:
         await update.callback_query.edit_message_text(text, reply_markup=district_keyboard())

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays help information."""
    help_text = (
        "This bot provides flood alerts for districts in Himachal Pradesh.\n\n"
        "1. Use /subscribe or the button below to see available districts.\n"
        "2. Select a district.\n"
        "3. You'll receive an invite link to join the Telegram channel for that district.\n"
        "4. Join the channel to receive alerts for all monitored locations within that district.\n\n"
        "To stop receiving alerts, simply leave the district's private channel.\n\n"
        "Commands:\n"
        "/start - Welcome message\n"
        "/subscribe - Choose district\n"
        "/help - Show this message"
    )
    if update.message:
        await update.message.reply_text(help_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back to Start", callback_data="start")]]))
    elif update.callback_query:
        await update.callback_query.edit_message_text(help_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back to Start", callback_data="start")]]))


# --- Keyboard Builders ---

def start_keyboard():
    """Creates the initial keyboard shown with /start."""
    keyboard = [
        [InlineKeyboardButton("🔔 Subscribe to District Alerts", callback_data="subscribe")],
        [InlineKeyboardButton("❓ Help / How it Works", callback_data="help")],
    ]
    return InlineKeyboardMarkup(keyboard)

def district_keyboard():
    """Creates the keyboard with district buttons."""
    keyboard = []
    # Create buttons for each district defined in the links map
    for district_name in DISTRICT_INVITE_LINKS.keys():
        # Use district name for both button text and callback data prefix
        keyboard.append([InlineKeyboardButton(district_name, callback_data=f"dist_{district_name}")])
    keyboard.append([InlineKeyboardButton("« Back to Start", callback_data="start")])
    return InlineKeyboardMarkup(keyboard)

# --- Callback Query Handler ---

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parses the CallbackQuery and acts accordingly."""
    query = update.callback_query
    await query.answer()

    data = query.data

    if data == "start":
        await query.edit_message_text(
            text=f"Hi {query.effective_user.mention_html()}! Welcome back.",
            reply_markup=start_keyboard(),
            parse_mode='HTML'
        )
    elif data == "subscribe":
        await subscribe(update, context) # Show district options
    elif data == "help":
        await help_command(update, context) # Show help info

    elif data.startswith("dist_"):
        # Extract district name from callback data (e.g., "dist_Mandi District" -> "Mandi District")
        district_name = data[len("dist_"):]

        if district_name in DISTRICT_INVITE_LINKS:
            invite_link = DISTRICT_INVITE_LINKS[district_name]

            # Check if the invite link seems properly configured
            if invite_link and not invite_link.startswith("YOUR_") and "INVITE_LINK" not in invite_link:
                # Send the invite link as a NEW message
                await query.message.reply_text(
                    f"To get flood alerts for **{district_name}**, please join the {district_name} Flood Alerts channel using this link:\n\n"
                    f"Channel Link: {invite_link}\n\n"
                    f"Joining this channel will give you alerts for all monitored locations in {district_name}.",
                    parse_mode='Markdown',
                    # Keep the original message showing district options but maybe add a confirmation
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Choose Another District", callback_data="subscribe")]])
                )
                try:
                    await query.edit_message_text(
                        text=f"Invite link for {district_name} sent!\nChoose another district?",
                        reply_markup=district_keyboard() # Keep showing district options
                    )
                except Exception as e:
                     logger.warning(f"Could not edit message after sending link: {e}")
            else:
                logger.error(f"Invite link for district '{district_name}' not found or not configured!")
                await query.message.reply_text(
                    f"Sorry, the alert channel link for **{district_name}** is not configured correctly. Please contact the administrator.",
                    parse_mode='Markdown',
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back to Districts", callback_data="subscribe")]])
                )
        else:
            await query.edit_message_text(
                text="Sorry, that district selection is not recognized.",
                reply_markup=district_keyboard() # Show options again
                )

# --- Main Bot Execution ---

def main() -> None:
    """Start the bot."""
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ ERROR: BOT_TOKEN is not configured in bot.py! Please add your Telegram Bot Token.")
        sys.exit(1)
    # Check invite links
    configured_links = 0
    for district, link in DISTRICT_INVITE_LINKS.items():
        if link.startswith("YOUR_") or "INVITE_LINK" in link:
             print(f"⚠️ WARNING: Invite link for '{district}' seems unconfigured in DISTRICT_INVITE_LINKS.")
        else:
            configured_links += 1
    if configured_links == 0:
         print("❌ ERROR: No district invite links seem to be configured correctly in DISTRICT_INVITE_LINKS!")
         # sys.exit(1) # Optionally exit if none are set

    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("subscribe", subscribe))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button))

    logger.info("Bot starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()