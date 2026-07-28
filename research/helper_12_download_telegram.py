from telethon import TelegramClient
import os

api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")

if not api_id or not api_hash:
    raise SystemExit("Missing TELEGRAM_API_ID or TELEGRAM_API_HASH in environment.")

client = TelegramClient("telegram_session_newspaper_wave_10", api_id, api_hash)

async def main():
    await client.start()
    async for dialog in client.iter_dialogs():
        # Filter channels you joined recently
        if dialog.is_channel and "paper" in dialog.name.lower():
            print(dialog.name, dialog.id)

with client:
    client.loop.run_until_complete(main())
