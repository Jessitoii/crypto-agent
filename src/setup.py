from telethon import TelegramClient
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

API_ID = int(os.getenv('API_ID'))
API_HASH = os.getenv('API_HASH')
TELETHON_SESSION_NAME = os.getenv('TELETHON_SESSION_NAME')

async def main():
    """Initializes Telegram session and generates session file."""
    path = os.path.realpath(__file__)
    dir = os.path.dirname(path)
    dir = dir.replace('src', 'data')
    os.chdir(dir)
    print("[TELEGRAM] Creating session...")
    client = TelegramClient(TELETHON_SESSION_NAME, API_ID, API_HASH)
    
    await client.start()
    
    print("[SUCCESS] Session file created.")
    print("[INFO] You can now run main.py.")
    
    me = await client.get_me()
    print(f"[INFO] Logged in as: {me.username}")
    
    await client.disconnect()

if __name__ == '__main__':
    asyncio.run(main())