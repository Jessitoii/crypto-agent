"""
NEXUS Infrastructure Setup (Telegram Handshake)

This script performs the initial authentication handshake with Telegram 
servers to generate the persistent session file used by the agent.

Run this once before starting the main execution loop.
"""

from telethon import TelegramClient
import asyncio
import os
from dotenv import load_dotenv

# Hydrate environment variables
load_dotenv()

API_ID = int(os.getenv('API_ID'))
API_HASH = os.getenv('API_HASH')
TELETHON_SESSION_NAME = os.getenv('TELETHON_SESSION_NAME')

async def initialize_nexus_handshake():
    """
    Initializes the Telegram session and serializes the auth token 
    to the data directory for headless operation.
    """
    # Dynamic path resolution for cross-platform compatibility
    path = os.path.realpath(__file__)
    dir = os.path.dirname(path)
    dir = dir.replace('src', 'data')
    os.chdir(dir)
    
    print("[TELEGRAM] Commencing secure session creation...")
    client = TelegramClient(TELETHON_SESSION_NAME, API_ID, API_HASH)
    
    # Execution pass: start() handles interactive phone/code/password entry
    await client.start()
    
    print("[SUCCESS] Forensic session file established.")
    print("[INFO] Infrastructure ready. main.py can now be initialized.")
    
    me = await client.get_me()
    print(f"[INFO] Authenticated Identity: @{me.username}")
    
    await client.disconnect()

if __name__ == '__main__':
    asyncio.run(initialize_nexus_handshake())