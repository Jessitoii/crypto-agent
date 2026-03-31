import logging
import asyncio
import os
import sys
from telethon import TelegramClient
from dotenv import load_dotenv
from services import send_telegram_alert

# Enable full debug logging for troubleshooting connections
logging.basicConfig(
    format='[%(levelname) 5s/%(asctime)s] %(name)s: %(message)s',
    level=logging.DEBUG 
)

load_dotenv()

API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
SESSION_NAME = 'crypto_agent_session'

# Path configuration
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('src', 'data')
os.chdir(dir)
SESSION_PATH = os.path.join(dir, SESSION_NAME)

class Context:
    pass
ctx = Context()
ctx.telegram_client = None

async def main():
    print(f"--- STARTING TELEGRAM DEBUG ANALYSIS ---")
    print(f"Python Version: {sys.version}")
    print(f"Session Path: {SESSION_PATH}")
    
    # Initialize client with IPv4 force and strict timeout
    client = TelegramClient(
        SESSION_PATH, 
        int(API_ID), 
        API_HASH,
        use_ipv6=False,    
        timeout=10         
    )

    print("Attempting client.connect()...")
    
    try:
        # Connection attempt
        await client.connect()
        ctx.telegram_client = client
        
        await send_telegram_alert(ctx, "Telegram Debug")
        if client.is_connected():
            print("\n[SUCCESS] Connection established.")
            me = await client.get_me()
            await client.send_message('me', 'Debug Message')
            if me:
                print(f"Identity: {me.username}")
            else:
                print("Connected but no identity found (Unauthorized session).")
        else:
            print("\n[ERROR] Connection failed (is_connected=False).")
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
    
    finally:
        await client.disconnect()
        print("--- ANALYSIS COMPLETED ---")

if __name__ == '__main__':
    asyncio.run(main())