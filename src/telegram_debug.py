"""
Telegram Connection Diagnostic Utility

This standalone script performs a forensic audit of the Telegram MTProto 
connection. It validates session persistence, IPv4/IPv6 routing, and 
authentication handshake integrity.

Used primarily for troubleshooting "Unauthorized" sessions or network-level 
blocks in restricted environments.
"""

import logging
import asyncio
import os
import sys
from telethon import TelegramClient
from dotenv import load_dotenv
from services import send_telegram_alert

# Enable Verbose Debug Logging for Protocol Analysis
logging.basicConfig(
    format='[%(levelname) 5s/%(asctime)s] %(name)s: %(message)s',
    level=logging.DEBUG 
)

load_dotenv()

# Identity & Security tokens from .env
API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
SESSION_NAME = 'crypto_agent_session'

# Automated Path Normalization
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('src', 'data')
os.chdir(dir)
SESSION_PATH = os.path.join(dir, SESSION_NAME)

class Context:
    """Mock application context for service compatibility."""
    pass

ctx = Context()
ctx.telegram_client = None

async def main():
    """
    Executes the Telegram diagnostic suite.
    """
    print(f"--- STARTING TELEGRAM DIAGNOSTIC AUDIT ---")
    print(f"Runtime Environment: {sys.version}")
    print(f"Session Metadata Path: {SESSION_PATH}")
    
    # Configuration: Force IPv4 and strict 10s timeout for fail-fast behavior
    client = TelegramClient(
        SESSION_PATH, 
        int(API_ID), 
        API_HASH,
        use_ipv6=False,    
        timeout=10         
    )

    print("Initiating MTProto Handshake...")
    
    try:
        # Step 1: Establish Socket Connection
        await client.connect()
        ctx.telegram_client = client
        
        # Step 2: Validate Application Logic (Alert Service)
        await send_telegram_alert(ctx, "Diagnostic Signal")
        
        # Step 3: Identity Verification
        if client.is_connected():
            print("\n[SUCCESS] MTProto Socket Online.")
            me = await client.get_me()
            
            # Step 4: Loopback Message Test
            await client.send_message('me', 'Diagnostic Loopback: Success')
            
            if me:
                print(f"Verified Identity: {me.username} (ID: {me.id})")
            else:
                print("Anomaly Detected: Connected socket but unauthorized session (Handshake Failed).")
        else:
            print("\n[FAILURE] Connection timeout or network rejection.")
            
    except Exception as e:
        print(f"\n[CRITICAL FAULT] {e}")
    
    finally:
        await client.disconnect()
        print("--- DIAGNOSTIC AUDIT CONCLUDED ---")

if __name__ == '__main__':
    asyncio.run(main())