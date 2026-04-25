"""
RSS Feed Monitoring Service

This module provides an asynchronous listener for RSS feeds. It tracks new 
articles from crypto news outlets, deduplicates them using link analysis, 
and filters out stale content to ensure the AI analyzes only real-time events.
"""

import feedparser
import asyncio
import time
from config import RSS_FEEDS

class RSSMonitor:
    """
    Monitor for tracking and processing crypto news via RSS.
    
    Attributes:
        callback (callable): Function to invoke when new news is detected.
        seen_links (set): Cache of processed article links for deduplication.
        is_running (bool): Lifecycle state of the monitor.
    """
    def __init__(self, callback_func):
        """
        Initializes the RSS monitor.
        
        Args:
            callback_func (callable): Async function for downstream processing.
        """
        self.callback = callback_func
        self.seen_links = set()
        self.is_running = False

    async def fetch_feed(self, url):
        """
        Fetches and processes entries from a specific RSS URL.
        
        Args:
            url (str): The RSS feed endpoint.
        """
        try:
            # Parse feed in a thread pool to avoid blocking the event loop
            feed = await asyncio.to_thread(feedparser.parse, url)
            
            # Focus on the most recent entries for low-latency analysis
            for entry in feed.entries[:3]:
                link = entry.link
                title = entry.title
                summary = getattr(entry, 'summary', '')
                
                # Filter out stale news (Older than 1 hour)
                if hasattr(entry, 'published_parsed'):
                    published_time = time.mktime(entry.published_parsed)
                    current_time = time.time()
                    if current_time - published_time > 3600:
                        continue
                
                # Deduplication check
                if link not in self.seen_links:
                    self.seen_links.add(link)
                    
                    full_text = f"{title}. {summary}"
                    print(f"[RSS] Alpha Signal Detected: {title[:60]}...")
                    
                    # Trigger the processing pipeline
                    await self.callback(full_text, "RSS")
                    
        except Exception as e:
            print(f"[ERROR] RSS Ingestion failure ({url}): {e}")

    async def start_loop(self):
        """
        Starts the persistent monitoring loop for all configured RSS feeds.
        """
        print("[SYSTEM] RSS Monitoring Service Online.")
        self.is_running = True
        
        while self.is_running:
            # Parallel fetching of all configured feeds
            tasks = [self.fetch_feed(url) for url in RSS_FEEDS]
            await asyncio.gather(*tasks)
            
            # Polling interval (60 seconds)
            await asyncio.sleep(60)