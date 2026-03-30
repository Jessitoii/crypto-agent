import feedparser
import asyncio
import time
from config import RSS_FEEDS

class RSSMonitor:
    def __init__(self, callback_func):
        self.callback = callback_func
        self.seen_links = set()
        self.is_running = False

    async def fetch_feed(self, url):
        """Fetches and processes entries from an RSS feed, filtering for new content."""
        try:
            feed = await asyncio.to_thread(feedparser.parse, url)
            
            for entry in feed.entries[:3]:
                link = entry.link
                title = entry.title
                summary = getattr(entry, 'summary', '')
                
                if hasattr(entry, 'published_parsed'):
                    published_time = time.mktime(entry.published_parsed)
                    current_time = time.time()
                    # Discard news older than 1 hour
                    if current_time - published_time > 3600:
                        continue
                
                if link not in self.seen_links:
                    self.seen_links.add(link)
                    
                    full_text = f"{title}. {summary}"
                    print(f"[RSS] New Entry Detected: {title[:50]}...")
                    await self.callback(full_text, "RSS")
                    
        except Exception as e:
            print(f"[ERROR] RSS Fetch failed ({url}): {e}")

    async def start_loop(self):
        """Main RSS monitoring loop."""
        print("[SYSTEM] RSS monitor started.")
        self.is_running = True
        
        while self.is_running:
            tasks = [self.fetch_feed(url) for url in RSS_FEEDS]
            await asyncio.gather(*tasks)
            
            await asyncio.sleep(60)