"""
Persistent Storage and Memory Management

This module manages the system's persistent memory layer using SQLite3.
It handles news deduplication through TF-IDF vectorization and cosine similarity,
logs AI-driven decisions, and maintains a comprehensive trade history.

Key Features:
- Semantic deduplication of incoming news streams.
- Relational mapping between AI decisions and realized trade outcomes.
- Startup hydration of runtime state from historical records.
"""

import sqlite3
import time
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class MemoryManager:
    """
    Manages SQLite database operations and semantic memory features.
    
    Attributes:
        db_path (str): File path to the SQLite database.
        vectorizer (TfidfVectorizer): Scikit-learn vectorizer for similarity analysis.
    """
    def __init__(self, db_path="nexus_db.sqlite"):
        """
        Initializes the MemoryManager and ensures schema integrity.
        
        Args:
            db_path (str): Absolute or relative path to the database file.
        """
        self.db_path = db_path
        self._init_db()
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def _init_db(self):
        """
        Initializes database tables and creates optimized indices.
        
        Schemas include:
        - news: Raw ingestion logs for deduplication.
        - decisions: AI-generated recommendations and confidence metrics.
        - trades: Realized virtual or live trade outcomes.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # TABLE: NEWS - Stores raw news context for semantic comparison
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                content TEXT,
                timestamp REAL
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_news_timestamp ON news (timestamp)')

        # TABLE: DECISIONS - Records the logic behind every trade recommendation
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                action TEXT,
                confidence INTEGER,
                reason TEXT,
                validity INTEGER,
                price REAL,
                tp_pct REAL,
                sl_pct REAL,
                news_snippet TEXT,
                raw_data TEXT
            )
        ''')

        # TABLE: TRADE HISTORY - Audit ledger for execution performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id INTEGER, 
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                reason TEXT,
                peak_price REAL,
                FOREIGN KEY(decision_id) REFERENCES decisions(id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def clean_text(self, text):
        """
        Normalizes raw text for improved semantic matching.
        
        Args:
            text (str): Input text block.
            
        Returns:
            str: Normalized, lowercase text without URLs or special characters.
        """
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def is_duplicate(self, new_text, threshold=0.75):
        """
        Evaluates the semantic similarity of new content against recent history.
        
        Utilizes TF-IDF vectorization and Cosine Similarity to identify 
        redundant news items across different providers.
        
        Args:
            new_text (str): The incoming news content.
            threshold (float): Minimum similarity coefficient [0, 1] to flag as duplicate.
            
        Returns:
            tuple: (bool: Is Duplicate, float: Maximum Similarity Score)
        """
        clean_new = self.clean_text(new_text)
        if not clean_new.strip(): return True, 1.0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Restrict comparison to the last 24 hours to maintain topical relevance
        limit_time = time.time() - (24 * 60 * 60)
        cursor.execute('SELECT content FROM news WHERE timestamp > ? ORDER BY id DESC LIMIT 100', (limit_time,))
        rows = cursor.fetchall()
        conn.close()

        if not rows: return False, 0.0

        past_news = [self.clean_text(row[0]) for row in rows]
        try:
            corpus = past_news + [clean_new]
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            # Compare the last item (new_text) with all previous items in the matrix
            similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
            max_sim = similarities.flatten().max() if similarities.size > 0 else 0.0
            
            if max_sim >= threshold:
                print(f"[SIMILARITY] Redundant content discarded: {max_sim:.2f}")
                return True, max_sim
            return False, max_sim
        except Exception as e:
            print(f"[ERROR] Similarity calculation failed: {e}")
            return False, 0.0

    def add_news(self, source, content):
        """
        Records a raw news event in the database.
        
        Args:
            source (str): Source identifier (e.g., 'Telegram').
            content (str): Raw news content.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('INSERT INTO news (source, content, timestamp) VALUES (?, ?, ?)', 
                          (source, content, time.time()))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[ERROR] Database ingestion failed: {e}")

    def log_decision(self, record):
        """
        Persists an AI trade decision and returns its unique identifier.
        
        Args:
            record (dict): Comprehensive decision metadata.
            
        Returns:
            int: The unique database ID of the inserted record.
        """
        decision_id = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO decisions (timestamp, symbol, action, confidence, reason, price, news_snippet, validity, tp_pct, sl_pct, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record['time'], record['symbol'], record['action'], record['confidence'], 
                record['reason'], record['price'], record['news_snippet'], record['validity'], record['tp_pct'], record['sl_pct'], json.dumps(record)
            ))
            decision_id = cursor.lastrowid
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[ERROR] Decision persistence failed: {e}")
        return decision_id

    def log_trade(self, record, decision_id=None):
        """
        Logs a realized trade outcome, linked to its original decision.
        
        Args:
            record (dict): Trade results (PnL, Exit Price, etc.).
            decision_id (int, optional): DB ID of the AI recommendation.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (decision_id, timestamp, symbol, side, entry_price, exit_price, pnl, reason, peak_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                    decision_id, 
                    record.get('time'), 
                    record.get('symbol'), 
                    record.get('side'),
                    record.get('entry'), 
                    record.get('exit'), 
                    record.get('pnl'), 
                    record.get('reason'), 
                    record.get('peak', 0)
                ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[ERROR] Trade outcome persistence failed: {e}")

    def load_recent_history(self, ctx):
        """
        Hydrates the application context with recent historical data on bootstrap.
        
        Loads the last 100 AI decisions and 50 closed trades into the UI/Runtime
        to provide immediate continuity for the operator.
        
        Args:
            ctx (BotContext): Application context.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Phase 1: Hydrate AI Recommendation Deque
        cursor.execute('SELECT * FROM decisions ORDER BY id DESC LIMIT 100')
        decisions = cursor.fetchall()
        for d in reversed(decisions):
            rec = {
                "time": d['timestamp'], "symbol": d['symbol'], "action": d['action'],
                "confidence": d['confidence'], "reason": d['reason'], "price": d['price'],
                "news_snippet": d['news_snippet']
            }
            ctx.ai_decisions.append(rec)

        # Phase 2: Hydrate Closed Trades Ledger
        cursor.execute('SELECT * FROM trades ORDER BY id DESC LIMIT 50')
        trades = cursor.fetchall()
        for t in reversed(trades):
            rec = {
                'time': t['timestamp'], 'symbol': t['symbol'], 'side': t['side'],
                'pnl': t['pnl'], 'reason': t['reason'], 'entry': t['entry_price'],
                'exit': t['exit_price']
            }
            ctx.exchange.history.append(rec)
            
        conn.close()
        print(f"[SYSTEM] State Hydrated: {len(decisions)} decisions, {len(trades)} trades restored.")

    def get_full_trade_story(self):
        """
        Generates a unified performance report by joining decisions and outcomes.
        
        Returns:
            list: A list of unified trade records (AI Reason -> Execution Outcome).
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # JOIN query to correlate predictive reasoning with real-world alpha
        query = '''
            SELECT 
                d.timestamp as time, d.symbol, d.action, d.confidence, d.reason as ai_reason,
                t.entry_price, t.exit_price, t.pnl, t.reason as close_reason, t.peak_price
            FROM decisions d
            LEFT JOIN trades t ON t.decision_id = d.id
            WHERE d.action IN ('LONG', 'SHORT')
            ORDER BY d.id DESC
            LIMIT 100
        '''
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]