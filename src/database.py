import sqlite3
import time
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class MemoryManager:
    def __init__(self, db_path="nexus_db.sqlite"):
        self.db_path = db_path
        self._init_db()
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def _init_db(self):
        """Initializes database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. TABLE: NEWS
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                content TEXT,
                timestamp REAL
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_news_timestamp ON news (timestamp)')

        # 2. TABLE: AI DECISIONS
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

        # 3. TABLE: TRADE HISTORY
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
        """Normalizes text by lowercasing and removing URLs/special characters."""
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def is_duplicate(self, new_text, threshold=0.75):
        """Checks if the news content is a duplicate of recent entries using TF-IDF and cosine similarity."""
        clean_new = self.clean_text(new_text)
        if not clean_new.strip(): return True, 1.0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        limit_time = time.time() - (24 * 60 * 60)
        cursor.execute('SELECT content FROM news WHERE timestamp > ? ORDER BY id DESC LIMIT 100', (limit_time,))
        rows = cursor.fetchall()
        conn.close()

        if not rows: return False, 0.0

        past_news = [self.clean_text(row[0]) for row in rows]
        try:
            corpus = past_news + [clean_new]
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
            max_sim = similarities.flatten().max() if similarities.size > 0 else 0.0
            
            if max_sim >= threshold:
                print(f"[SIMILARITY] Duplicate content detected: {max_sim:.2f}")
                return True, max_sim
            return False, max_sim
        except Exception as e:
            print(f"[ERROR] Similarity check failed: {e}")
            return False, 0.0

    def add_news(self, source, content):
        """Adds news entry to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('INSERT INTO news (source, content, timestamp) VALUES (?, ?, ?)', 
                          (source, content, time.time()))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[ERROR] Database write failed: {e}")

    def log_decision(self, record):
        """
        Logs an AI decision to the database and returns the inserted ID.
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
            print(f"[ERROR] Database decision log failed: {e}")
        return decision_id

    def log_trade(self, record, decision_id=None):
        """
        Logs a completed trade to the database.
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
            print(f"[ERROR] Database trade log failed: {e}")

    def load_recent_history(self, ctx):
        """
        Loads the last 100 decisions and 50 trades into context on startup.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 1. Load Decisions
        cursor.execute('SELECT * FROM decisions ORDER BY id DESC LIMIT 100')
        decisions = cursor.fetchall()
        for d in reversed(decisions):
            rec = {
                "time": d['timestamp'], "symbol": d['symbol'], "action": d['action'],
                "confidence": d['confidence'], "reason": d['reason'], "price": d['price'],
                "news_snippet": d['news_snippet']
            }
            ctx.ai_decisions.append(rec)

        # 2. Load Trades
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
        print(f"[SYSTEM] Memory refreshed: {len(decisions)} decisions, {len(trades)} trades loaded.")

    def get_full_trade_story(self):
        """Retrieves combined report: Decision -> Trade -> Outcome."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
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