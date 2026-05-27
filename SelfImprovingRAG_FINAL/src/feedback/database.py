import sqlite3
import os
import json
from typing import Dict, Any, List
from datetime import datetime

class FeedbackDB:
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self.init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """Initialize feedback tracking tables."""
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS feedback_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT, 
                    answer TEXT,
                    user_rating INTEGER,
                    faithfulness_score REAL, 
                    relevancy_score REAL,
                    strategy_used TEXT,
                    bm25_weight REAL, 
                    vector_weight REAL,
                    retrieval_score REAL, 
                    overall_reward REAL,
                    session_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )""")
            c.execute("""
                CREATE TABLE IF NOT EXISTS strategy_weights (
                    strategy_name TEXT PRIMARY KEY,
                    weight REAL DEFAULT 0.25,
                    sample_count INTEGER DEFAULT 0,
                    avg_reward REAL DEFAULT 0.5
                )""")
            
            # Initial strategies
            strategies = ["sub_query_decomposition", "step_back_prompting", "hyde", "multi_query"]
            for s in strategies:
                c.execute("INSERT OR IGNORE INTO strategy_weights (strategy_name) VALUES (?)", (s,))
            conn.commit()

    def save_feedback(
        self, 
        query: str, 
        answer: str, 
        strategy: str, 
        faithfulness: float, 
        relevancy: float,
        user_rating: int = None, 
        session_id: str = ""
    ) -> float:
        """Save feedback and update strategy weights via EMA."""
        reward = (faithfulness + relevancy) / 2
        if user_rating == 1: reward = min(reward + 0.3, 1.0)
        elif user_rating == -1: reward = max(reward - 0.4, 0.0)

        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO feedback_records
                (query, answer, user_rating, faithfulness_score, relevancy_score,
                 strategy_used, bm25_weight, vector_weight, retrieval_score, overall_reward, session_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (query, answer, user_rating, faithfulness, relevancy,
                 strategy, 0.3, 0.7, faithfulness, reward, session_id))
            
            # Update strategy weights (EMA)
            c.execute("SELECT avg_reward, sample_count FROM strategy_weights WHERE strategy_name=?", (strategy,))
            row = c.fetchone()
            if row:
                old_avg, count = row
                learning_rate = float(os.environ.get("LEARNING_RATE", "0.1"))
                new_avg = (1 - learning_rate) * old_avg + learning_rate * reward
                c.execute("""UPDATE strategy_weights SET avg_reward=?, sample_count=?+1
                             WHERE strategy_name=?""", (new_avg, count, strategy))
            conn.commit()
        return reward

    def get_strategy_weights(self) -> Dict[str, float]:
        """Get normalized strategy weights based on historical performance."""
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT strategy_name, avg_reward FROM strategy_weights")
            rows = c.fetchall()
            
        if not rows:
            return {}
            
        weights = {s: r for s, r in rows}
        total = sum(weights.values()) or 1
        return {s: r / total for s, r in weights.items()}
    
    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent feedback history."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM feedback_records ORDER BY created_at DESC LIMIT ?", (limit,))
            return [dict(row) for row in c.fetchall()]
