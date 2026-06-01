import os
import sqlite3

PERSISTENCE_DIR = "persistence"
DB_PATH = os.path.join(PERSISTENCE_DIR, "clustering_results.db")
JOBLIB_PATH = os.path.join(PERSISTENCE_DIR, "final_results.joblib")

def init_db():
    if not os.path.exists(PERSISTENCE_DIR):
        os.makedirs(PERSISTENCE_DIR)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS kfold_tuning (
            k INTEGER PRIMARY KEY,
            avg_silhouette REAL,
            avg_iters REAL,
            time_taken REAL,
            fold_scores TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS final_training_results (
            best_k INTEGER PRIMARY KEY,
            train_score REAL,
            val_score REAL,
            test_score REAL,
            variance REAL,
            cached_data BLOB,
            created_at TEXT
        )
    ''')
    # Ensure old DBs get the new column if needed.
    cursor.execute("PRAGMA table_info(kfold_tuning)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'fold_scores' not in columns:
        cursor.execute('ALTER TABLE kfold_tuning ADD COLUMN fold_scores TEXT')
    conn.commit()
    conn.close()

def clear_cache_files():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(JOBLIB_PATH):
        os.remove(JOBLIB_PATH)
    # Re-initialize the empty DB structure
    init_db()
