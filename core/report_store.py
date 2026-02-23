import json
import os
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd


DEFAULT_DB_PATH = os.path.join("reports", "reports.db")


def init_report_db(db_path: str = DEFAULT_DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type TEXT NOT NULL,
            file_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            filters_json TEXT,
            csv_data TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_report(
    report_type: str,
    file_name: str,
    dataframe: pd.DataFrame,
    filters: Optional[dict] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    init_report_db(db_path)
    csv_data = dataframe.to_csv(index=False)
    filters_json = json.dumps(filters or {})
    created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    row_count = len(dataframe)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO reports (report_type, file_name, created_at, row_count, filters_json, csv_data)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (report_type, file_name, created_at, row_count, filters_json, csv_data),
    )
    report_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(report_id)

