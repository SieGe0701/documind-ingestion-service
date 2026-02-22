import os
import sqlite3
from typing import Dict, List


class SQLiteMetadataStore:
    """SQLite-backed metadata persistence for documents and chunks."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                upload_timestamp TEXT NOT NULL,
                num_chunks INTEGER NOT NULL,
                embedding_model TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                document_id TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                PRIMARY KEY (document_id, chunk_id),
                FOREIGN KEY (document_id) REFERENCES documents(document_id)
            )
            """
        )
        self.conn.commit()

    def save_document(
        self,
        document_id: str,
        filename: str,
        upload_timestamp: str,
        num_chunks: int,
        embedding_model: str,
    ) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO documents (document_id, filename, upload_timestamp, num_chunks, embedding_model)
            VALUES (?, ?, ?, ?, ?)
            """,
            (document_id, filename, upload_timestamp, num_chunks, embedding_model),
        )
        self.conn.commit()

    def save_chunks(self, document_id: str, chunks: List[Dict[str, int | str]]) -> None:
        if not chunks:
            return

        rows = [(document_id, int(c["chunk_id"]), str(c["text"])) for c in chunks]
        cursor = self.conn.cursor()
        cursor.executemany(
            """
            INSERT INTO chunks (document_id, chunk_id, chunk_text)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
