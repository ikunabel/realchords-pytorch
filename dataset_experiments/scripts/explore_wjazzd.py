#!/usr/bin/env python3
"""Inspect wjazzd SQLite schema and export tables to CSV."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

from _common import PROJECT_ROOT, ensure_output, write_text


def list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [row[0] for row in rows]


def table_columns(conn: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [(row[1], row[2]) for row in rows]


def export_table_csv(conn: sqlite3.Connection, table: str, csv_path: Path) -> tuple[int, int]:
    cursor = conn.execute(f"SELECT * FROM {table}")
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    return len(rows), len(columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        default=str(PROJECT_ROOT / "data/wjazzd/wjazzd.db"),
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export every table to output/wjazzd/csv/",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    if not db_path.exists():
        msg = f"Database not found: {db_path}\nSkipping CSV export.\n"
        write_text("wjazzd", "run_summary.txt", msg)
        print(msg)
        return

    conn = sqlite3.connect(db_path)
    tables = list_tables(conn)

    schema_lines = ["Tables in wjazzd database:", ""]
    for table in tables:
        columns = table_columns(conn, table)
        col_desc = ", ".join(f"{name} ({ctype})" for name, ctype in columns)
        schema_lines.append(f"  {table}: {col_desc}")

    write_text("wjazzd", "schema.txt", "\n".join(schema_lines) + "\n")

    export_lines = []
    if args.export_csv:
        csv_dir = ensure_output("wjazzd/csv")
        for table_name in tables:
            csv_path = csv_dir / f"{table_name}.csv"
            n_rows, n_cols = export_table_csv(conn, table_name, csv_path)
            export_lines.append(
                f"Exported {table_name}: {n_rows} rows, "
                f"{n_cols} columns -> {csv_path}"
            )

    conn.close()

    summary = "\n".join(
        [
            f"Database: {db_path}",
            f"Tables: {len(tables)}",
            "Schema: output/wjazzd/schema.txt",
            *export_lines,
            "",
        ]
    )
    write_text("wjazzd", "run_summary.txt", summary)
    print(summary)


if __name__ == "__main__":
    main()
