import logging
from pathlib import Path
from typing import Dict, List, Optional

import duckdb

logger = logging.getLogger(__name__)


def load_csv_to_duckdb(
    csv_dir: str = "data/processed_lamina",
    db_path: str = "data/br_funds.db",
    table_prefix: str = "lamina_",
    pattern: str = "*.csv",
    if_exists: str = "replace",
) -> Dict[str, int]:
    """
    Load processed lamina CSV files into DuckDB database tables.

    Creates one table per CSV file with the naming convention: {table_prefix}{csv_name}.
    For example, 'carteira.csv' becomes 'lamina_carteira' table.

    Parameters
    ----------
    csv_dir : str, optional
        Directory containing the CSV files to load.
        Default is "data/processed_lamina".
    db_path : str, optional
        Path to the DuckDB database file.
        Default is "data/br_funds.db".
    table_prefix : str, optional
        Prefix to add to table names. Default is "lamina_".
    pattern : str, optional
        Glob pattern to match CSV files. Default is "*.csv".
    if_exists : str, optional
        Action if table exists: 'replace', 'append', or 'fail'.
        Default is 'replace'.

    Returns
    -------
    Dict[str, int]
        Dictionary mapping table names to row counts.

    Raises
    ------
    FileNotFoundError
        If csv_dir doesn't exist or contains no matching files.
    ValueError
        If if_exists parameter is invalid.
    """
    # Validate parameters
    if if_exists not in ['replace', 'append', 'fail']:
        raise ValueError(f"if_exists must be 'replace', 'append', or 'fail', got '{if_exists}'")

    csv_path = Path(csv_dir)
    db_file = Path(db_path)

    # Validate CSV directory
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    # Find all CSV files
    csv_files = sorted(csv_path.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No files matching pattern '{pattern}' found in {csv_dir}"
        )

    logger.info(f"Found {len(csv_files)} CSV files to load into {db_path}")

    # Create database directory if needed
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # Connect to DuckDB
    conn = duckdb.connect(str(db_file))

    try:
        row_counts: Dict[str, int] = {}

        # Process each CSV file
        for idx, csv_file in enumerate(csv_files, 1):
            # Generate table name from CSV filename
            csv_name = csv_file.stem  # filename without extension
            table_name = f"{table_prefix}{csv_name}"

            logger.info(f"[{idx}/{len(csv_files)}] Loading {csv_file.name} -> {table_name}")

            try:
                # Check if table exists
                table_exists = conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                    [table_name]
                ).fetchone()[0] > 0

                if table_exists:
                    if if_exists == 'fail':
                        raise ValueError(f"Table '{table_name}' already exists")
                    elif if_exists == 'replace':
                        logger.debug(f"Dropping existing table: {table_name}")
                        conn.execute(f"DROP TABLE IF EXISTS {table_name}")

                # Load CSV into table
                # DuckDB's read_csv_auto is very efficient and handles type inference
                if if_exists == 'append' and table_exists:
                    # Insert into existing table
                    conn.execute(f"""
                        INSERT INTO {table_name}
                        SELECT * FROM read_csv_auto(
                            '{csv_file.absolute()}',
                            delim=',',
                            header=true,
                            auto_detect=true,
                            ignore_errors=true
                        )
                    """)
                else:
                    # Create new table from CSV
                    conn.execute(f"""
                        CREATE TABLE {table_name} AS
                        SELECT * FROM read_csv_auto(
                            '{csv_file.absolute()}',
                            delim=',',
                            header=true,
                            auto_detect=true,
                            ignore_errors=true
                        )
                    """)

                # Get row count
                row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                row_counts[table_name] = row_count

                # Get column count
                col_count = conn.execute(
                    f"SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{table_name}'"
                ).fetchone()[0]

                logger.info(
                    f"[{idx}/{len(csv_files)}] ✓ {table_name}: "
                    f"{row_count:,} rows, {col_count} columns"
                )

            except Exception as e:
                logger.error(f"Failed to load {csv_file.name}: {e}")
                # Continue with next file instead of failing completely
                continue

        if not row_counts:
            raise ValueError("No tables were successfully loaded")

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info("Load complete! Summary:")
        logger.info(f"{'='*60}")

        total_rows = 0
        for table_name, row_count in row_counts.items():
            total_rows += row_count
            logger.info(f"  {table_name:30s}: {row_count:10,} rows")

        logger.info(f"{'='*60}")
        logger.info(f"Total rows loaded: {total_rows:,}")
        logger.info(f"Database: {db_file.absolute()}")
        logger.info(f"Database size: {db_file.stat().st_size:,} bytes")

        return row_counts

    finally:
        conn.close()


def verify_tables(
    db_path: str = "data/br_funds.db",
    table_prefix: str = "lamina_",
) -> Dict[str, Dict[str, any]]:
    """
    Verify tables in the DuckDB database and return metadata.

    Parameters
    ----------
    db_path : str, optional
        Path to the DuckDB database file.
        Default is "data/br_funds.db".
    table_prefix : str, optional
        Filter tables by prefix. Default is "lamina_".

    Returns
    -------
    Dict[str, Dict[str, any]]
        Dictionary with table metadata including row count and columns.
    """
    db_file = Path(db_path)

    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    logger.info(f"Verifying tables in {db_path}")

    conn = duckdb.connect(str(db_file), read_only=True)

    try:
        # Get all tables matching prefix
        tables = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE ? || '%'
            ORDER BY table_name
        """, [table_prefix]).fetchall()

        if not tables:
            logger.warning(f"No tables found with prefix '{table_prefix}'")
            return {}

        metadata = {}

        for (table_name,) in tables:
            # Get row count
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            # Get column info
            columns = conn.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = ?
                ORDER BY ordinal_position
            """, [table_name]).fetchall()

            metadata[table_name] = {
                'row_count': row_count,
                'column_count': len(columns),
                'columns': {col: dtype for col, dtype in columns}
            }

            logger.info(f"  {table_name:30s}: {row_count:10,} rows, {len(columns):3} columns")

        return metadata

    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load CSV files into DuckDB
    row_counts = load_csv_to_duckdb(
        csv_dir="data/processed_lamina",
        db_path="data/br_funds.db",
        table_prefix="lamina_",
        if_exists="replace",
    )

    print(f"\n{'='*60}")
    print("Verifying loaded tables...")
    print(f"{'='*60}")

    # Verify tables
    metadata = verify_tables(
        db_path="data/br_funds.db",
        table_prefix="lamina_",
    )
