import re
import duckdb

# ============================================================================
# Utility Functions
# ============================================================================


def extract_sql_from_markdown(text: str) -> str:
    """Extract SQL query from markdown code blocks"""
    # Try to find SQL in markdown code blocks
    patterns = [
        r"```sql\s*(.*?)\s*```",  # SQL code block
        r"```\s*(SELECT.*?)\s*```",  # Generic code block with SELECT
        r"(SELECT.*?)(?:\n\n|$)",  # Raw SELECT statement
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(1).strip()
            if sql:
                return sql

    # If no match, return the original text (might be raw SQL)
    return text.strip()


def validate_sql_syntax(sql: str, db_path: str) -> tuple[bool, str]:
    """
    Validate SQL syntax without executing the query.

    Returns (is_valid, error_message)
    """
    try:
        conn = duckdb.connect(db_path, read_only=True)
        # Use EXPLAIN to validate without execution
        conn.execute(f"EXPLAIN {sql}")
        conn.close()
        return True, ""
    except Exception as e:
        error_msg = str(e)
        return False, error_msg


def execute_sql_safe(sql: str, db_path: str) -> tuple[bool, str]:
    """
    Safely execute SQL query with error handling.

    Returns (success, error_or_success_message)
    """
    try:
        conn = duckdb.connect(db_path, read_only=True)
        result = conn.execute(sql)
        row_count = len(result.fetchall())
        conn.close()
        return True, f"Query executed successfully. Returned {row_count} rows."
    except Exception as e:
        error_msg = str(e)
        return False, f"Query execution error: {error_msg}"
