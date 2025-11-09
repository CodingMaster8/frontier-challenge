"""
Database View Manager
Applies SQL views to DuckDB database with proper error handling
"""
import duckdb
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class ViewManager:
    """Manages SQL views for the fund search system"""

    def __init__(self, db_path: str = "data/br_funds.db"):
        self.db_path = db_path
        self.sql_dir = Path(__file__).parent.parent.parent / "sql"

    def apply_views(self, force: bool = False) -> Dict[str, bool]:
        """
        Apply all SQL views to the database

        Args:
            force: If True, recreate views even if they exist

        Returns:
            Dictionary of view_name -> success status
        """
        results = {}

        view_files = [
            "01_create_semantic_search_view.sql",
            "02_create_structured_filter_view.sql",
            "03_create_portfolio_analysis_view.sql",
        ]

        conn = duckdb.connect(self.db_path)

        for sql_file in view_files:
            file_path = self.sql_dir / sql_file
            view_name = self._extract_view_name(sql_file)

            try:
                # Check if view exists
                exists = self._view_exists(conn, view_name)

                if exists and not force:
                    logger.info(f"View '{view_name}' already exists, skipping")
                    results[view_name] = True
                    continue

                # Apply the view
                logger.info(f"Creating view '{view_name}' from {sql_file}")
                sql_content = file_path.read_text()

                # Extract CREATE VIEW statement
                view_sql = self._extract_create_statement(sql_content)

                if view_sql:
                    conn.execute(view_sql)
                    logger.info(f"✓ Successfully created view '{view_name}'")
                    results[view_name] = True
                else:
                    logger.error(f"✗ Could not extract CREATE statement from {sql_file}")
                    results[view_name] = False

            except Exception as e:
                logger.error(f"✗ Error creating view '{view_name}': {str(e)}")
                results[view_name] = False

        conn.close()
        return results

    def drop_views(self) -> Dict[str, bool]:
        """Drop all managed views"""
        results = {}
        view_names = [
            "fund_semantic_search_view",
            "fund_structured_filter_view",
            "fund_portfolio_analysis_view",
            "fund_holdings_detail_view"
        ]

        conn = duckdb.connect(self.db_path)

        for view_name in view_names:
            try:
                conn.execute(f"DROP VIEW IF EXISTS {view_name}")
                logger.info(f"✓ Dropped view '{view_name}'")
                results[view_name] = True
            except Exception as e:
                logger.error(f"✗ Error dropping view '{view_name}': {str(e)}")
                results[view_name] = False

        conn.close()
        return results

    def list_views(self) -> List[str]:
        """List all views in the database"""
        conn = duckdb.connect(self.db_path, read_only=True)

        views = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_type = 'VIEW'
            ORDER BY table_name
        """).fetchall()

        conn.close()
        return [v[0] for v in views]

    def _view_exists(self, conn, view_name: str) -> bool:
        """Check if a view exists"""
        result = conn.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_type = 'VIEW'
              AND table_name = '{view_name}'
        """).fetchone()
        return result[0] > 0

    def _extract_view_name(self, filename: str) -> str:
        """Extract view name from filename"""
        mapping = {
            "01_create_semantic_search_view.sql": "fund_semantic_search_view",
            "02_create_structured_filter_view.sql": "fund_structured_filter_view",
            "03_create_portfolio_analysis_view.sql": "fund_portfolio_analysis_view",
        }
        return mapping.get(filename, "unknown")

    def _extract_create_statement(self, sql_content: str) -> str:
        """Extract CREATE VIEW statement from SQL file"""
        # Split by CREATE OR REPLACE VIEW
        parts = sql_content.split('CREATE OR REPLACE VIEW')

        if len(parts) < 2:
            return ""

        # Take first view definition (stops at next comment block or end)
        view_sql = 'CREATE OR REPLACE VIEW ' + parts[1].split('-- =====')[0].strip()
        return view_sql

    def validate_views(self) -> Dict[str, bool]:
        """Validate that all views return data"""
        results = {}
        conn = duckdb.connect(self.db_path, read_only=True)
        for view_name in self.list_views():
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
                results[view_name] = count > 0
            except Exception as e:
                results[view_name] = False
        conn.close()
        return results


def apply_all_views(force: bool = False):
    """Convenience function to apply all views"""
    logging.basicConfig(level=logging.INFO)
    manager = ViewManager()
    results = manager.apply_views(force=force)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    print(f"\n{'='*60}")
    print(f"Applied {success_count}/{total_count} views successfully")
    print(f"{'='*60}\n")

    for view_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {view_name}")

    return results



if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    apply_all_views(force=force)
