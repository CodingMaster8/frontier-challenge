# Database Module

Database view management and setup for DuckDB.

## Purpose

This module manages the creation and maintenance of SQL views that power the agent's search and analysis capabilities.

## Components

### view_manager.py
`ViewManager` class that handles SQL view creation and management:
- Reads SQL definitions from the `sql/` directory
- Creates or recreates views in the DuckDB database
- Validates view existence and provides error handling
- Supports force recreation of existing views

## Design Decisions

**View-Based Architecture**: Uses database views instead of direct table queries to:
- Centralize complex join logic
- Optimize query performance through pre-computed joins
- Provide clean, application-specific data models
- Enable schema evolution without breaking tool queries

**Lazy Loading**: Views are created on-demand rather than at startup to avoid unnecessary database overhead during development and testing.

**Idempotent Operations**: View creation is idempotent with configurable force recreation, allowing safe repeated execution.

## SQL Views

The system relies on three main views defined in the `sql/` directory:

1. **fund_semantic_search_view**: Denormalized view combining fund metadata for vector search
2. **fund_structured_filter_view**: Performance metrics and attributes for SQL filtering
3. **portfolio_analysis_view**: Holdings data with fund relationships for portfolio analysis

## Usage

```python
from frontier_challenge.db import ViewManager

# Initialize manager
vm = ViewManager(db_path="data/br_funds.db")

# Apply all views
results = vm.apply_views(force=False)

# Check results
for view_name, success in results.items():
    print(f"{view_name}: {'✓' if success else '✗'}")
```
