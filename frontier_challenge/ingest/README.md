# Ingest Module

ETL pipeline for downloading, processing, and loading CVM lamina data into DuckDB.

## Purpose

This module implements a complete ETL pipeline to ingest Brazilian fund data from CVM (Comissão de Valores Mobiliários) into a DuckDB database for analysis.

## Components

### download_lamina.py
Downloads lamina fund data from CVM's public data repository:
- Scrapes CVM website for available data files
- Filters files by date range (last N months)
- Downloads ZIP files with progress tracking
- Skips already downloaded files
- Validates file patterns and dates

### process_lamina.py
Processes raw ZIP files into structured CSV files:
- Extracts CSV files from multiple ZIP archives
- Normalizes filenames by removing date suffixes
- Groups data by type (carteira, composicao, etc.)
- Combines data across months into unified files
- Uses chunked reading for memory efficiency

### load_to_db.py
Loads processed CSV files into DuckDB tables:
- Creates one table per CSV file type
- Adds configurable table prefix (default: `lamina_`)
- Supports replace, append, or fail modes
- Returns row counts for validation
- Handles large files efficiently with DuckDB's native CSV reader

## Design Decisions

**Three-Stage Pipeline**: Separates download, processing, and loading to enable:
- Independent execution of each stage
- Better error recovery
- Flexible data retention policies
- Easier testing and validation

**Monthly File Pattern**: CVM publishes monthly files in format `lamina_fi_YYYYMM.zip`. The pipeline handles this pattern explicitly with date extraction and sorting.

**Chunked Processing**: Uses pandas chunking to handle large files without loading everything into memory, critical for production environments.

**DuckDB Native Loading**: Leverages DuckDB's optimized CSV reader instead of pandas for final loading, significantly faster for large datasets.

**Normalization**: Removes date suffixes from filenames to create stable table names (e.g., `lamina_fi_carteira_202510.csv` → `lamina_carteira` table).

## Usage

```python
from frontier_challenge.ingest import (
    download_latest_lamina_data,
    process_lamina_files,
    load_csv_to_duckdb
)

# Download last 3 months
files = download_latest_lamina_data(
    output_dir="data/lamina",
    n_months=3
)

# Process into unified CSVs
output_files = process_lamina_files(
    input_dir="data/lamina",
    output_dir="data/processed"
)

# Load to database
row_counts = load_csv_to_duckdb(
    csv_dir="data/processed",
    db_path="data/br_funds.db"
)
```

## Data Flow

1. CVM publishes → `lamina_fi_YYYYMM.zip` files
2. Download → `data/lamina/` directory
3. Process → Extract and combine by type → `data/processed/`
4. Load → DuckDB tables with `lamina_` prefix
