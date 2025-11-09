import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
from zipfile import ZipFile
import pandas as pd


logger = logging.getLogger(__name__)


def _normalize_csv_name(filename: str) -> str:
    """
    Normalize CSV filename by removing date suffix.

    Examples:
        lamina_fi_carteira_202510.csv -> carteira
        lamina_fi_composicao_202508.csv -> composicao
    """
    # Remove .csv extension
    name = filename.replace('.csv', '')
    # Remove date pattern (YYYYMM at the end)
    name = re.sub(r'_\d{6}$', '', name)
    # Remove lamina_fi_ prefix if present
    name = re.sub(r'^lamina_fi_', '', name)
    return name


def process_lamina_files(
    input_dir: str = "data/lamina",
    output_dir: str = "data/processed",
    pattern: str = "*.zip",
    encoding: str = "latin-1",
    sep: str = ";",
    chunksize: Optional[int] = 50000,
) -> Dict[str, Path]:
    """
    Process and combine lamina zip files into separate CSVs by type.

    Each zip file contains multiple CSV files (e.g., carteira, composicao, etc.).
    This function groups CSVs by type across all zip files and creates one
    combined output file per CSV type.

    Example:
        If you have 3 zip files, each containing:
        - lamina_fi_carteira_YYYYMM.csv
        - lamina_fi_composicao_YYYYMM.csv

        This will create 2 output files:
        - data/processed/carteira.csv (combining all carteira files)
        - data/processed/composicao.csv (combining all composicao files)

    Parameters
    ----------
    input_dir : str, optional
        Directory containing the zip files to process.
        Default is "data/lamina".
    output_dir : str, optional
        Directory where combined CSV files will be saved.
        Default is "data/processed".
    pattern : str, optional
        Glob pattern to match zip files. Default is "*.zip".
    encoding : str, optional
        Character encoding for CSV files. Default is "latin-1" (common for CVM data).
    sep : str, optional
        CSV separator/delimiter. Default is ";" (common for CVM data).
    chunksize : Optional[int], optional
        Number of rows to process at a time. If None, loads entire file at once.
        Default is 50000. Use for memory efficiency with large files.

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping CSV type name to output file path.
        Example: {"carteira": Path("data/processed/carteira.csv"), ...}

    Raises
    ------
    FileNotFoundError
        If input directory doesn't exist or contains no matching files.
    ValueError
        If no valid CSV files found in zip archives.
    IOError
        If there are issues reading zip files or writing output.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Validate input directory
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all zip files
    zip_files = sorted(input_path.glob(pattern))

    if not zip_files:
        raise FileNotFoundError(
            f"No files matching pattern '{pattern}' found in {input_dir}"
        )

    logger.info(f"Found {len(zip_files)} zip files to process")

    # Create output directory if needed
    output_path.mkdir(parents=True, exist_ok=True)

    # Track first write per CSV type for headers
    csv_type_first_write: Dict[str, bool] = {}
    csv_type_rows: Dict[str, int] = {}
    output_files: Dict[str, Path] = {}

    # Process each zip file
    for idx, zip_path in enumerate(zip_files, 1):
        logger.info(f"[{idx}/{len(zip_files)}] Processing: {zip_path.name}")

        try:
            with ZipFile(zip_path, 'r') as zip_ref:
                # Get list of CSV files in the zip
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]

                if not csv_files:
                    logger.warning(f"No CSV files found in {zip_path.name}")
                    continue

                logger.debug(f"Found {len(csv_files)} CSV file(s) in {zip_path.name}")

                # Process each CSV file in the zip
                for csv_file in csv_files:
                    # Normalize the CSV name to group similar files
                    csv_type = _normalize_csv_name(Path(csv_file).name)

                    # Initialize tracking for this CSV type if first time seeing it
                    if csv_type not in csv_type_first_write:
                        csv_type_first_write[csv_type] = True
                        csv_type_rows[csv_type] = 0
                        output_files[csv_type] = output_path / f"{csv_type}.csv"
                        logger.info(f"Discovered CSV type: '{csv_type}' -> {output_files[csv_type].name}")

                    output_file = output_files[csv_type]
                    logger.debug(f"Processing: {csv_file} -> {csv_type}")

                    try:
                        # Open CSV file from zip without extracting
                        with zip_ref.open(csv_file) as f:
                            # Process in chunks for memory efficiency
                            if chunksize:
                                for chunk_idx, chunk in enumerate(
                                    pd.read_csv(
                                        f,
                                        encoding=encoding,
                                        sep=sep,
                                        chunksize=chunksize,
                                        low_memory=False,
                                        on_bad_lines='warn',  # Skip bad lines with warning
                                    )
                                ):
                                    # Write chunk to output
                                    chunk.to_csv(
                                        output_file,
                                        mode='a',
                                        header=csv_type_first_write[csv_type],
                                        index=False,
                                        encoding='utf-8',
                                    )

                                    rows_in_chunk = len(chunk)
                                    csv_type_rows[csv_type] += rows_in_chunk
                                    csv_type_first_write[csv_type] = False

                                    logger.debug(
                                        f"[{csv_type}] Chunk {chunk_idx + 1}: {rows_in_chunk:,} rows "
                                        f"(total: {csv_type_rows[csv_type]:,})"
                                    )
                            else:
                                # Load entire file at once
                                df = pd.read_csv(
                                    f,
                                    encoding=encoding,
                                    sep=sep,
                                    low_memory=False,
                                    on_bad_lines='warn',  # Skip bad lines with warning
                                )

                                df.to_csv(
                                    output_file,
                                    mode='a',
                                    header=csv_type_first_write[csv_type],
                                    index=False,
                                    encoding='utf-8',
                                )

                                rows_in_df = len(df)
                                csv_type_rows[csv_type] += rows_in_df
                                csv_type_first_write[csv_type] = False

                                logger.debug(
                                    f"[{csv_type}] Loaded {rows_in_df:,} rows "
                                    f"(total: {csv_type_rows[csv_type]:,})"
                                )

                    except Exception as e:
                        logger.error(f"Failed to process {csv_file} from {zip_path.name}: {e}")
                        # Continue with next file instead of failing completely
                        continue

        except Exception as e:
            logger.error(f"Failed to open zip file {zip_path.name}: {e}")
            continue

        logger.info(f"[{idx}/{len(zip_files)}] Completed: {zip_path.name}")

    if not output_files:
        raise ValueError("No valid data found in any zip files")

    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info("Processing complete! Summary:")
    logger.info(f"{'='*60}")

    total_rows = 0
    for csv_type, output_file in output_files.items():
        rows = csv_type_rows[csv_type]
        total_rows += rows
        file_size = output_file.stat().st_size
        logger.info(
            f"  {csv_type:20s}: {rows:8,} rows -> {output_file.name} ({file_size:,} bytes)"
        )

    logger.info(f"{'='*60}")
    logger.info(f"Total rows across all files: {total_rows:,}")
    logger.info(f"Output directory: {output_path.absolute()}")

    return output_files


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    output_files = process_lamina_files(
        input_dir="data/lamina",
        output_dir="data/processed_lamina",
    )

    print(f"\n{'='*60}")
    print("Created files:")
    print(f"{'='*60}")
    for csv_type, path in output_files.items():
        print(f"  {csv_type:20s}: {path}")
        print(f"  {'':20s}  Size: {path.stat().st_size:,} bytes")
