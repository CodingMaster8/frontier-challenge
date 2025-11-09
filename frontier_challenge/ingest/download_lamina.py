import os
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Configure logging
logger = logging.getLogger(__name__)


def _extract_year_month(filename: str) -> Tuple[int, int]:
    """
    Extract year and month from lamina filename.

    Parameters
    ----------
    filename : str
        Filename in format 'lamina_fi_YYYYMM.zip'

    Returns
    -------
    Tuple[int, int]
        Tuple of (year, month). Returns (0, 0) if pattern not found.

    Examples
    --------
    >>> _extract_year_month('lamina_fi_202404.zip')
    (2024, 4)
    >>> _extract_year_month('lamina_fi_202512.zip')
    (2025, 12)
    """
    # Match pattern: lamina_fi_YYYYMM.zip or YYYYMM.zip
    pattern = r'(\d{4})(\d{2})\.zip$'
    match = re.search(pattern, filename)

    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        return (year, month)

    return (0, 0)


def download_latest_lamina_data(
    base_url: str = "https://dados.cvm.gov.br/dados/FI/DOC/LAMINA/DADOS",
    output_dir: str = "data/lamina",
    n_months: int = 3,
    timeout: int = 30,
    skip_existing: bool = True,
) -> List[Path]:
    """
    Download the latest N months of data from CVM Lamina directory.

    This function scrapes the specified URL to find .zip files corresponding
    to monthly data releases, downloads the most recent N files, and saves
    them to the specified output directory.

    Parameters
    ----------
    base_url : str, optional
        Base URL of the CVM Lamina data directory.
        Default is "https://dados.cvm.gov.br/dados/FI/DOC/LAMINA/DADOS".
    output_dir : str, optional
        Directory path where downloaded files will be saved.
        Default is "data/lamina". Directory will be created if it doesn't exist.
    n_months : int, optional
        Number of most recent monthly data files to download.
        Default is 3. Must be positive.
    timeout : int, optional
        Timeout in seconds for HTTP requests. Default is 30.
    skip_existing : bool, optional
        If True, skip downloading files that already exist in output_dir.
        Default is True.

    Returns
    -------
    List[Path]
        List of Path objects for successfully downloaded files.

    Raises
    ------
    ValueError
        If n_months is not a positive integer.
    requests.RequestException
        If there are network connectivity issues or HTTP errors.
    IOError
        If there are issues creating the output directory or writing files.
    """
    # Input validation
    if not isinstance(n_months, int) or n_months <= 0:
        raise ValueError(f"n_months must be a positive integer, got {n_months}")

    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")

    logger.info(
        f"Starting download of {n_months} latest files from {base_url}"
    )

    # Create output directory
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory ensured: {output_path.absolute()}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise IOError(f"Cannot create output directory: {e}") from e

    # Fetch and parse the directory listing
    try:
        logger.debug(f"Fetching directory listing from {base_url}")
        response = requests.get(base_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch directory listing from {base_url}: {e}")
        raise

    soup = BeautifulSoup(response.content, "html.parser")

    # Extract .zip file links
    links = soup.find_all("a", href=True)
    zip_links = []

    for link in links:
        href = link["href"]
        if href.endswith(".zip"):
            # Handle both absolute and relative URLs
            full_url = urljoin(base_url + "/", href)
            zip_links.append(full_url)

    if not zip_links:
        logger.warning(f"No .zip files found at {base_url}")
        return []

    logger.info(f"Found {len(zip_links)} .zip files in directory")

    # Sort files by name (assuming chronological naming) and get the latest N
    zip_links = sorted(zip_links, key=lambda x: _extract_year_month(Path(x).name), reverse=True)[:n_months]

    # Log the selected files with their dates
    selected_dates = [_extract_year_month(Path(url).name) for url in zip_links]
    logger.info(
        f"Selected {len(zip_links)} most recent files for download "
        f"(dates: {[(y, m) for y, m in selected_dates if y > 0]})"
    )

    # Download files
    downloaded_files: List[Path] = []

    for idx, zip_url in enumerate(zip_links, 1):
        file_name = Path(zip_url).name
        output_file = output_path / file_name

        # Check if file already exists
        if skip_existing and output_file.exists():
            file_size = output_file.stat().st_size
            logger.info(
                f"[{idx}/{len(zip_links)}] Skipping existing file: "
                f"{file_name} ({file_size:,} bytes)"
            )
            downloaded_files.append(output_file)
            continue

        try:
            logger.info(f"[{idx}/{len(zip_links)}] Downloading: {file_name}")

            # Stream download for large files
            with requests.get(zip_url, stream=True, timeout=timeout) as r:
                r.raise_for_status()

                # Get file size if available
                total_size = r.headers.get("content-length")
                if total_size:
                    logger.debug(f"File size: {int(total_size):,} bytes")

                # Write to file
                with open(output_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            file_size = output_file.stat().st_size
            logger.info(
                f"[{idx}/{len(zip_links)}] Successfully downloaded: "
                f"{file_name} ({file_size:,} bytes)"
            )
            downloaded_files.append(output_file)

        except requests.RequestException as e:
            logger.error(
                f"[{idx}/{len(zip_links)}] Failed to download {file_name}: {e}"
            )
            # Continue with next file instead of failing completely
            continue
        except IOError as e:
            logger.error(
                f"[{idx}/{len(zip_links)}] Failed to write {file_name}: {e}"
            )
            # Clean up partial file
            if output_file.exists():
                try:
                    output_file.unlink()
                except Exception:
                    pass
            continue

    logger.info(
        f"Download complete. Successfully downloaded {len(downloaded_files)}/{len(zip_links)} files"
    )

    return downloaded_files

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    downloaded_files = download_latest_lamina_data(
        n_months=12,
        skip_existing=True
    )

    print(f"\nDownloaded {len(downloaded_files)} files:")
    for file in downloaded_files:
        print(f"  - {file}")
