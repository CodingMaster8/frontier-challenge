"""Test script for download_lamina function."""
import logging
from frontier_challenge.ingest.download_lamina import download_latest_lamina_data, _extract_year_month

# Configure logging to see the output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_extract_year_month():
    """Test the date extraction function."""
    print("\n=== Testing date extraction ===")
    test_cases = [
        'lamina_fi_202404.zip',
        'lamina_fi_202505.zip',
        'lamina_fi_202510.zip',
        'invalid_name.zip',
    ]

    for filename in test_cases:
        year, month = _extract_year_month(filename)
        print(f"{filename:30} -> Year: {year:4}, Month: {month:2}")

def test_download():
    """Test downloading the latest files."""
    print("\n=== Testing download (dry run with n_months=2) ===")
    try:
        downloaded = download_latest_lamina_data(
            output_dir="data/lamina_test",
            n_months=2,
            skip_existing=True
        )
        print(f"\n✓ Successfully downloaded {len(downloaded)} files:")
        for file_path in downloaded:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  - {file_path.name} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"\n✗ Error during download: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_extract_year_month()

    test_download()
    print("\n=== Download test completed ===")
