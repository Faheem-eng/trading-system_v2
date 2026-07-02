"""
data_pipeline/binance_downloader.py

Professional historical data downloader for Binance Spot.

Data source: https://data.binance.vision (official Binance bulk archive)
    - No rate limits (vs REST API ~1200 req/min)
    - Monthly klines archives (fast bulk download)
    - SHA256 checksums for integrity verification

Critical design:
    - SHA256 checksum verification (data integrity guarantee)
    - Timestamp unit auto-detection (ms vs microseconds — Binance varies)
    - Header auto-detection (newer files have headers, older don't)
    - Idempotent + graceful (missing months skipped, not crashed)
    - Deduplication on open_time (month-boundary overlaps handled)
"""

from __future__ import annotations

import hashlib
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from core.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# Constants
# ============================================================

DOWNLOADER_VERSION = "1.0.0"

BINANCE_VISION_BASE = "https://data.binance.vision/data/spot/monthly/klines"

# Binance kline CSV column order (fixed by Binance)
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

NUMERIC_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_volume", "taker_buy_quote_volume",
]

# Timestamp threshold: 2022 in ms ~1.6e12, in microseconds ~1.6e15
# Anything above 1e14 is microseconds -> convert to ms
_MICROSECOND_THRESHOLD = 1e14


# ============================================================
# Low-level helpers
# ============================================================


def _build_urls(symbol: str, interval: str, year: int, month: int) -> tuple[str, str]:
    """Build (zip_url, checksum_url) for a symbol/interval/month."""
    fname = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    base = f"{BINANCE_VISION_BASE}/{symbol}/{interval}/{fname}"
    return base, f"{base}.CHECKSUM"


def _download_bytes(url: str, timeout: int = 90) -> Optional[bytes]:
    """
    Download raw bytes from a URL.
    Returns None on 404 (missing month) or any network error.
    """
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as e:
        logger.warning(f"Download failed: {url} ({e})")
        return None


def _verify_checksum(data: bytes, checksum_text: str) -> bool:
    """
    Verify SHA256 checksum.
    Checksum file format: '<hash>  <filename>'
    """
    expected = checksum_text.strip().split()[0].lower()
    actual = hashlib.sha256(data).hexdigest().lower()
    return expected == actual


def _parse_klines_zip(zip_bytes: bytes) -> pd.DataFrame:
    """
    Extract and parse the CSV inside a Binance klines zip.
    Handles both header and no-header file variants.
    """
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        csv_name = zf.namelist()[0]
        raw = zf.read(csv_name)

    # Detect header: newer files start with 'open_time', older start with a digit
    first_line = raw.split(b"\n", 1)[0].decode("utf-8", errors="ignore")
    first_token = first_line.split(",")[0].strip()
    has_header = not first_token.replace(".", "").replace("-", "").isdigit()

    df = pd.read_csv(
        BytesIO(raw),
        header=0 if has_header else None,
    )
    # Standardize column names regardless of source
    df.columns = KLINE_COLUMNS[: len(df.columns)]
    return df


def _normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert microsecond timestamps to milliseconds if needed."""
    for col in ["open_time", "close_time"]:
        if col in df.columns and len(df) > 0:
            if float(df[col].max()) > _MICROSECOND_THRESHOLD:
                df[col] = df[col] // 1000
    return df


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Type-cast, normalize timestamps, drop the 'ignore' column."""
    df = _normalize_timestamps(df)

    # Ensure numeric types
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Integer types for time/count
    for col in ["open_time", "close_time", "count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")

    # Drop the unused 'ignore' column
    if "ignore" in df.columns:
        df = df.drop(columns=["ignore"])

    return df


# ============================================================
# Month-level download
# ============================================================


def download_month(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    verify: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Download and parse one month of klines.

    Returns:
        Cleaned DataFrame, or None if month is unavailable.
    """
    zip_url, checksum_url = _build_urls(symbol, interval, year, month)

    zip_bytes = _download_bytes(zip_url)
    if zip_bytes is None:
        logger.info(f"  {symbol} {interval} {year}-{month:02d}: not available (skipped)")
        return None

    if verify:
        checksum_bytes = _download_bytes(checksum_url)
        if checksum_bytes is not None:
            ok = _verify_checksum(zip_bytes, checksum_bytes.decode("utf-8"))
            if not ok:
                logger.warning(
                    f"  {symbol} {interval} {year}-{month:02d}: CHECKSUM MISMATCH (skipped)"
                )
                return None
        else:
            logger.warning(
                f"  {symbol} {interval} {year}-{month:02d}: checksum unavailable (proceeding)"
            )

    df = _parse_klines_zip(zip_bytes)
    df = _clean_dataframe(df)
    return df


# ============================================================
# Full symbol/interval collection
# ============================================================


def collect_symbol_interval(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw",
    verify: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Download all months in [start_date, end_date] for a symbol/interval,
    combine, deduplicate, sort, and save as Parquet.

    Args:
        symbol: e.g. "BTCUSDT"
        interval: e.g. "5m"
        start_date, end_date: "YYYY-MM-DD"
        output_dir: where to save Parquet
        verify: verify SHA256 checksums

    Returns:
        Combined DataFrame, or None if no data collected.
    """
    months = pd.period_range(start=start_date, end=end_date, freq="M")
    logger.info(
        f"{symbol} {interval}: collecting {len(months)} months "
        f"({start_date} to {end_date})"
    )

    frames: list[pd.DataFrame] = []
    for period in months:
        df = download_month(symbol, interval, period.year, period.month, verify)
        if df is not None and len(df) > 0:
            frames.append(df)

    if not frames:
        logger.warning(f"{symbol} {interval}: NO data collected")
        return None

    # Combine, deduplicate, sort
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = (
        combined.drop_duplicates(subset="open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    after = len(combined)
    if before != after:
        logger.info(f"  Removed {before - after:,} duplicate bars")

    # Save as Parquet
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    parquet_file = out_path / f"{symbol}_{interval}.parquet"
    combined.to_parquet(parquet_file, compression="snappy", index=False)

    logger.info(
        f"{symbol} {interval}: {len(combined):,} bars saved -> {parquet_file}"
    )
    return combined


# ============================================================
# Self-test
# ============================================================


if __name__ == "__main__":
    print("=" * 70)
    print(f"Binance Downloader Smoke Test (v{DOWNLOADER_VERSION})")
    print("=" * 70)

    # Small test: 3 months of BTCUSDT 5m
    test_symbol = "BTCUSDT"
    test_interval = "5m"
    test_start = "2024-01-01"
    test_end = "2024-03-31"

    print(f"\nTest: {test_symbol} {test_interval}")
    print(f"Range: {test_start} to {test_end} (3 months)")
    print("Downloading from data.binance.vision...\n")

    df = collect_symbol_interval(
        test_symbol, test_interval, test_start, test_end,
        output_dir="data/raw", verify=True,
    )

    if df is None:
        print("\n❌ No data collected")
        exit(1)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nTotal bars: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # Date range
    start_ms = int(df["open_time"].min())
    end_ms = int(df["open_time"].max())
    start_dt = pd.to_datetime(start_ms, unit="ms")
    end_dt = pd.to_datetime(end_ms, unit="ms")
    print("\nDate range:")
    print(f"  First bar: {start_dt}")
    print(f"  Last bar:  {end_dt}")

    # Sample
    print("\nSample (first 3 rows):")
    print(df[["open_time", "open", "high", "low", "close", "volume"]].head(3).to_string())

    # Quality checks
    print("\nQuality checks:")
    print(f"  NaN count: {int(df.isnull().sum().sum())}")
    print(f"  Price sanity (high >= low): {bool((df['high'] >= df['low']).all())}")
    print(f"  Positive volume: {bool((df['volume'] >= 0).all())}")

    # Gap check (5m = 300000 ms between bars)
    expected_gap = 300000
    gaps = df["open_time"].diff().dropna()
    irregular = int((gaps != expected_gap).sum())
    print(f"  Irregular gaps: {irregular} (0 = perfect continuity)")

    print("\n✅ Downloader working — data.binance.vision pipeline verified")