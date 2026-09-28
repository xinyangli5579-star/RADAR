"""Shared helpers for reading the raw price and news files."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

DATE_FMT = "%Y-%m-%d"
EMPTY_TEXT = {"", "nan", "none", "null"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, DATE_FMT)


def parse_date_with_timezone(s: str) -> datetime:
    """Parse dates such as '2020-06-05 06:30:54 UTC'; the time zone is ignored."""
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {s!r}")


def fmt_date(d: datetime) -> str:
    return d.strftime(DATE_FMT)


def in_range(d: datetime, start: datetime, end: datetime) -> bool:
    return start <= d <= end


def load_universe(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        tickers = json.load(f)["tickers"]
    if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
        raise ValueError(f"Invalid universe format: {path}")
    return tickers


def save_universe(path: str, tickers: List[str], meta: Optional[dict] = None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"tickers": tickers, "meta": meta or {}}, f, ensure_ascii=False, indent=2, sort_keys=True)


@dataclass(frozen=True)
class SP500WideSchema:
    """Header of the price file: row 1 tickers, row 2 fields, row 3 a 'Date' placeholder."""

    tickers: List[str]
    fields_per_ticker: List[str]
    col_index: Dict[Tuple[str, str], int]  # (ticker, field) -> column index


def parse_sp500_full_multiindex_schema(path: str) -> SP500WideSchema:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        row1, row2, row3 = next(reader), next(reader), next(reader)
    if not row1 or not row2 or not row3:
        raise ValueError("Unexpected empty header in sp500 file")
    if row3[0].strip() != "Date":
        raise ValueError(f"Expected third header first cell to be 'Date', got: {row3[0]!r}")

    tickers = [c.strip() for c in row1[1:]]
    fields = [c.strip() for c in row2[1:]]
    if len(tickers) != len(fields):
        raise ValueError(f"Header length mismatch: tickers={len(tickers)} fields={len(fields)}")
    if not tickers:
        raise ValueError("No tickers found in sp500 header")

    col_index = {(t, fld): i for i, (t, fld) in enumerate(zip(tickers, fields), start=1) if t and fld}
    fields_per_ticker = []
    for t, fld in zip(tickers, fields):
        if t != tickers[0]:
            break
        fields_per_ticker.append(fld)
    unique = list(dict.fromkeys(t for t in tickers if t))
    return SP500WideSchema(tickers=unique, fields_per_ticker=fields_per_ticker, col_index=col_index)


def read_news_header_tickers(path: str) -> List[str]:
    """Tickers of a wide news file (first column 'date', one column per ticker)."""
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        header = next(csv.reader(f))
    if not header or header[0].strip() != "date":
        raise ValueError("Expected first column to be 'date' in the wide news file")
    return [h.strip() for h in header[1:]]


def build_news_column_index(path: str, wanted_tickers: Sequence[str]) -> Tuple[int, Dict[str, int]]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        header = next(csv.reader(f))
    col_map = {name: i for i, name in enumerate(header)}
    if "date" not in col_map:
        raise ValueError("news header missing 'date'")
    return col_map["date"], {t: col_map[t] for t in wanted_tickers if t in col_map}


def _long_format_columns(header: List[str]) -> Dict[str, int]:
    col_map = {name.strip(): i for i, name in enumerate(header)}
    for required in ("Date", "Stock_symbol"):
        if required not in col_map:
            raise ValueError(f"news header missing '{required}' column")
    return col_map


def read_news_long_format_tickers(path: str) -> List[str]:
    """Tickers of a long news file (FNSPID layout with a 'Stock_symbol' column)."""
    tickers = set()
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        symbol_idx = _long_format_columns(next(reader))["Stock_symbol"]
        for row in reader:
            if len(row) > symbol_idx:
                symbol = row[symbol_idx].strip()
                if symbol.lower() not in EMPTY_TEXT:
                    tickers.add(symbol)
    return sorted(tickers)


def iter_news_rows_long_format(path: str, wanted_tickers: Sequence[str], start: datetime,
                               end: datetime) -> Iterator[Tuple[str, Dict[str, str]]]:
    """Yield (date, {ticker: text}) in date order; same-day articles of a ticker are concatenated."""
    wanted = set(wanted_tickers)
    buffer: Dict[str, Dict[str, List[str]]] = {}
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        cols = _long_format_columns(next(reader))
        date_idx, symbol_idx = cols["Date"], cols["Stock_symbol"]
        article_idx, title_idx = cols.get("Article", -1), cols.get("Article_title", -1)
        for row in reader:
            if len(row) <= max(date_idx, symbol_idx) or not row[date_idx].strip():
                continue
            try:
                d = parse_date_with_timezone(row[date_idx])
            except ValueError:
                continue
            symbol = row[symbol_idx].strip()
            if not in_range(d, start, end) or symbol not in wanted:
                continue
            text = row[article_idx].strip() if 0 <= article_idx < len(row) else ""
            if not text and 0 <= title_idx < len(row):
                text = row[title_idx].strip()
            if text.lower() in EMPTY_TEXT:
                continue
            buffer.setdefault(fmt_date(d), {}).setdefault(symbol, []).append(text)
    for date_str in sorted(buffer):
        yield date_str, {ticker: " ".join(texts) for ticker, texts in buffer[date_str].items()}


def iter_news_rows(path: str, wanted_tickers: Sequence[str], start: datetime,
                   end: datetime) -> Iterator[Tuple[str, Dict[str, str]]]:
    """Yield (date, {ticker: text}) from a wide news file."""
    date_idx, idx_map = build_news_column_index(path, wanted_tickers)
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[date_idx].strip():
                continue
            ds = row[date_idx].strip()
            try:
                d = parse_date(ds)
            except ValueError:
                continue
            if not in_range(d, start, end):
                continue
            out = {}
            for t, j in idx_map.items():
                text = row[j].strip() if j < len(row) and row[j] is not None else ""
                if text.lower() not in EMPTY_TEXT:
                    out[t] = text
            yield ds, out
