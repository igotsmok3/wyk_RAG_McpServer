#!/usr/bin/env python3
"""Ingest documents into the WYK RAG pipeline.

Usage:
    python scripts/ingest.py --path /path/to/doc.pdf --collection my_kb
    python scripts/ingest.py --path /path/to/docs/ --collection my_kb --force
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.settings import load_settings
from ingestion.pipeline import IngestionPipeline


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _collect_files(path: str) -> list[str]:
    """Return sorted list of PDF files from a file path or directory."""
    p = Path(path)
    if p.is_file():
        return [str(p.resolve())]
    if p.is_dir():
        files = sorted(str(f.resolve()) for f in p.rglob("*.pdf"))
        return files
    raise FileNotFoundError(f"Path not found: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into WYK RAG pipeline.")
    parser.add_argument("--path", required=True, help="File or directory to ingest (PDF)")
    parser.add_argument(
        "--collection", default="default", help="Collection name (default: default)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-ingest even if already processed"
    )
    parser.add_argument(
        "--config", default="config/settings.yaml", help="Path to settings.yaml"
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    _setup_logging(settings.observability.log_level)
    logger = logging.getLogger(__name__)

    try:
        files = _collect_files(args.path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not files:
        print("No PDF files found at the given path.", file=sys.stderr)
        return 1

    pipeline = IngestionPipeline(settings)

    total = len(files)
    n_processed = 0
    n_skipped = 0
    n_failed = 0

    for idx, file_path in enumerate(files, 1):
        print(f"[{idx}/{total}] {file_path}")
        try:
            result = pipeline.run(file_path, collection=args.collection, force=args.force)
            if result.skipped:
                print("  → skipped (already ingested; use --force to re-ingest)")
                n_skipped += 1
            else:
                print(
                    f"  → done: chunks={result.chunk_count}"
                    f" records={result.record_count}"
                    f" images={result.image_count}"
                )
                n_processed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  → FAILED: {exc}", file=sys.stderr)
            logger.exception("Ingestion failed for %s", file_path)
            n_failed += 1

    print(f"\nSummary: {n_processed} processed, {n_skipped} skipped, {n_failed} failed")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
