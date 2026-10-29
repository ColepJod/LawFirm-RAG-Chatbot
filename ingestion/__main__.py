"""Run ingestion from the command line: ``python -m ingestion``."""

from __future__ import annotations

import argparse
import json

from config import get_settings
from ingestion.pipeline import ingest_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion",
        description="Ingest supported documents into the local manifest.",
    )
    parser.add_argument("--path", default="data", help="File or directory to ingest.")
    parser.add_argument("--manifest", default=None, help="Override the manifest path.")
    parser.add_argument("--force", action="store_true", help="Re-ingest already seen documents.")
    parser.add_argument(
        "--pdf-parser",
        choices=("pdfmux", "unstructured", "docling"),
        default=None,
        help="Override the configured PDF parser.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    from pathlib import Path

    path = Path(args.path)
    if not path.is_absolute():
        path = settings.project_root / path

    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path is not None and not manifest_path.is_absolute():
        manifest_path = settings.project_root / manifest_path

    result = ingest_directory(
        path,
        manifest_path=manifest_path,
        force=args.force,
        pdf_parser=args.pdf_parser,
    )

    summary = {
        "ingested_documents": len(result.documents),
        "skipped_documents": len(result.skipped),
        "manifest_path": str(result.manifest_path) if result.manifest_path else None,
        "skipped": [str(path) for path in result.skipped],
    }
    print(json.dumps(summary, indent=2))

    # Ingest and index in one pass so a freshly added corpus is immediately
    # usable by retrieval. Indexing is incremental via the index manifest.
    if result.documents:
        from chunking import chunk_documents
        from retrieval import BM25Index, create_vector_store, index_chunks

        settings = get_settings()
        chunks = chunk_documents(result.documents)
        if chunks:
            vector_store = create_vector_store(settings)
            bm25_index = BM25Index(k=settings.top_k)
            index_result = index_chunks(
                chunks,
                vector_store,
                bm25_index,
                manifest_path=settings.project_root / "var" / "index" / "manifest.json",
                chunk_store_path=settings.project_root / "var" / "index" / "chunks.jsonl",
            )
            summary["indexed_chunks"] = index_result.total_count
            summary["new_chunks"] = index_result.new_count

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
