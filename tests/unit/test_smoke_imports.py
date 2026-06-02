"""Smoke tests: verify all top-level packages can be imported."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def test_import_mcp_server():
    import mcp_server


def test_import_core():
    import core


def test_import_ingestion():
    import ingestion


def test_import_libs():
    import libs


def test_import_observability():
    import observability


def test_import_core_submodules():
    from core import query_engine, response, trace


def test_import_ingestion_submodules():
    from ingestion import chunking, transform, embedding, storage


def test_import_libs_submodules():
    from libs import loader, llm, embedding, splitter, vector_store, reranker, evaluator
