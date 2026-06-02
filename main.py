"""WYK RAG MCP Server - main entry point."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from core.settings import load_settings
from observability.logger import get_logger

logger = get_logger(__name__)


def main():
    settings = load_settings("config/settings.yaml")
    logger.info("Settings loaded: llm=%s, embedding=%s", settings.llm.provider, settings.embedding.provider)


if __name__ == "__main__":
    main()
