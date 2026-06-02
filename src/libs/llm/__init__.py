# Register built-in LLM providers on import.
from libs.llm.azure_llm import AzureLLM  # noqa: F401  registers "azure"
from libs.llm.deepseek_llm import DeepSeekLLM  # noqa: F401  registers "deepseek"
from libs.llm.openai_llm import OpenAILLM  # noqa: F401  registers "openai"
