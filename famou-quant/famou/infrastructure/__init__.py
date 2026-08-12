"""
Infrastructure services for Famou 2.0.

Sub-packages:
- llm: LLM client interfaces and implementations
- env: Execution environment interfaces and implementations
- storage: Data persistence interfaces and implementations
- logger: Logging interfaces and implementations
- embedding: Embedding client interfaces and implementations

Factory:
- factory: InfraFactory for creating services from config

Import from sub-packages directly:
    from famou.infrastructure.llm import LLMClient, OpenAIClient
    from famou.infrastructure.storage import DataService, LocalStorage
    from famou.infrastructure.env import ExecutionEnvironment, LocalEnv
    from famou.infrastructure.logger import Logger, LocalLogger
    from famou.infrastructure.embedding import EmbeddingClient
    from famou.infrastructure.factory import InfraFactory, Infrastructure
"""
