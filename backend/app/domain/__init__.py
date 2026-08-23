"""Domain layer: entities, DTOs, port interfaces and errors.

This layer has no knowledge of FastAPI, SQLite or OpenAI — it defines *what*
the system does, not *how*. Adapters (repositories, providers, API) depend on
these abstractions, not the other way around.
"""
