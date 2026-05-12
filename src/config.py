"""
Application configuration module.

This file contains all global configuration variables used across the
application, including:

- Project directory paths
- Vector database settings
- Available LLM models
- Ollama server configuration
- Embedding model configuration
- Text chunking parameters

All configuration values are centralized here to simplify maintenance
and future scalability.
"""

import os

# ============================================================================
# Project Base Paths
# ============================================================================

# Root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory used for storing application data
DATA_DIR = os.path.join(BASE_DIR, "data")

# Path for the FAISS vector database storage
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "faiss_db")


# ============================================================================
# Available LLM Models
# ============================================================================

# List of supported Ollama models available inside the application
AVAILABLE_MODELS = [
    "llama3.1:8b",
    "deepseek-r1:14b",
    "qwen3:14b",
]


# ============================================================================
# Default Model Configuration
# ============================================================================

# Default model selected at application startup
OLLAMA_MODEL = AVAILABLE_MODELS[0]

# Base URL of the local Ollama server
OLLAMA_BASE_URL = "http://localhost:11434"


# ============================================================================
# Embedding Configuration
# ============================================================================

# Embedding model used for vector generation and semantic search
EMBEDDING_MODEL_NAME = "embeddinggemma:latest"


# ============================================================================
# Text Chunking Configuration
# ============================================================================

# Maximum size of each text chunk used in RAG processing
CHUNK_SIZE = 500

# Number of overlapping characters between consecutive chunks
# to preserve context continuity
CHUNK_OVERLAP = 50
