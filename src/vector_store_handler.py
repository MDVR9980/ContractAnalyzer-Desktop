"""
Vector store handler module.

This class is responsible for:

- Managing embedding model initialization
- Processing documents into vector chunks
- Creating and saving FAISS vector indices locally
- Loading saved FAISS indices for later retrieval

It supports separate vector databases for:

- 'kb' (knowledge base, i.e. legal regulations)
- 'contract' documents

Uses Ollama embeddings for vectorization and LangChain's FAISS integration.
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from src.document_processor import DocumentProcessor
from src.config import EMBEDDING_MODEL_NAME, VECTOR_STORE_PATH


class VectorStoreHandler:
    """
    Handles vector store creation, saving, and loading.

    Also responsible for embedding model loading and document chunk processing.
    """

    def __init__(self):
        """
        Initialize vector store handler with embedding model and document processor.
        """
        self.db_path = VECTOR_STORE_PATH
        self.embeddings = self.get_embeddings_model()

        # Instantiate document processor for loading and chunking files
        self.doc_processor = DocumentProcessor()

    def get_embeddings_model(self):
        """
        Load Ollama embeddings model.

        Returns:
            OllamaEmbeddings:
                Initialized embeddings model for vectorization.
        """
        return OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

    def process_documents(self, file_paths, store_type="kb"):
        """
        Process a list of files and create separate FAISS vector databases.

        Args:
            file_paths (list[str]):
                List of file paths to process.
            store_type (str, optional):
                Type of the store; either 'kb' for legal knowledge base
                or 'contract' for contracts.
                Defaults to 'kb'.

        Raises:
            ValueError:
                If no textual content could be extracted from the files.
        """
        print(f"در حال پردازش فایل‌ها ({store_type})...")

        # Process and chunk documents using DocumentProcessor
        all_chunks = self.doc_processor.process_files(file_paths)

        if all_chunks:
            # Determine save path based on store type
            save_path = os.path.join(self.db_path, store_type)

            # Create vector index and save it
            self.create_and_save_index(all_chunks, save_path)

            print(f"دیتابیس {store_type} با موفقیت ایجاد و ذخیره شد.")
        else:
            raise ValueError(f"هیچ متنی از فایل‌های {store_type} استخراج نشد.")

    def create_and_save_index(self, document_chunks, save_path):
        """
        Build a FAISS index from document chunks and save locally.

        Args:
            document_chunks (list[Document]):
                List of preprocessed document chunks.
            save_path (str):
                Directory in which to save the FAISS index.

        Returns:
            FAISS:
                Created FAISS vector store instance.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        vector_store = FAISS.from_documents(document_chunks, embedding=self.embeddings)
        vector_store.save_local(save_path)
        return vector_store

    def load_index(self, load_path):
        """
        Load a saved FAISS index from disk.

        Args:
            load_path (str):
                Path to saved FAISS index directory.

        Returns:
            FAISS | None:
                Loaded FAISS vector store instance or None if
                path does not exist or is empty.
        """
        if not os.path.exists(load_path) or not os.listdir(load_path):
            return None

        return FAISS.load_local(
            load_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
