"""
Document processing module.

This module is responsible for:

- Loading supported document formats
- Extracting textual content
- Fixing Persian text extraction issues
- Splitting large documents into semantic chunks
- Preparing documents for embedding and RAG pipelines

Supported file formats:
- PDF
- DOCX
- TXT
- CSV
"""

import os
import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader
)


class DocumentProcessor:
    """
    Handle document loading, preprocessing, and chunk generation.

    This class provides utilities for:
    - Loading different document types
    - Fixing Persian number direction issues
    - Enriching document metadata
    - Splitting large texts into smaller chunks for vector databases
    """

    def __init__(self):
        """
        Initialize the document processor and configure the text splitter.

        The chunking strategy is optimized for Persian legal and structured
        documents by using:
        - Large chunk sizes for better context preservation
        - Overlapping chunks for semantic continuity
        - Persian-specific separators such as:
            ماده / تبصره / بند
        """

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\nماده",
                "\nتبصره",
                "\nبند",
                "\n\n",
                "\n",
                ".",
                "؛",
                " ",
                ""
            ],
            length_function=len,
            is_separator_regex=False
        )

    def fix_number_direction(self, text: str) -> str:
        """
        Fix reversed Persian and English numbers extracted from PDF files.

        Some Persian PDF extraction tools reverse numeric sequences during
        text extraction. This method detects numeric patterns and reverses
        them back to their correct order.

        Args:
            text (str):
                Extracted text content.

        Returns:
            str:
                Corrected text with properly ordered numbers.
        """

        if not text:
            return text

        # Match both Persian and English digits
        pattern = r'[0-9۰-۹]+'

        def reverse_match(match):
            """
            Reverse detected numeric strings.

            Args:
                match:
                    Regex match object.

            Returns:
                str:
                    Reversed number string.
            """
            return match.group(0)[::-1]

        return re.sub(pattern, reverse_match, text)

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document based on its file extension.

        Supported formats:
        - PDF
        - DOCX
        - TXT
        - CSV

        The method also:
        - Fixes Persian number direction issues
        - Ensures metadata contains source information

        Args:
            file_path (str):
                Path to the input document.

        Returns:
            List[Document]:
                List of loaded LangChain document objects.
        """

        ext = os.path.splitext(file_path)[1].lower()

        try:
            # Select the appropriate loader based on file type
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)

            elif ext == '.docx':
                loader = Docx2txtLoader(file_path)

            elif ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')

            elif ext == '.csv':
                loader = CSVLoader(file_path, encoding='utf-8')

            else:
                raise ValueError(
                    f"فرمت فایل {ext} پشتیبانی نمی‌شود."
                )

            # Load document content
            documents = loader.load()

            # Extract file name for metadata usage
            file_name = os.path.basename(file_path)

            # Apply preprocessing and metadata enrichment
            for doc in documents:

                # Fix reversed numbers in Persian text
                doc.page_content = self.fix_number_direction(
                    doc.page_content
                )

                # Ensure source metadata exists
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = file_name

            return documents

        except Exception as e:
            print(f"خطا در خواندن فایل {file_path}: {str(e)}")
            return []

    def process_files(self, file_paths: List[str]) -> List[Document]:
        """
        Process multiple files and convert them into text chunks.

        Workflow:
        1. Validate file existence
        2. Load document content
        3. Merge all documents
        4. Split documents into overlapping chunks

        Args:
            file_paths (List[str]):
                List of file paths to process.

        Returns:
            List[Document]:
                List of chunked LangChain document objects.
        """

        all_documents = []

        # Load all available files
        for path in file_paths:

            if os.path.exists(path):
                docs = self.load_document(path)
                all_documents.extend(docs)

            else:
                print(f"فایل یافت نشد: {path}")

        # Return empty list if no documents were loaded
        if not all_documents:
            return []

        # Split documents into smaller overlapping chunks
        chunks = self.text_splitter.split_documents(all_documents)

        return chunks
