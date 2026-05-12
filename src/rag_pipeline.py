"""
RAG pipeline implementation for the legal assistant.

This module implements the Retrieval-Augmented Generation (RAG) workflow
used by the application. It is responsible for:

- Loading vector databases (FAISS) for contracts and legal knowledge
- Creating hybrid retrievers (semantic + keyword search)
- Building the prompt used by the language model
- Running the LangChain pipeline to generate answers
- Post-processing responses (especially Persian date normalization)

Main components:
- FAISS vector search for semantic retrieval
- BM25 keyword retrieval
- Ensemble retriever combining both methods
- Ollama LLM integration
- Persian date normalization and formatting
"""

import os
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from src.llm_handler import get_llm
from src.vector_store_handler import VectorStoreHandler
from src.config import VECTOR_STORE_PATH

from persiantools.jdatetime import JalaliDate
from persiantools.characters import ar_to_fa


def normalize_persian_dates(text: str) -> str:
    date_pattern = re.compile(r'(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})')

    def replace_date(match):
        year, month, day = map(int, match.groups())

        try:
            _ = JalaliDate(year, month, day)
        except Exception:
            return match.group(0)

        year_str = ar_to_fa(str(year))
        month_str = ar_to_fa(f"{month:02d}")
        day_str = ar_to_fa(f"{day:02d}")

        # اصلاح شد: برگرداندن به فرمت استاندارد سال/ماه/روز
        return f"{year_str}/{month_str}/{day_str}"

    return date_pattern.sub(replace_date, text)

class RAGPipeline:
    """
    Core RAG pipeline used for answering legal questions.

    This class handles:

    - Loading FAISS vector databases
    - Creating hybrid retrieval systems
    - Running the LangChain inference pipeline
    - Combining contract and legal knowledge contexts
    """

    def __init__(self):
        """
        Initialize the pipeline components.

        The initialization process:
        - Loads the LLM
        - Initializes vector store handler
        - Loads FAISS databases
        - Builds retrievers for knowledge and contracts
        """

        self.llm = get_llm()

        self.vsh = VectorStoreHandler()

        self.kb_retriever = None
        self.contract_retriever = None

        self.kb_store = None
        self.contract_store = None

        # Load vector databases on startup
        self.load_stores()

    def update_llm(self, model_name: str):
        """
        Dynamically update the active language model.

        This allows switching models at runtime without restarting
        the application.

        Args:
            model_name (str):
                Name of the Ollama model to load.
        """
        self.llm = get_llm(model_name=model_name)

    def _create_hybrid_retriever(self, faiss_store):
        """
        Create a hybrid retriever combining semantic and keyword search.

        The hybrid retriever merges:

        1. FAISS semantic vector search
        2. BM25 keyword-based search

        This improves retrieval accuracy, especially for legal queries
        where both meaning and exact terms matter.

        Args:
            faiss_store:
                Loaded FAISS vector store.

        Returns:
            EnsembleRetriever | None
        """

        if not faiss_store:
            return None

        # Semantic vector retriever
        faiss_retriever = faiss_store.as_retriever(
            search_kwargs={"k": 4}
        )

        # Extract documents from FAISS store
        docs = list(faiss_store.docstore._dict.values())

        if not docs:
            return faiss_retriever

        # Keyword-based BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 4

        # Combine retrievers with weighted scoring
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.6, 0.4],
        )

        return ensemble_retriever

    def load_stores(self):
        """
        Load FAISS vector databases for knowledge base and contracts.

        Two independent vector databases are used:

        - Knowledge base (legal references)
        - Contract documents

        Each store is converted into a hybrid retriever for improved search.
        """

        # Load legal knowledge base
        kb_path = os.path.join(VECTOR_STORE_PATH, "kb")

        if os.path.exists(kb_path):
            try:
                self.kb_store = self.vsh.load_index(load_path=kb_path)

                if self.kb_store:
                    self.kb_retriever = self._create_hybrid_retriever(
                        self.kb_store
                    )

                print("پایگاه دانش قوانین با موفقیت بارگذاری شد.")

            except Exception as e:
                print(f"خطا در بارگذاری پایگاه دانش: {e}")

        # Load contract database
        contract_path = os.path.join(VECTOR_STORE_PATH, "contract")

        if os.path.exists(contract_path):
            try:
                self.contract_store = self.vsh.load_index(
                    load_path=contract_path
                )

                if self.contract_store:
                    self.contract_retriever = (
                        self._create_hybrid_retriever(
                            self.contract_store
                        )
                    )

                print("دیتابیس قرارداد با موفقیت بارگذاری شد.")

            except Exception as e:
                print(f"خطا در بارگذاری قرارداد: {e}")

    def ask_question(self, question: str) -> str:
        """
        Ask a legal question and generate an answer using RAG.

        Workflow:

        1. Retrieve relevant contract sections
        2. Retrieve relevant legal references
        3. Construct a Persian legal prompt
        4. Run the LangChain pipeline
        5. Normalize date formats in the final response

        Args:
            question (str):
                User's question in Persian.

        Returns:
            str:
                Final generated answer.
        """

        # Retrieve relevant contract documents
        if self.contract_retriever:
            contract_docs = self.contract_retriever.invoke(question)

            contract_context = (
                "\n\n".join([doc.page_content for doc in contract_docs])
                if contract_docs
                else "بند مرتبطی در قرارداد یافت نشد."
            )
        else:
            contract_context = "هیچ قراردادی بارگذاری نشده است."

        # Retrieve relevant legal knowledge
        if self.kb_retriever:
            kb_docs = self.kb_retriever.invoke(question)

            kb_context = (
                "\n\n".join([doc.page_content for doc in kb_docs])
                if kb_docs
                else "ماده قانونی مرتبطی یافت نشد."
            )
        else:
            kb_context = "هیچ قانون مرجعی بارگذاری نشده است."

        # ---------------------------------------------------------
        # Persian legal prompt optimized for Llama 3.1 8B
        # Strictly prevents hallucinations, context bleeding, and 
        # enforces strict date/number formatting rules.
        # ---------------------------------------------------------
        prompt = ChatPromptTemplate.from_template(
            """تو یک دستیار حقوقی هوشمند و بسیار دقیق هستی. 
وظیفه تو تحلیل درخواست کاربر فقط و فقط بر اساس اطلاعات داده شده در بخش "قوانین مرجع" و "محتوای قرارداد" است.
هیچ اطلاعاتی خارج از این اسناد تولید نکن.

<قوانین الزامی و ضد هذیان (Strict & Anti-Hallucination Rules)>
۱. عدم تولید داده جعلی: هیچ عدد، تاریخ، نام، شماره قرارداد، شماره بخشنامه یا شناسه جدید نساز. فقط مواردی را ذکر کن که دقیقاً در متون زیر وجود دارد. اگر چیزی نبود، دقیقاً بگو: "این مورد در اسناد یافت نشد."

۲. عدم تعمیم نابجای قوانین: قوانین و بخشنامه های قانون کار فقط برای رابطه کارگر-کارفرما معتبرند. آنها را به قراردادهای پیمانکاری، توسعه نرم افزار، خدماتی، اجاره یا تجاری تعمیم نده (مگر اینکه قرارداد صراحتاً استخدامی باشد).

۳. ممنوعیت اختراع بند: هیچ بند، ماده، حق، تکلیف یا جزئیات جدیدی اختراع نکن.

۴. ارجاع و تناقضات دقیق: تناقض ها (مانند اختلاف تاریخ و مدت) را دقیق و فقط بر اساس داده های واقعی اعلام کن. فقط به شماره ها و اسنادی ارجاع بده که در متن وجود دارند و از ارجاع اشتباه (مثلاً بخشنامه های خیالی) خودداری کن.

۵. اعلام ابهام: در صورت مبهم بودن موضوع در متن، آن را صراحتاً مبهم اعلام کن (مثال: "در متن داور مشخص نشده است").

۶. فرمت تاریخ (مهم): تمامی تاریخ ها را حتماً به فرمت «سال/ماه/روز» بنویس (مثال: «۱۴۰۲/۳/۵»). اعداد را با ارقام فارسی و بدون صفر پیشوندی (مثلاً «۱» نه «۰۱») بنویس.

۷. گیومه و اعداد: تمام اعداد، تاریخ ها، مبالغ و شماره قراردادها را حتماً داخل گیومه فارسی قرار بده (مثال: «۱۴۰۲/۳/۵» یا «۵۰۰۰»). مبالغ حروفی را به عدد تبدیل کن، اما روی آنها عملیات ریاضی انجام نده.

۸. بدون حاشیه: به هیچ عنوان این قوانین را در پاسخ بازگو نکن و فقط جواب نهایی را به زبان فارسی رسمی بنویس.
</قوانین الزامی و ضد هذیان>

--- قوانین مرجع ---
{kb_context}

--- محتوای قرارداد ---
{contract_context}

سوال کاربر:
{question}

پاسخ نهایی و دقیق:"""
        )

        # Build LangChain LCEL pipeline
        # ---------------------------------------------------------
        # The chain injects the retrieved contexts into the prompt, 
        # passes it to the LLM, and parses the output as a string.
        # ---------------------------------------------------------


        # Build LangChain LCEL pipeline
        chain = (
            {
                "contract_context": lambda x: contract_context,
                "kb_context": lambda x: kb_context,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        try:
            # Generate raw response from the LLM
            raw_response = chain.invoke(question)

            # Normalize dates inside the response
            final_response = normalize_persian_dates(raw_response)

            return final_response

        except Exception as e:
            return f"متاسفانه در تولید پاسخ خطایی رخ داد: {str(e)}"
