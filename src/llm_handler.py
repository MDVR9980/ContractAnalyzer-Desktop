"""
LLM and prompt configuration module.

This module is responsible for:

- Creating and configuring the Ollama-based chat model
- Defining the system and human prompts for the legal assistant
- Providing reusable helpers for the rest of the application

Notes:
- Uses `langchain_ollama` to avoid `LangChainDeprecationWarning`.
- All user-facing texts and instructions for the model are written in Persian,
  while internal documentation and comments are in English.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from src.config import OLLAMA_MODEL


def get_llm(model_name: str = OLLAMA_MODEL, temperature: float = 0.0) -> ChatOllama:
    """
    Create and configure the Ollama chat LLM instance.

    Args:
        model_name (str, optional):
            Name of the Ollama model to use.
            Defaults to the global `OLLAMA_MODEL` from config.
        temperature (float, optional):
            Sampling temperature for controlling randomness.
            Lower values make output more deterministic.
            Defaults to 0.0.

    Returns:
        ChatOllama:
            Configured LangChain-compatible chat LLM.
    """
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        top_p=0.1,
    )
    return llm


def get_contract_prompt() -> ChatPromptTemplate:
    """
    Build and return the main prompt template for the legal assistant.

    The system prompt:
    - Instructs the model to act as a precise legal assistant
    - Forces it to only use the provided context (documents)
    - Specifies formatting rules for dates, numbers, and money
    - Enforces strict, formal Persian output

    Placeholders:
        {context}  : Textual context retrieved from documents (RAG results)
        {question} : User's question in Persian

    Returns:
        ChatPromptTemplate:
            Composed chat prompt with system + human messages.
    """
    # System prompt in Persian for better alignment with small models
    system_template = """تو یک دستیار حقوقی هوشمند و بسیار دقیق هستی. 
وظیفه تو تحلیل درخواست کاربر فقط و فقط بر اساس متن ارائه شده است.

<قوانین الزامی>
۱. فقط از متن داده شده استفاده کن. اگر جواب در متن نیست، دقیقاً بنویس: "این مورد در اسناد یافت نشد." (هیچ توضیح اضافه‌ای نده).
۲. تمام مبالغ، مدت‌زمان‌ها و تاریخ‌هایی که به حروف نوشته شده‌اند را در پاسخ به عدد تبدیل کن.
۳. فرمت تاریخ: تاریخ‌ها را حتماً به شکل «سال/ماه/روز» بنویس (مثلاً «1402/1/1»). صفرهای ابتدای ماه و روز را حذف کن.
۴. استفاده از گیومه: تمام اعداد، تاریخ‌ها و مبالغ را حتماً داخل گیومه فارسی بگذار (مانند «1402/1/1» یا «10000»).
۵. فرآیند تبدیل را توضیح نده و فقط پاسخ نهایی را به فارسی رسمی بنویس.
</قوانین الزامی>

Context:
{context}
"""

    # Human message template: inject user question
    human_template = "سوال کاربر: {question}"

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template),
        ]
    )

    return prompt
