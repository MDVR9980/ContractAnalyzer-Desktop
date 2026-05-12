"""
Application entry point.

This module configures the runtime environment, patches print for
correct Persian display in the terminal, initializes the AI backend,
and launches the PyQt6 user interface.
"""

from __future__ import annotations

import builtins
import ctypes
import os
import sys
import warnings
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication


# ---------------------------------------------------------------------
# Paths and application identity
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
ICON_PATH = BASE_DIR / "data" / "app_icon.ico"

APP_USER_MODEL_ID = "contract.analyzer.desktop.v1"


# ---------------------------------------------------------------------
# Print patch for correct Persian output in terminal
# ---------------------------------------------------------------------

_original_print = builtins.print


def custom_print(*args, **kwargs):
    """
    Print function wrapper that reshapes Arabic/Persian text
    so it appears correctly in terminals that do not handle RTL text well.
    """
    reshaped_args = []

    for arg in args:
        if isinstance(arg, str):
            reshaped_text = arabic_reshaper.reshape(arg)
            bidi_text = get_display(reshaped_text)
            reshaped_args.append(bidi_text)
        else:
            reshaped_args.append(arg)

    _original_print(*reshaped_args, **kwargs)


builtins.print = custom_print


# ---------------------------------------------------------------------
# Environment and runtime configuration
# ---------------------------------------------------------------------

def configure_environment() -> None:
    """
    Configure encoding, warnings, and external library verbosity.
    """
    # Ensure stdout uses UTF-8 whenever possible.
    try:
        if getattr(sys.stdout, "encoding", None) != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        # Some environments may not support reconfigure().
        pass

    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    warnings.filterwarnings("ignore")

    # Make local source modules importable.
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


def configure_windows_app_id() -> None:
    """
    Set the Windows AppUserModelID so the application gets its own
    taskbar icon instead of inheriting python.exe's identity.
    """
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                APP_USER_MODEL_ID
            )
        except Exception:
            # Fail silently if the platform or API is unavailable.
            pass


def create_application() -> QApplication:
    """
    Create and configure the QApplication instance.
    """
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    return app


def initialize_backend():
    """
    Initialize backend services required by the UI.

    Returns
    -------
    tuple
        A tuple containing:
        - rag_pipeline
        - vector_store_handler
    """
    from src.rag_pipeline import RAGPipeline
    from src.vector_store_handler import VectorStoreHandler

    print("در حال بارگذاری مدل‌های هوش مصنوعی (این مرحله ممکن است زمان‌بر باشد)...")

    rag_pipeline = RAGPipeline()
    vector_store_handler = VectorStoreHandler()

    return rag_pipeline, vector_store_handler


def main() -> int:
    """
    Main application entry point.

    Returns
    -------
    int
        Process exit code.
    """
    configure_environment()
    configure_windows_app_id()

    print("در حال بارگذاری برنامه...")
    app = create_application()

    try:
        rag_pipeline, vector_store_handler = initialize_backend()
    except Exception as exc:
        print(f"خطا در بارگذاری هسته هوش مصنوعی: {exc}")
        return 1

    print("در حال آماده‌سازی رابط کاربری...")

    from gui.main_window import MainWindow

    window = MainWindow(rag_pipeline, vector_store_handler)
    window.show()

    print("برنامه با موفقیت اجرا شد.")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
