"""
Legal RAG Assistant UI built with PyQt6.

This module provides a dark-themed Persian RTL desktop interface for:
- selecting legal knowledge-base files,
- selecting active contract files,
- processing documents into vector stores,
- chatting with a RAG pipeline,
- switching AI models dynamically,
- displaying formatted Persian responses with RTL-friendly rendering.

The visible UI text remains in Persian, while all internal documentation
and comments are written in English for maintainability and team use.
"""

import os
import re
from pathlib import Path
import markdown
import src.config as config

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTextBrowser,
    QLineEdit,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QComboBox,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QSettings
from PyQt6.QtGui import QFont, QTextOption, QIcon
import arabic_reshaper
from bidi.algorithm import get_display

BASE_DIR = Path(__file__).resolve().parent.parent
ICON_PATH = BASE_DIR / "data" / "app_icon.ico"

# ----------------- Threads ----------------- #
class ProcessDocsThread(QThread):
    """
    Background worker thread for processing uploaded documents.

    This thread delegates document ingestion to the vector store handler
    in order to avoid blocking the main GUI thread while files are being
    processed and embedded.

    Signals:
        finished(str): Emitted when processing completes successfully.
        error(str): Emitted when an exception occurs during processing.
    """

    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, vsh_instance, kb_files, contract_files):
        """
        Initialize the document processing thread.

        Args:
            vsh_instance: Vector store handler instance responsible for
                processing and storing documents.
            kb_files (list[str]): List of knowledge-base file paths.
            contract_files (list[str]): List of contract file paths.
        """
        super().__init__()
        self.vsh = vsh_instance
        self.kb_files = kb_files
        self.contract_files = contract_files

    def run(self):
        """
        Execute document processing in the background.

        The method processes knowledge-base files first and contract files
        second, if they exist. Any exception is caught and emitted through
        the error signal.
        """
        try:
            # Process knowledge-base documents if any were selected.
            if self.kb_files:
                self.vsh.process_documents(self.kb_files, store_type="kb")

            # Process contract documents if any were selected.
            if self.contract_files:
                self.vsh.process_documents(self.contract_files, store_type="contract")

            # Notify the UI that processing finished successfully.
            self.finished.emit("اسناد با موفقیت پردازش و دیتابیس‌ها به‌روزرسانی شدند.")
        except Exception as e:
            # Forward any processing error to the UI thread.
            self.error.emit(f"خطا در پردازش: {str(e)}")


class AIChatThread(QThread):
    """
    Background worker thread for sending a query to the RAG pipeline.

    This thread executes question answering outside the GUI thread so that
    the interface remains responsive while waiting for the AI response.

    Signals:
        finished(str): Emitted with the AI response text.
        error(str): Emitted when an exception occurs during inference.
    """

    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, pipeline, query):
        """
        Initialize the chat worker thread.

        Args:
            pipeline: RAG pipeline instance used for question answering.
            query (str): User or system-generated query to send.
        """
        super().__init__()
        self.pipeline = pipeline
        self.query = query

    def run(self):
        """
        Execute the AI query in the background.

        The method calls the RAG pipeline's ask_question method and emits
        the returned response. Any exception is captured and reported to
        the main UI through the error signal.
        """
        try:
            # Ask the pipeline to generate an answer for the query.
            response = self.pipeline.ask_question(self.query)

            # Return the generated response to the UI thread.
            self.finished.emit(response)
        except Exception as e:
            # Forward any inference or communication error.
            self.error.emit(f"خطا در ارتباط با هوش مصنوعی: {str(e)}")


# ----------------- Main UI ----------------- #
class MainWindow(QWidget):
    """
    Main application window for the legal AI assistant.

    This widget provides:
    - a sidebar for model selection and document management,
    - a main chat panel for conversation with the assistant,
    - persistent storage of selected file paths,
    - asynchronous document processing and AI interaction,
    - RTL-aware Persian message rendering.

    Args:
        rag_pipeline: Main retrieval-augmented generation pipeline.
        vector_store_handler: Handler used to process and persist document vectors.
    """

    def __init__(self, rag_pipeline, vector_store_handler):
        """
        Initialize the main window and its dependencies.

        Args:
            rag_pipeline: Pipeline object for model updates and QA.
            vector_store_handler: Backend object for document ingestion.
        """
        super().__init__()

        # Store backend dependencies.
        self.rag_pipeline = rag_pipeline
        self.vsh = vector_store_handler

        # Store selected file paths for each category.
        self.kb_files = []
        self.contract_files = []

        # Create persistent application settings storage.
        self.settings = QSettings("LegalAITools", "RAGAssistant")

        # Configure base window properties.
        self.setWindowTitle("دستیار هوشمند حقوقی")
        self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(1150, 750)

        # Set overall layout direction to RTL for Persian UI.
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

        # Apply a base application font.
        app_font = QFont("Segoe UI", 10)
        self.setFont(app_font)

        # Build UI widgets and layouts.
        self.init_ui()

        # Apply custom dark stylesheet.
        self.apply_dark_theme()

        # Load previously saved file selections from settings.
        self.load_saved_files()

        # Try loading pre-existing vector stores on startup if supported.
        try:
            if hasattr(self.rag_pipeline, 'load_stores'):
                self.rag_pipeline.load_stores()
        except Exception as e:
            print(f"بارگذاری اولیه دیتابیس انجام نشد: {str(e)}")

    def init_ui(self):
        """
        Build and arrange all user interface components.

        This method creates the sidebar, model selector, file lists,
        processing controls, chat display, and message input area.
        """
        # Re-assert RTL layout direction for safety.
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

        # Create the top-level horizontal layout.
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # ================= Sidebar =================
        # Create a vertical layout for the left/right sidebar panel.
        sidebar_layout = QVBoxLayout()

        # Create the AI model selection group.
        model_group = QGroupBox("🤖 مدل هوش مصنوعی")
        model_layout = QVBoxLayout()

        # Create model selector combo box.
        self.combo_models = QComboBox()
        self.combo_models.addItems(config.AVAILABLE_MODELS)
        self.combo_models.setCurrentText(config.OLLAMA_MODEL)
        self.combo_models.setStyleSheet(
            "background-color: #181825; border: 1px solid #313244; "
            "padding: 5px; border-radius: 5px; color: #cdd6f4;"
        )

        # Connect model change event to backend update.
        self.combo_models.currentTextChanged.connect(self.change_model)

        # Add the combo box to the model group.
        model_layout.addWidget(self.combo_models)
        model_group.setLayout(model_layout)

        # Create the knowledge-base file management group.
        kb_group = QGroupBox("📚 پایگاه دانش (قوانین مرجع)")
        kb_layout = QVBoxLayout()

        # List widget to display selected legal reference files.
        self.kb_list_widget = QListWidget()

        # Button for adding knowledge-base files.
        btn_add_kb = QPushButton("➕ افزودن فایل قانون")
        btn_add_kb.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_add_kb.clicked.connect(
            lambda: self.add_files(self.kb_list_widget, self.kb_files, "kb")
        )

        # Add KB widgets to layout.
        kb_layout.addWidget(self.kb_list_widget)
        kb_layout.addWidget(btn_add_kb)
        kb_group.setLayout(kb_layout)

        # Create the contract file management group.
        contract_group = QGroupBox("📝 قراردادهای جاری")
        contract_layout = QVBoxLayout()

        # List widget to display selected contract files.
        self.contract_list_widget = QListWidget()

        # Button for adding contract files.
        btn_add_contract = QPushButton("➕ افزودن فایل قرارداد")
        btn_add_contract.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_add_contract.clicked.connect(
            lambda: self.add_files(self.contract_list_widget, self.contract_files, "contract")
        )

        # Add contract widgets to layout.
        contract_layout.addWidget(self.contract_list_widget)
        contract_layout.addWidget(btn_add_contract)
        contract_group.setLayout(contract_layout)

        # Create document processing button.
        self.btn_process = QPushButton("⚙️ پردازش اسناد و ساخت دیتابیس")
        self.btn_process.setMinimumHeight(45)
        self.btn_process.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_process.setProperty("class", "primary-btn")
        self.btn_process.clicked.connect(self.process_documents)

        # Assemble sidebar layout.
        sidebar_layout.addWidget(model_group)
        sidebar_layout.addWidget(kb_group)
        sidebar_layout.addWidget(contract_group)
        sidebar_layout.addWidget(self.btn_process)

        # ================= Main Chat Area =================
        # Create the main chat section layout.
        chat_layout = QVBoxLayout()

        # Header label for the chat panel.
        chat_header = QLabel("💬 محیط گفتگوی هوشمند")
        chat_header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))

        # Create the rich text chat display.
        self.chat_display = QTextBrowser()
        self.chat_display.setOpenExternalLinks(True)
        self.chat_display.setPlaceholderText(
            "سیستم آماده است. اسناد خود را اضافه کنید، پردازش کنید و سپس سوال بپرسید..."
        )

        # Configure RTL display behavior for Persian text.
        self.chat_display.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.chat_display.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop
        )

        # Apply default text option settings to the underlying document.
        doc = self.chat_display.document()
        text_option = QTextOption()
        text_option.setTextDirection(Qt.LayoutDirection.RightToLeft)
        text_option.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        doc.setDefaultTextOption(text_option)

        # Add an initial welcome message to the chat view.
        self.append_chat_message(
            "سیستم",
            "سلام! من دستیار حقوقی شما هستم. پایگاه دانش قوانین آماده است. قرارداد خود را بارگذاری کنید و سوال بپرسید.",
            "#a6e3a1",
            is_system=True,
        )

        # Create the bottom input row.
        input_layout = QHBoxLayout()

        # Create user input field.
        self.msg_input = QLineEdit()
        self.msg_input.setPlaceholderText("درخواست یا سوال خود را اینجا تایپ کنید...")
        self.msg_input.setMinimumHeight(45)

        # Allow pressing Enter to send the message.
        self.msg_input.returnPressed.connect(self.send_message)

        # Create send button.
        self.btn_send = QPushButton("ارسال 🚀")
        self.btn_send.setMinimumHeight(45)
        self.btn_send.setMinimumWidth(100)
        self.btn_send.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_send.setProperty("class", "primary-btn")
        self.btn_send.clicked.connect(self.send_message)

        # Assemble input row.
        input_layout.addWidget(self.msg_input)
        input_layout.addWidget(self.btn_send)

        # Assemble chat layout.
        chat_layout.addWidget(chat_header)
        chat_layout.addWidget(self.chat_display)
        chat_layout.addLayout(input_layout)

        # Wrap the sidebar in a widget so width constraints can be applied.
        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar_layout)
        sidebar_widget.setMinimumWidth(320)
        sidebar_widget.setMaximumWidth(380)

        # Add chat area and sidebar to the main layout.
        main_layout.addLayout(chat_layout, 3)
        main_layout.addWidget(sidebar_widget, 1)

    def load_saved_files(self):
        """
        Load previously selected file paths from persistent settings.

        Existing file paths are restored into their respective internal
        lists and displayed in the associated list widgets if the files
        still exist on disk.
        """
        # Retrieve saved file lists from settings storage.
        saved_kb = self.settings.value("kb_files", [])
        saved_contracts = self.settings.value("contract_files", [])

        # Restore saved knowledge-base files if valid.
        if isinstance(saved_kb, list):
            for f in saved_kb:
                if os.path.exists(f):
                    self.kb_files.append(f)
                    self.create_list_item(self.kb_list_widget, f, self.kb_files)

        # Restore saved contract files if valid.
        if isinstance(saved_contracts, list):
            for f in saved_contracts:
                if os.path.exists(f):
                    self.contract_files.append(f)
                    self.create_list_item(self.contract_list_widget, f, self.contract_files)

    def closeEvent(self, event):
        """
        Persist selected file paths before the window closes.

        Args:
            event: Qt close event object.
        """
        # Save currently selected KB and contract files.
        self.settings.setValue("kb_files", self.kb_files)
        self.settings.setValue("contract_files", self.contract_files)

        # Continue with the default close behavior.
        super().closeEvent(event)

    def change_model(self, model_name):
        """
        Update the currently active AI model in the RAG pipeline.

        Args:
            model_name (str): The model selected by the user.
        """
        try:
            # Ask the backend pipeline to switch the active model.
            self.rag_pipeline.update_llm(model_name)

            # Inform the user that the model change was successful.
            self.append_chat_message(
                "سیستم",
                f"مدل با موفقیت به «{model_name}» تغییر یافت.",
                "#f9e2af",
                is_system=True,
            )
        except Exception as e:
            # Report model switching errors to the chat view.
            self.append_chat_message(
                "سیستم",
                f"خطا در تغییر مدل: {str(e)}",
                "#f38ba8",
                is_system=True,
            )

    def apply_dark_theme(self):
        """
        Apply the custom dark mode stylesheet to the entire window.

        The stylesheet covers the overall widget palette, grouped sections,
        list widgets, buttons, text browser, input controls, and scrollbars.
        """
        dark_stylesheet = """
        QWidget {
            background-color: #1e1e2e;
            color: #cdd6f4;
            font-family: Tahoma, 'Segoe UI', sans-serif;
            font-size: 14px;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 1px solid #45475a;
            border-radius: 8px;
            margin-top: 15px;
            padding-top: 15px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 10px;
            color: #89b4fa;
        }
        
        QListWidget {
            background-color: #181825;
            border: 1px solid #313244;
            border-radius: 6px;
            padding: 5px;
        }
        QListWidget::item {
            border-bottom: 1px solid #313244;
        }
        QListWidget::item:selected {
            background-color: #313244;
            border-radius: 4px;
        }
        
        QPushButton {
            background-color: #313244;
            color: #cdd6f4;
            border: none;
            border-radius: 6px;
            padding: 8px 15px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #45475a;
        }
        QPushButton:pressed {
            background-color: #585b70;
        }
        
        QPushButton[class="primary-btn"] {
            background-color: #89b4fa;
            color: #11111b;
        }
        QPushButton[class="primary-btn"]:hover {
            background-color: #b4befe;
        }
        QPushButton[class="primary-btn"]:disabled {
            background-color: #45475a;
            color: #a6adc8;
        }
        
        QPushButton#DeleteBtn { 
            background-color: #f38ba8; 
            color: #11111b; 
            padding: 2px; 
            font-size: 12px; 
            border-radius: 4px; 
        }
        QPushButton#DeleteBtn:hover { 
            background-color: #eba0ac; 
        }
        
        QTextBrowser {
            background-color: #181825;
            border: 1px solid #313244;
            border-radius: 8px;
            padding: 10px;
            line-height: 1.5;
        }
        
        QLineEdit {
            background-color: #181825;
            border: 1px solid #313244;
            border-radius: 8px;
            padding: 0 15px;
        }
        QLineEdit:focus {
            border: 1px solid #89b4fa;
        }
        
        QScrollBar:vertical {
            border: none;
            background: #181825;
            width: 10px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background: #45475a;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover {
            background: #585b70;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
        """
        # Apply the stylesheet to the current window.
        self.setStyleSheet(dark_stylesheet)

    def add_files(self, list_widget, file_list, file_type):
        """
        Open a file picker and add selected files to the target list widget.

        Args:
            list_widget (QListWidget): The list widget that displays selected files.
            file_list (list): The internal list that stores selected file paths.
            file_type (str): Logical file category ("kb" or "contract") used to customize the dialog.
        """

        # Set dialog title and file filter based on the file category.
        if file_type == "kb":
            dialog_title = "Select Knowledge Base Files"
            file_filter = "Knowledge Base Files (*.pdf *.docx *.txt)"
        elif file_type == "contract":
            dialog_title = "Select Contract Files"
            file_filter = "Contract Files (*.pdf *.docx *.txt)"
        else:
            dialog_title = "Select Files"
            file_filter = "All Supported Files (*.pdf *.docx *.txt)"

        # Open the file selection dialog.
        files, _ = QFileDialog.getOpenFileNames(
            self,
            dialog_title,
            "",
            file_filter
        )

        # Add only unique files to the internal list and UI.
        for file in files:
            if file not in file_list:
                file_list.append(file)
                self.create_list_item(list_widget, file, file_list)

    def create_list_item(self, list_widget, file_path, file_list):
        """
        Create a custom list item with file name and delete button.

        Args:
            list_widget: Target QListWidget.
            file_path (str): Full file path to display.
            file_list (list[str]): Backing list that stores file paths.
        """
        # Create a new list item container.
        item = QListWidgetItem(list_widget)

        # Create a custom widget for richer item layout.
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 2, 5, 2)

        # Display only the file name while keeping full path in tooltip.
        lbl = QLabel(os.path.basename(file_path))
        lbl.setToolTip(file_path)

        # Create a delete button for removing this file from the list.
        btn_delete = QPushButton("✖")
        btn_delete.setObjectName("DeleteBtn")
        btn_delete.setFixedSize(24, 24)
        btn_delete.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_delete.clicked.connect(
            lambda: self.remove_item(list_widget, item, file_path, file_list)
        )

        # Arrange item contents.
        layout.addWidget(lbl)
        layout.addStretch()
        layout.addWidget(btn_delete)

        # Set the visual size of the item to match the custom widget.
        item.setSizeHint(widget.sizeHint())
        list_widget.setItemWidget(item, widget)

    def remove_item(self, list_widget, item, file_path, file_list):
        """
        Remove a file entry from both the widget and the internal list.

        Args:
            list_widget: Source QListWidget.
            item: QListWidgetItem to remove.
            file_path (str): File path to remove from storage list.
            file_list (list[str]): Backing Python list of file paths.
        """
        # Find the row of the target item.
        row = list_widget.row(item)

        # Remove the visual item from the list widget.
        list_widget.takeItem(row)

        # Remove the corresponding path from the data list if present.
        if file_path in file_list:
            file_list.remove(file_path)

    def process_documents(self):
        """
        Start asynchronous processing of selected documents.

        This method validates that at least one file exists, updates the UI
        state, and launches a background thread for vectorization/storage.
        """
        # Prevent processing when no files have been selected.
        if not self.kb_files and not self.contract_files:
            self.show_custom_msg(
                "اخطار",
                "لطفاً ابتدا حداقل یک فایل قانون یا قرارداد برای پردازش انتخاب کنید.",
                "warning"
            )
            return

        # Disable the processing button while work is in progress.
        self.btn_process.setEnabled(False)
        self.btn_process.setText("⏳ در حال پردازش و استخراج بردارها...")

        # Inform the user that processing has started.
        self.append_chat_message(
            "سیستم",
            "در حال پردازش اسناد... لطفاً شکیبا باشید.",
            "#f9e2af",
            is_system=True,
        )

        # Create and start the document processing worker thread.
        self.process_thread = ProcessDocsThread(self.vsh, self.kb_files, self.contract_files)
        self.process_thread.finished.connect(self.on_process_finished)
        self.process_thread.error.connect(self.on_process_error)
        self.process_thread.start()

    def on_process_finished(self, msg):
        """
        Handle the successful completion of the document processing workflow.

        This method is triggered when the background document-processing thread
        finishes without errors. It restores the UI state, reloads the vector
        stores used by the RAG pipeline, notifies the user about the successful
        operation, and automatically starts an initial legal analysis of the
        processed contract.

        Workflow performed by this method:
        1. Re-enable the processing button and restore its original label.
        2. Reload vector stores so newly processed documents become searchable.
        3. Notify the user via the chat panel and a success message dialog.
        4. Inform the user that an automatic legal analysis will begin.
        5. Temporarily disable chat input to prevent overlapping queries.
        6. Launch an AI-powered legal analysis in a background thread.

        Args:
            msg (str): Success message emitted by the document processing worker.
        """

        # Re-enable the document processing button and restore its original label.
        self.btn_process.setEnabled(True)
        self.btn_process.setText("⚙️ پردازش اسناد و ساخت دیتابیس")

        try:
            # Reload the vector stores so that the newly processed documents
            # become immediately available to the RAG pipeline for retrieval.
            self.rag_pipeline.load_stores()

            # Display the success message inside the chat interface.
            # System messages use a dedicated color to distinguish them from user messages.
            self.append_chat_message("سیستم", msg, "#a6e3a1", is_system=True)

            # Show a styled success dialog to visually notify the user.
            self.show_custom_msg("موفقیت", msg, "success")

            # Inform the user that an automatic legal analysis is about to start.
            self.append_chat_message(
                "سیستم",
                "🔍 در حال تحلیل خودکار قرارداد و استخراج مغایرت‌ها... لطفاً چند لحظه صبر کنید.",
                "#f9e2af",
                is_system=True,
            )

            # Disable chat controls while the automated analysis is running
            # to prevent the user from sending overlapping queries.
            self.btn_send.setEnabled(False)
            self.msg_input.setEnabled(False)

            # Build a proactive system-generated prompt for legal analysis.
            # This instructs the AI to examine the contract for inconsistencies,
            # legal risks, and ambiguous clauses.
            proactive_query = (
                "لطفا قرارداد موجود را به صورت کامل بررسی کرده و مغایرت‌های آن با قوانین، "
                "ریسک‌های حقوقی احتمالی و ابهامات مهم را به صورت ساختاریافته تحلیل کن."
            )

            # Start the AI analysis in a separate worker thread so the UI
            # remains responsive during the potentially long-running task.
            self.chat_thread = AIChatThread(self.rag_pipeline, proactive_query)

            # Connect thread signals to their corresponding UI handlers.
            self.chat_thread.finished.connect(self.on_chat_finished)
            self.chat_thread.error.connect(self.on_chat_error)

            # Start the background AI analysis.
            self.chat_thread.start()

        except Exception as e:
            # If the vector database was created but failed to reload,
            # notify the user via both the chat panel and a warning dialog.
            error_text = f"دیتابیس ساخته شد اما در بارگذاری خطا داد: {str(e)}"

            # Display the error message in the chat interface.
            self.append_chat_message("سیستم", error_text, "#f38ba8", is_system=True)

            # Show a warning dialog describing the issue.
            self.show_custom_msg("هشدار", error_text, "warning")

    def on_process_error(self, error_msg):
        """
        Handle a document processing failure.

        Args:
            error_msg (str): Error message emitted by the worker thread.
        """
        # Re-enable the processing button and restore its original text.
        self.btn_process.setEnabled(True)
        self.btn_process.setText("⚙️ پردازش اسناد و ساخت دیتابیس")

        # Show the error in the chat panel and as a modal dialog.
        self.append_chat_message("سیستم", error_msg, "#f38ba8", is_system=True)
        self.show_custom_msg("خطا", error_msg, "error")

    def send_message(self):
        """
        Send the current user message to the AI assistant asynchronously.

        The method validates input, appends the user's message to the chat,
        disables the input controls temporarily, and starts the chat thread.
        """
        # Read and sanitize the current input text.
        query = self.msg_input.text().strip()

        # Do nothing if the input is empty.
        if not query:
            return

        # Add the user's message to the chat history.
        self.append_chat_message("شما", query, "#89b4fa")

        # Clear the input field after sending.
        self.msg_input.clear()

        # Disable input controls while awaiting the response.
        self.btn_send.setEnabled(False)
        self.btn_send.setText("در حال فکر...")
        self.msg_input.setEnabled(False)

        # Create and start the background chat worker.
        self.chat_thread = AIChatThread(self.rag_pipeline, query)
        self.chat_thread.finished.connect(self.on_chat_finished)
        self.chat_thread.error.connect(self.on_chat_error)
        self.chat_thread.start()

    def on_chat_finished(self, response):
        """
        Handle a successful AI response.

        Args:
            response (str): Response text returned by the AI thread.
        """
        # Restore chat input controls.
        self.btn_send.setEnabled(True)
        self.btn_send.setText("ارسال 🚀")
        self.msg_input.setEnabled(True)
        self.msg_input.setFocus()

        # Display the assistant response in the chat panel.
        self.append_chat_message("دستیار حقوقی", response, "#a6e3a1")

    def on_chat_error(self, error_msg):
        """
        Handle an AI communication or inference error.

        Args:
            error_msg (str): Error message returned by the AI thread.
        """
        # Restore chat input controls after failure.
        self.btn_send.setEnabled(True)
        self.btn_send.setText("ارسال 🚀")
        self.msg_input.setEnabled(True)

        # Show the error in both the chat panel and a modal dialog.
        self.append_chat_message("سیستم", error_msg, "#f38ba8", is_system=True)
        self.show_custom_msg("خطا", error_msg, "error")
    
    def show_custom_msg(self, title, text, icon_type="info"):
        """
        Display a localized and styled message dialog.

        This helper method creates a customized QMessageBox that replaces the
        default static message box functions (such as QMessageBox.information).
        It allows localization of button labels and consistent styling across
        the application UI.

        Args:
            title (str): Title of the message dialog window.
            text (str): Main message text displayed to the user.
            icon_type (str): Type of icon to display. Supported values:
                             "info", "success", "warning", "error".

        Returns:
            int: The result code returned by QMessageBox.exec().
        """

        # Create a message box instance with the main window as its parent.
        msg_box = QMessageBox(self)

        # Set dialog title and message text.
        msg_box.setWindowTitle(title)
        msg_box.setText(text)

        # Select the appropriate icon based on the message type.
        if icon_type == "success":
            # Information icon is commonly used to represent successful actions.
            msg_box.setIcon(QMessageBox.Icon.Information)

        elif icon_type == "warning":
            # Warning icon for recoverable or attention-required situations.
            msg_box.setIcon(QMessageBox.Icon.Warning)

        elif icon_type == "error":
            # Critical icon for serious errors.
            msg_box.setIcon(QMessageBox.Icon.Critical)

        else:
            # Default informational icon.
            msg_box.setIcon(QMessageBox.Icon.Information)

        # Add a localized confirmation button instead of the default "OK".
        msg_box.addButton("باشه", QMessageBox.ButtonRole.AcceptRole)

        # Apply custom styling so the button visually matches the application's theme.
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }

            QLabel {
                color: #cdd6f4;
            }

            QPushButton {
                background-color: #89b4fa;
                color: #11111b;
                border-radius: 4px;
                padding: 5px 20px;
                font-weight: bold;
                min-width: 70px;
            }

            QPushButton:hover {
                background-color: #b4befe;
            }
        """)

        # Execute the dialog modally and return the result.
        return msg_box.exec()
    
    def append_chat_message(self, sender, message, color="white", is_system=False):
        import re
        import markdown
        
        # ۱. اصلاح ساختار لیست‌ها
        message = re.sub(r'^(\s*)([0-9]+)\.', r'\1\2-', message, flags=re.MULTILINE)
        message = re.sub(r'^(\s*[0-9]+-.*)$', r'\n\n\1\n\n', message, flags=re.MULTILINE)
        message = re.sub(r'\n{3,}', '\n\n', message)

        # ۲. تبدیل اعداد انگلیسی به فارسی
        english_to_persian = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')
        message = str(message).translate(english_to_persian)

        # ۳. رفع مشکل به‌هم‌ریختگی تاریخ‌ها و مبالغ در PyQt
        # استفاده از تگ HTML و کاراکترهای یونیکد برای اجبار به رندر چپ‌به‌راست اعداد
        def fix_bidi_numbers(match):
            # کاراکترهای LRE و PDF برای اصلاح جهت‌دهی
            return f'<span dir="ltr" style="unicode-bidi: embed;">&#x202A;{match.group(0)}&#x202C;</span>'
            
        pattern = r'[۰-۹]+(?:[/\-.,:][۰-۹]+)*'
        message = re.sub(pattern, fix_bidi_numbers, message)

        # ۴. تبدیل به HTML
        html_content = markdown.markdown(message, extensions=['tables'])

        # ۵. اعمال استایل
        html_content = html_content.replace('<p>', '<p dir="rtl" align="right" style="margin-bottom: 10px; text-align: justify; line-height: 1.6;">')
        for i in range(1, 7):
            html_content = html_content.replace(f'<h{i}>', f'<h{i} dir="rtl" align="right" style="margin-top: 10px; margin-bottom: 5px;">')

        html_content = html_content.replace('<table>', '<table dir="rtl" align="right" width="100%" border="1" style="border-collapse: collapse; margin-top: 10px; margin-bottom: 10px;">')
        html_content = html_content.replace('<th>', '<th style="padding: 8px; background-color: #313244; text-align: right; border: 1px solid #45475a;">')
        html_content = html_content.replace('<td>', '<td style="padding: 8px; text-align: right; border: 1px solid #45475a;">')

        html = f"""
        <div dir="rtl" align="right" style="margin-bottom: 15px;">
            <span style="color: {color}; font-weight: bold; font-size: 14px;">{sender}:</span>
            <div dir="rtl" align="right" style="margin-top: 5px; font-size: 14px;">
                {html_content}
            </div>
        </div>
        """
        
        self.chat_display.append(html)
        
        scroll_bar = self.chat_display.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())