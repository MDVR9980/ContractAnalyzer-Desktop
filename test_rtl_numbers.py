import sys
import re
import markdown
from PyQt6.QtWidgets import QApplication, QTextBrowser, QVBoxLayout, QWidget

class TestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("تست قطعی نمایش اعداد و تاریخ‌ها در QTextBrowser")
        self.resize(600, 400)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
        
        layout = QVBoxLayout(self)
        self.chat_display = QTextBrowser()
        layout.addWidget(self.chat_display)
        
        # نمونه متنی که شامل انواع چالش‌های اعداد در محیط RTL است
        sample_message = """
**گزارش پردازش متن:**
سلام! این یک تست برای نمایش صحیح مقادیر است.
- تاریخ امروز 1405/02/22 می‌باشد.
- مبلغ نهایی قرارداد 1,250,000.50 تومان محاسبه شد.
- شماره تماس پشتیبانی: 0912-345-6789
- نسخه سیستم: 3.1.4
این اعداد باید بدون به هم ریختگی و کاملا راست‌چین نمایش داده شوند.
        """
        
        self.append_chat_message("دستیار هوشمند", sample_message, color="#a6e3a1")

    # تابع مشابه با کدی که بالا به شما داده شد
    def append_chat_message(self, sender, message, color="white"):
        message = re.sub(r'^(\s*)([0-9]+)\.', r'\1\2-', message, flags=re.MULTILINE)
        message = re.sub(r'^(\s*[0-9]+-.*)$', r'\n\n\1\n\n', message, flags=re.MULTILINE)
        message = re.sub(r'\n{3,}', '\n\n', message)

        english_to_persian = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')
        message = str(message).translate(english_to_persian)

        pattern = r'[۰-۹]+(?:[/\-.,:][۰-۹]+)*'
        def wrap_number_ltr(match):
            return f'<span dir="ltr" style="unicode-bidi: embed;">\u202A{match.group(0)}\u202C</span>'
            
        message = re.sub(pattern, wrap_number_ltr, message)
        html_content = markdown.markdown(message, extensions=['tables'])
        html_content = html_content.replace('<p>', '<p dir="rtl" align="right" style="margin-bottom: 10px; text-align: justify; line-height: 1.6;">')

        html = f"""
        <div dir="rtl" align="right" style="margin-bottom: 15px;">
            <span style="color: {color}; font-weight: bold; font-size: 14px;">{sender}:</span>
            <div dir="rtl" align="right" style="margin-top: 5px; font-size: 14px;">
                {html_content}
            </div>
        </div>
        """
        self.chat_display.append(html)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())
