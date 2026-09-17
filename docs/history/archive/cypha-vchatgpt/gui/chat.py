from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton

class ChatWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.input_line = QLineEdit()
        self.send = QPushButton("Send")
        self.layout.addWidget(self.text_area)
        self.layout.addWidget(self.input_line)
        self.layout.addWidget(self.send)
        self.send.clicked.connect(self.send_text)
    def send_text(self):
        txt = self.input_line.text()
        self.input_line.clear()
        self.text_area.append(f"<b>You:</b> {txt}")
        # Place cypha backend here for response, mock for now:
        self.text_area.append(f"<i>Cypha:</i> {txt[::-1]}")
