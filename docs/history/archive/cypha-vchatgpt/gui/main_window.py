from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter
from gui.chat import ChatWidget
from gui.monitor import MonitorWidget
from gui.training import TrainingDashboard
from gui.settings import SettingsPanel

class CyphaMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cypha AGI")
        self._main = QWidget()
        self.setCentralWidget(self._main)
        layout = QHBoxLayout(self._main)
        splitter = QSplitter()
        self.chat = ChatWidget()
        self.monitor = MonitorWidget()
        self.training = TrainingDashboard()
        self.settings = SettingsPanel()
        splitter.addWidget(self.chat)
        splitter.addWidget(self.monitor)
        splitter.addWidget(self.training)
        splitter.addWidget(self.settings)
        layout.addWidget(splitter)

def run_main():
    import sys
    app = QApplication(sys.argv)
    win = CyphaMainWindow()
    win.show()
    sys.exit(app.exec())
