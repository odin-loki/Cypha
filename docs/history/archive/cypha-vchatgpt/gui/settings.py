from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QCheckBox

class SettingsPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Settings"))
        self.gpu = QCheckBox("Use GPU")
        self.gpu.setChecked(True)
        self.sandbox = QCheckBox("Enable sandbox")
        self.sandbox.setChecked(True)
        layout.addWidget(self.gpu)
        layout.addWidget(self.sandbox)
    def get_settings(self):
        return {"gpu": self.gpu.isChecked(), "sandbox": self.sandbox.isChecked()}
