from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class TrainingDashboard(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.lcurves = QLabel("Training Loss Curves")
        self.status = QLabel("Status: Idle")
        layout.addWidget(self.lcurves)
        layout.addWidget(self.status)
    def update_status(self, msg):
        self.status.setText(f"Status: {msg}")
