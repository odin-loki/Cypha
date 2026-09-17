from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar

class MonitorWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.eventq_lb = QLabel("Event Queue: 0")
        self.mem_bar = QProgressBar()
        self.cpu_bar = QProgressBar()
        layout.addWidget(self.eventq_lb)
        layout.addWidget(QLabel("Memory Usage"))
        layout.addWidget(self.mem_bar)
        layout.addWidget(QLabel("CPU Usage"))
        layout.addWidget(self.cpu_bar)
    def update_metrics(self, eq, mem, cpu):
        self.eventq_lb.setText(f"Event Queue: {eq}")
        self.mem_bar.setValue(int(mem*100))
        self.cpu_bar.setValue(int(cpu*100))
