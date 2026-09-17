class CyphaMonitor:
    def __init__(self):
        self.metrics = {}
    def report(self, k, v):
        self.metrics[k] = v
    def get_all(self):
        return dict(self.metrics)
    def alert(self, msg):
        print("ALERT:", msg)
