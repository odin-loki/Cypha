import os
import time

class CyphaAudit:
    def __init__(self, path="cypha_audit.log"):
        self.path = path
    def log(self, msg):
        with open(self.path, "a") as f:
            f.write(f"[{time.ctime()}] {msg}\n")
    def rotate(self):
        if os.path.exists(self.path) and os.path.getsize(self.path) > 5e6:
            os.rename(self.path, self.path + ".bak")
