import os
import subprocess

class CyphaSystem:
    def run(self, cmd: str):
        allowed = ["ls", "pwd", "head", "tail"]
        head = cmd.split()[0]
        if head not in allowed:
            return "Command not allowed"
        try:
            return subprocess.check_output(cmd, shell=True, timeout=2).decode()
        except Exception as e:
            return f"Error: {str(e)}"
    def env(self):
        return dict(os.environ)
    def ps(self):
        return os.popen('ps aux').read()

