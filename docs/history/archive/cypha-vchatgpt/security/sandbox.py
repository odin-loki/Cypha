import subprocess

class CyphaSandbox:
    def __init__(self, whitelist=None):
        self.whitelist = whitelist or ["ls", "pwd", "head", "tail"]
    def run(self, cmd: str):
        if cmd.split()[0] not in self.whitelist:
            return "Blocked by sandbox."
        try:
            return subprocess.check_output(cmd, shell=True, timeout=2).decode()
        except Exception as e:
            return f"Error: {e}"
