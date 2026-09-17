import re

class InputValidator:
    def __init__(self):
        self.bad = re.compile(r"(rm\s+-rf|del\s+/|shutdown|:(){:|: &};:)")
    def validate(self, inp: str):
        if self.bad.search(inp):
            return False
        return True
    def sanitize(self, inp: str):
        return re.sub(r'[^\w\d\.\,\!\? ]', '', inp)
