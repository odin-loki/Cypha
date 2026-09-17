import re

class CyphaCode:
    def parse(self, code: str, lang='python'):
        if lang == 'python':
            return re.findall(r'def (\w+)\(', code), re.findall(r'class (\w+)\(', code)
        if lang == 'js':
            return re.findall(r'function (\w+)\(', code), []
        return [], []
    def generate(self, desc: str):
        if 'sort' in desc:
            return "def sort(xs):\n    return sorted(xs)\n"
        return f"# {desc}\npass\n"
    def debug(self, code: str):
        if "SyntaxError" in code:
            return code.replace('SyntaxError', '# Fixed SyntaxError')
        return code
    def review(self, code: str):
        issues = []
        if 'eval(' in code:
            issues.append("Avoid using eval.")
        if len(code) > 1000:
            issues.append("Code is long, consider refactoring.")
        return issues