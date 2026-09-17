from agi.code import CyphaCode

def test_parse_python():
    code = "def foo(): pass\nclass Bar: pass"
    funcs, classes = CyphaCode().parse(code)
    assert "foo" in funcs
    assert "Bar" in classes
