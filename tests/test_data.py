from ctrsink.data import parse_behaviors


def test_parse_python_list():
    assert parse_behaviors("['Toy Story', 'Heat']") == ["Toy Story", "Heat"]


def test_parse_comma_string():
    assert parse_behaviors("Toy Story, Heat") == ["Toy Story", "Heat"]


def test_parse_sep_string():
    assert parse_behaviors("Toy Story|SEP|Heat") == ["Toy Story", "Heat"]
