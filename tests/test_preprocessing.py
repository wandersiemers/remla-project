import pandas as pd

from src.data import preprocessing


def test_text_prepare():
    examples = [
        "SQL Server - any equivalent of Excel's CHOOSE function?",
        "How to free c++ memory vector<int> * arr?",
    ]

    answers = [
        "sql server equivalent excels choose function",
        "free c++ memory vectorint arr",
    ]

    for ex, ans in zip(examples, answers):
        if preprocessing.text_prepare(ex) != ans:
            return f"Wrong answer for the case: '{ex}'"

    return "Basic tests are passed."
