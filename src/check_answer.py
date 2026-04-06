import re
from collections import Counter

def extract_equation(raw_text: str) -> str:
    text_before_eq = raw_text.split('=')[0]

    matches = re.findall(r'[\d\s\+\-\*\/\(\)]+', text_before_eq)

    if not matches:
        return ""

    equation = max(matches, key=len).strip()
    return equation

def check_answer(raw_text: str, target: int, available_numbers: list[int]) -> bool:
    equation = extract_equation(raw_text)

    if not equation or not any(op in equation for op in '+-*/'):
        return False

    used_numbers_str = re.findall(r'\d+', equation)
    used_numbers = [int(n) for n in used_numbers_str]

    available_counts = Counter(available_numbers)
    used_counts = Counter(used_numbers)

    for num, count in used_counts.items():
        if count > available_counts.get(num, 0):
            return False

    try:
        result = eval(equation, {"__builtins__": None}, {})
        return abs(result - target) < 1e-9

    except Exception:
        return False
