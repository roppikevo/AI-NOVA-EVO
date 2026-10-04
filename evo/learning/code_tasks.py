"""
Procedural Python task bank for NOVA's code school.

Every task = function signature + short Slovak/English description +
unit tests + reference solution. Families are parameterised (different
names/constants), so there are hundreds of distinct tasks. Difficulty:
    1 one-liners (arithmetic, strings)
    2 loops / conditions over lists and strings
    3 small algorithms (gcd, primes, fibonacci, palindromes, counting)
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field


@dataclass
class CodeTask:
    name: str
    signature: str          # e.g. "add_3(x)"
    description: str        # docstring text
    tests: list[str]        # assert lines
    solution: str           # body (indented with 4 spaces)
    level: int
    lang: str = "py"
    family: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return hashlib.sha256(f"{self.signature}|{self.description}".encode()).hexdigest()[:16]

    @property
    def pool(self) -> str:
        return "exam" if int(self.key[:6], 16) % 10 < 3 else "practice"

    @property
    def prompt(self) -> str:
        return f'def {self.signature}:\n    """{self.description}"""\n'

    def program(self, body: str) -> str:
        return self.prompt + body.rstrip() + "\n\n\n" + "\n".join(self.tests) + "\nprint('ALL_TESTS_PASSED')\n"


def _t(name, sig, desc, tests, sol, level, family):
    return CodeTask(name, sig, desc, tests, sol, level, family=family)


def _level1(rng: random.Random) -> list[CodeTask]:
    out = []
    for k in range(1, 41):  # many constants: NOVA must read the number, not memorise one
        out.append(_t(f"add_{k}", f"add_{k}(x)", f"Vráť x zväčšené o {k}.",
                      [f"assert add_{k}(0) == {k}", f"assert add_{k}(5) == {5 + k}", f"assert add_{k}(-2) == {k - 2}"],
                      f"    return x + {k}", 1, "add_const"))
        out.append(_t(f"times_{k}", f"times_{k}(x)", f"Return x multiplied by {k}.",
                      [f"assert times_{k}(0) == 0", f"assert times_{k}(3) == {3 * k}", f"assert times_{k}(-1) == {-k}"],
                      f"    return x * {k}", 1, "mul_const"))
    for k in range(1, 31):
        out.append(_t(f"minus_{k}", f"minus_{k}(x)", f"Vráť x zmenšené o {k}.",
                      [f"assert minus_{k}(0) == {-k}", f"assert minus_{k}(50) == {50 - k}"],
                      f"    return x - {k}", 1, "sub_const"))
    for k in range(2, 21):
        out.append(_t(f"mod_{k}", f"mod_{k}(x)", f"Return the remainder of x divided by {k}.",
                      [f"assert mod_{k}({k}) == 0", f"assert mod_{k}({k + 1}) == 1", f"assert mod_{k}({3 * k - 1}) == {k - 1}"],
                      f"    return x % {k}", 1, "mod_const"))
        out.append(_t(f"greater_than_{k}", f"greater_than_{k}(x)", f"Vráť True, ak je x väčšie ako {k}.",
                      [f"assert greater_than_{k}({k + 1}) is True", f"assert greater_than_{k}({k}) is False"],
                      f"    return x > {k}", 1, "cmp_const"))
    for k in range(2, 11):
        out.append(_t(f"repeat_{k}", f"repeat_{k}(s)", f"Return the string repeated {k} times.",
                      [f"assert repeat_{k}('ab') == {'ab' * k!r}", f"assert repeat_{k}('') == ''"],
                      f"    return s * {k}", 1, "repeat_const"))
    out += [
        _t("add", "add(a, b)", "Vráť súčet a a b.", ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"], "    return a + b", 1, "binop"),
        _t("subtract", "subtract(a, b)", "Return a minus b.", ["assert subtract(5, 3) == 2", "assert subtract(0, 4) == -4"], "    return a - b", 1, "binop"),
        _t("multiply", "multiply(a, b)", "Vráť súčin a a b.", ["assert multiply(3, 4) == 12", "assert multiply(-2, 5) == -10"], "    return a * b", 1, "binop"),
        _t("is_even", "is_even(n)", "Vráť True, ak je n párne.", ["assert is_even(4) is True", "assert is_even(7) is False", "assert is_even(0) is True"], "    return n % 2 == 0", 1, "predicate"),
        _t("is_odd", "is_odd(n)", "Return True if n is odd.", ["assert is_odd(3) is True", "assert is_odd(8) is False"], "    return n % 2 == 1", 1, "predicate"),
        _t("square", "square(x)", "Vráť druhú mocninu x.", ["assert square(3) == 9", "assert square(-4) == 16"], "    return x * x", 1, "unary"),
        _t("negate", "negate(x)", "Return -x.", ["assert negate(5) == -5", "assert negate(-2) == 2"], "    return -x", 1, "unary"),
        _t("to_upper", "to_upper(s)", "Vráť reťazec s veľkými písmenami.", ["assert to_upper('abc') == 'ABC'", "assert to_upper('Ahoj') == 'AHOJ'"], "    return s.upper()", 1, "string"),
        _t("to_lower", "to_lower(s)", "Return the string in lower case.", ["assert to_lower('ABC') == 'abc'"], "    return s.lower()", 1, "string"),
        _t("first_char", "first_char(s)", "Vráť prvý znak reťazca s.", ["assert first_char('ahoj') == 'a'", "assert first_char('Z') == 'Z'"], "    return s[0]", 1, "string"),
        _t("last_item", "last_item(xs)", "Return the last item of the list.", ["assert last_item([1, 2, 3]) == 3", "assert last_item(['a']) == 'a'"], "    return xs[-1]", 1, "list"),
        _t("length", "length(xs)", "Vráť počet prvkov zoznamu.", ["assert length([1, 2, 3]) == 3", "assert length([]) == 0"], "    return len(xs)", 1, "list"),
        _t("reverse_string", "reverse_string(s)", "Vráť obrátený reťazec.", ["assert reverse_string('abc') == 'cba'", "assert reverse_string('') == ''"], "    return s[::-1]", 1, "string"),
        _t("celsius_to_f", "celsius_to_f(c)", "Prepočítaj stupne Celzia na Fahrenheita.", ["assert celsius_to_f(0) == 32", "assert celsius_to_f(100) == 212"], "    return c * 9 / 5 + 32", 1, "formula"),
        _t("average2", "average2(a, b)", "Return the average of a and b.", ["assert average2(2, 4) == 3", "assert average2(1, 2) == 1.5"], "    return (a + b) / 2", 1, "formula"),
    ]
    return out


def _level2(rng: random.Random) -> list[CodeTask]:
    out = [
        _t("sum_list", "sum_list(xs)", "Vráť súčet všetkých čísel v zozname.", ["assert sum_list([1, 2, 3]) == 6", "assert sum_list([]) == 0"],
           "    total = 0\n    for x in xs:\n        total += x\n    return total", 2, "loop"),
        _t("max_list", "max_list(xs)", "Return the largest number in the list.", ["assert max_list([3, 9, 2]) == 9", "assert max_list([-5, -1]) == -1"],
           "    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best", 2, "loop"),
        _t("min_list", "min_list(xs)", "Vráť najmenšie číslo v zozname.", ["assert min_list([3, 9, 2]) == 2", "assert min_list([7]) == 7"],
           "    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best", 2, "loop"),
        _t("count_vowels", "count_vowels(s)", "Spočítaj samohlásky a, e, i, o, u v reťazci.", ["assert count_vowels('ahoj') == 2", "assert count_vowels('xyz') == 0"],
           "    return sum(1 for ch in s.lower() if ch in 'aeiou')", 2, "string"),
        _t("only_even", "only_even(xs)", "Return a list with only the even numbers.", ["assert only_even([1, 2, 3, 4]) == [2, 4]", "assert only_even([1, 3]) == []"],
           "    return [x for x in xs if x % 2 == 0]", 2, "filter"),
        _t("squares", "squares(xs)", "Vráť zoznam druhých mocnín.", ["assert squares([1, 2, 3]) == [1, 4, 9]", "assert squares([]) == []"],
           "    return [x * x for x in xs]", 2, "map"),
        _t("count_words", "count_words(s)", "Return the number of words in the sentence.", ["assert count_words('ahoj ako sa máš') == 4", "assert count_words('') == 0"],
           "    return len(s.split())", 2, "string"),
        _t("average", "average(xs)", "Vráť priemer čísel v zozname.", ["assert average([2, 4, 6]) == 4", "assert average([1, 2]) == 1.5"],
           "    return sum(xs) / len(xs)", 2, "formula"),
        _t("abs_value", "abs_value(x)", "Vráť absolútnu hodnotu x bez funkcie abs.", ["assert abs_value(-3) == 3", "assert abs_value(4) == 4"],
           "    if x < 0:\n        return -x\n    return x", 2, "branch"),
        _t("sign", "sign(x)", "Return 1 for positive, -1 for negative and 0 for zero.", ["assert sign(5) == 1", "assert sign(-2) == -1", "assert sign(0) == 0"],
           "    if x > 0:\n        return 1\n    if x < 0:\n        return -1\n    return 0", 2, "branch"),
        _t("join_words", "join_words(words)", "Spoj slová do jednej vety oddelenej medzerami.", ["assert join_words(['ahoj', 'svet']) == 'ahoj svet'"],
           "    return ' '.join(words)", 2, "string"),
    ]
    for ch in "aeosbkmntz":
        out.append(_t(f"count_{ch}", f"count_{ch}(s)", f"Spočítaj, koľkokrát je v reťazci písmeno '{ch}'.",
                      [f"assert count_{ch}('{ch}{ch}x{ch}') == 3", f"assert count_{ch}('123') == 0"],
                      f"    return s.count('{ch}')", 2, "count_char"))
    for k in range(1, 13):
        out.append(_t(f"first_{k}", f"first_{k}(xs)", f"Vráť prvých {k} prvkov zoznamu.",
                      [f"assert first_{k}(list(range(20))) == {list(range(k))!r}", f"assert first_{k}([]) == []"],
                      f"    return xs[:{k}]", 2, "slice"))
        out.append(_t(f"above_{k}", f"above_{k}(xs)", f"Return the numbers greater than {k}.",
                      [f"assert above_{k}([{k - 1}, {k}, {k + 1}, {k + 5}]) == [{k + 1}, {k + 5}]", f"assert above_{k}([]) == []"],
                      f"    return [x for x in xs if x > {k}]", 2, "filter"))
        out.append(_t(f"add_{k}_to_all", f"add_{k}_to_all(xs)", f"Pripočítaj {k} ku každému číslu v zozname.",
                      [f"assert add_{k}_to_all([0, 1, 10]) == [{k}, {k + 1}, {k + 10}]", f"assert add_{k}_to_all([]) == []"],
                      f"    return [x + {k} for x in xs]", 2, "map"))
    for k in range(2, 16):
        out.append(_t(f"divisible_by_{k}", f"divisible_by_{k}(xs)", f"Return the numbers divisible by {k}.",
                      [f"assert divisible_by_{k}([{k}, {k + 1}, {2 * k}]) == [{k}, {2 * k}]", f"assert divisible_by_{k}([]) == []"],
                      f"    return [x for x in xs if x % {k} == 0]", 2, "filter"))
    return out


def _level3(rng: random.Random) -> list[CodeTask]:
    return [
        _t("factorial", "factorial(n)", "Vráť faktoriál čísla n.", ["assert factorial(0) == 1", "assert factorial(5) == 120"],
           "    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result", 3, "algorithm"),
        _t("fibonacci", "fibonacci(n)", "Return the n-th Fibonacci number (fibonacci(0) == 0).", ["assert fibonacci(0) == 0", "assert fibonacci(1) == 1", "assert fibonacci(10) == 55"],
           "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a", 3, "algorithm"),
        _t("gcd", "gcd(a, b)", "Vráť najväčší spoločný deliteľ a a b.", ["assert gcd(12, 18) == 6", "assert gcd(7, 5) == 1"],
           "    while b:\n        a, b = b, a % b\n    return a", 3, "algorithm"),
        _t("is_prime", "is_prime(n)", "Return True if n is a prime number.", ["assert is_prime(2) is True", "assert is_prime(15) is False", "assert is_prime(13) is True", "assert is_prime(1) is False"],
           "    if n < 2:\n        return False\n    for d in range(2, int(n ** 0.5) + 1):\n        if n % d == 0:\n            return False\n    return True", 3, "algorithm"),
        _t("is_palindrome", "is_palindrome(s)", "Vráť True, ak sa reťazec číta rovnako odpredu aj odzadu.", ["assert is_palindrome('kajak') is True", "assert is_palindrome('ahoj') is False"],
           "    return s == s[::-1]", 3, "string"),
        _t("char_counts", "char_counts(s)", "Return a dict mapping each character to its count.", ["assert char_counts('aab') == {'a': 2, 'b': 1}", "assert char_counts('') == {}"],
           "    counts = {}\n    for ch in s:\n        counts[ch] = counts.get(ch, 0) + 1\n    return counts", 3, "dict"),
        _t("unique", "unique(xs)", "Vráť zoznam bez duplicít v pôvodnom poradí.", ["assert unique([1, 2, 1, 3, 2]) == [1, 2, 3]"],
           "    seen = []\n    for x in xs:\n        if x not in seen:\n            seen.append(x)\n    return seen", 3, "list"),
        _t("power", "power(base, exp)", "Return base to the power exp using a loop (exp >= 0).", ["assert power(2, 10) == 1024", "assert power(5, 0) == 1"],
           "    result = 1\n    for _ in range(exp):\n        result *= base\n    return result", 3, "algorithm"),
        _t("digit_sum", "digit_sum(n)", "Vráť súčet cifier nezáporného čísla n.", ["assert digit_sum(123) == 6", "assert digit_sum(0) == 0"],
           "    total = 0\n    while n > 0:\n        total += n % 10\n        n //= 10\n    return total", 3, "algorithm"),
        _t("fizzbuzz", "fizzbuzz(n)", "Return 'Fizz' for multiples of 3, 'Buzz' for 5, 'FizzBuzz' for both, else str(n).",
           ["assert fizzbuzz(3) == 'Fizz'", "assert fizzbuzz(10) == 'Buzz'", "assert fizzbuzz(15) == 'FizzBuzz'", "assert fizzbuzz(7) == '7'"],
           "    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)", 3, "branch"),
    ]


def task_bank(seed: int = 0) -> list[CodeTask]:
    rng = random.Random(seed)
    return _level1(rng) + _level2(rng) + _level3(rng)
