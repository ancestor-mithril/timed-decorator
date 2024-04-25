def recursive_fibonacci(n: int) -> int:
    assert n > 0
    if n > 3:
        return recursive_fibonacci(n - 1) + recursive_fibonacci(n - 2)
    if n == 1:
        return 0
    return 1


def fibonacci(n: int) -> int:
    assert n > 0
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
