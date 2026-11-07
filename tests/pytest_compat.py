"""Замена pytest.raises, чтобы тесты работали и без pytest."""


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return
    raise AssertionError(f"ожидалось исключение {exc.__name__}")
