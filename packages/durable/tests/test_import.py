"""Smoke test: the package imports."""


def test_imports() -> None:
    import pi_durable

    assert pi_durable.__doc__
