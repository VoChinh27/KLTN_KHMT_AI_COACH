"""Reuse exact frozen function code with a private, explicit globals dictionary.

No source rewrite, global monkeypatch, changed loop, or legacy-module mutation.
All module-local calls are rebound together, including checkpoint validators.
The frozen source files and this binding are included in manifest identities.
"""
from types import FunctionType, SimpleNamespace


def bind(module, overrides):
    namespace = dict(vars(module))
    namespace.update(overrides)
    for name, value in vars(module).items():
        if isinstance(value, FunctionType) and value.__module__ == module.__name__:
            clone = FunctionType(value.__code__, namespace, value.__name__, value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = value.__kwdefaults__
            clone.__annotations__ = value.__annotations__.copy()
            clone.__doc__ = value.__doc__
            namespace[name] = clone
    return SimpleNamespace(**namespace), namespace
