import collections


def fullname(obj):
    _class = obj.__class__
    module = _class.__module__
    if module == "builtins":
        return _class.__name__
    return module + '.' + _class.__name__


class _CustomType:

    def __init__(self, *allowed_types):
        self.allowed_types = allowed_types

    def __call__(self, obj):
        return self.__typecheck__(obj)

    def __instancecheck__(self, obj):
        return self.__typecheck__(obj)

    def __typecheck__(self, obj, dtype_string=False):
        if dtype_string:
            return obj in self.allowed_types
        return type(obj).__qualname__ in self.allowed_types


def create(typename, *allowed_types, **custom_functions):
    def __typecheck__(self, obj, dtype_string=False):
        if dtype_string:
            return obj in self.allowed_types
        return type(obj).__qualname__ in self.allowed_types

    def __repr__(self):
        return str(list(self.allowed_types))

    attrs = {"__typecheck__": __typecheck__,
             "__repr__": __repr__, "allowed_types": allowed_types}
    attrs.update(**custom_functions)
    custom_type = type(typename, (_CustomType,), attrs) 
    return custom_type


NoneType = create("NoneType")("NoneType")
Callable = create("Callable")("function", "type", "generator")
Text = create("Text")("str", "bytes", "collections.UserString")
Number = create("Number")("int", "float", "complex")
Iterable = create("Iterable")(
    "str", "list", "dict", "tuple", "dict_keys", "dict_values",
    "dict_items", "str_iterator", "list_iterator", "tuple_iterator", "dict_keyiterator",
    "dict_valueiterator", "dict_itemiterator", "range", "enumerate", "filter", "list_reverseiterator",
    "numpy.ndarray"
)
