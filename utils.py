from string import *
import shutil
import os
import sys
import stat
import base64
import re
import numpy as np
import threading
from threading import Thread as _thread
from six.moves import input as raw_input
import time
import timeit
import ctypes
import traceback
import pkg_resources
import functools
import inspect

chars = ascii_letters + digits + punctuation 
char2int = {c: i for i, c in enumerate(chars, 1)}
int2char = {i: c for i, c in enumerate(chars, 1)}

# Errors
class AuthenticationError(Exception): ...
class InvalidFileException(Exception): ...
class ThreadStillAliveError(Exception): ...

# Decorators
def pipe(filename):
    def wrapper(fn):
        def inner(*args, **kwargs):
            value = fn(*args, **kwargs)
            with open(filename) as f:
                f.write(str(value))
            return value
        return inner
    return wrapper

def catch(exc=Exception, verbose=False):
    def wrapper(fn):
        def inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except exc as e:
                if verbose:
                    print(e.__class__.__name__ + ": " + str(e))
        return inner
    return wrapper

def log(filename, exceptions=True):
    def wrapper(fn):
        def inner(*args, **kwargs):
            e = None
            try:
                value = fn(*args, **kwargs)
            except Exception as e:
                if not exceptions:
                    raise
            finally:
                if not os.path.exists(filename):
                    open(filename, "w").close()
                with open(filename, "a") as f:
                    f.write(str(value) + "\n" +
                            traceback.format_exc(e) if e else "")
                return value
        return inner
    return wrapper

def timed(fn):
    def inner(*args, **kwargs):
        start = timeit.default_timer()
        value = fn(*args, **kwargs)
        delta = timeit.default_timer() - start
        return value, delta
    return inner

def repeat(times):
    def wrapper(fn):
        def inner(*args, **kwargs):
            return [fn(*args, **kwargs) for i in range(times)]
        return inner
    return wrapper

def threaded(terminate_after=0):
    def wrapper(fn):
        def inner(*args, **kwargs):
            thread = Thread(target=fn, args=args, kwargs=kwargs, group=None,
                            name=next(_gen), terminate_after=terminate_after, _timer=bool(terminate_after)
                            )
            fn.thread = thread
            fn.get = thread._get
            return fn
        return inner
    return wrapper


class PythonParser:

    obj_dunders = [
        "__class__", "__dict__", "__module__", "__doc__", "__weakref__"
    ]
    builtin = [
        "__builtins__", "__cached__", "__doc__", "__file__",
        "__loader__", "__name__", "__package__", "__spec__"
    ]

    def lazy_init(self, init):
        arg_names, def_kwargs = get_params(init)

        @functools.wraps(init)
        def inner(self, *args, **kwargs):
            def_kwargs.update(kwargs)
            for name, value in zip(arg_names[1:], args):
                try:
                    setattr(self, name, value)
                except Exception as e:
                    pass
            for name, value in def_kwargs.items():
                try:
                    setattr(self, name, value)
                except Exception:
                    pass
            init(self, *args, **kwargs)
        return inner

    def _dir(self, obj=None):
        return [attr for attr in dir(obj) if attr not in self.builtin + self.obj_dunders]

    def classattrs(self, obj):
        return [attr for attr in obj.__dict__.keys() if attr not in self.builtin + self.obj_dunders]

    def isbuiltin(self, class_):
        return hasattr(class_, "__dict__")

    def get_attrs(self, obj, rlevel=0):
        if rlevel == 0:
            global attributes
            attributes = []
        attrs = self._dir(obj)
        for attr_name in attrs:
            attr = getattr(obj, attr_name)
            if attr.__class__.__name__ == "type":
                attributes.extend(
                    [f"{obj.__name__}.{attr_name}.{classattr}" for classattr in (
                        self.classattrs(attr))]
                )
            attributes.append(f"{obj.__name__}.{attr_name}")
        if rlevel == 0:
            return attributes

    def remove_whitespace(self, code):
        return "\n".join([line for line in code.split("\n") if len(line.replace(" ", ""))] + [""])

    def get_indent_level(self, line, tab_size):
        spaces = 0
        for char in line:
            if char.isspace():
                spaces += 1
            else:
                break
        return spaces // tab_size

    def find_class(self, code, class_, tab_size):
        indent_level = None
        start_line, end_line = 0, len(code.split("\n"))
        if f"class {class_}(" in code:
            declaration = f"class {class_}():"
        elif f"class {class_}:" in code:
            declaration = f"class {class_}:"
        for i, line in enumerate(code.split("\n")):
            if declaration in line:
                indent_level = self.get_indent_level(line, tab_size)
                start_line = i
            else:
                if indent_level is not None:
                    if self.get_indent_level(line, tab_size) == indent_level:
                        end_line = i
                        break
        return "\n".join(code.split("\n")[start_line:end_line])

    def find(self, code=None, py_file=None, fn=None, class_=None, tab_size=4):
        if code is None and py_file is not None:
            with open(py_file, "r") as f:
                code = f.read()
        code = re.sub(" *\(", "(", code)
        indent_level = None
        declaration = f"def {fn}("
        code = self.remove_whitespace(code)
        if class_ is not None:
            code = self.find_class(code, class_, tab_size)
        start_line, end_line = 0, len(code.split("\n"))
        for i, line in enumerate(code.split("\n")):
            if declaration in line:
                indent_level = self.get_indent_level(line, tab_size)
                start_line = i
            else:
                if indent_level is not None:
                    if self.get_indent_level(line, tab_size) == indent_level:
                        end_line = i
                        break
        return "\n".join(code.split("\n")[start_line:end_line])

    def get_method_info(self, code):
        print(code)
        return

    def get_code(self, file):
        def wrapper(fn):
            def inner(*args, **kwargs):
                full_name = fn.__qualname__.split(".")
                method = fn.__name__
                class_ = None
                if len(full_name) > 1:
                    class_ = full_name[full_name.index(method) - 1]
                code = self.find(py_file=file, fn=method, class_=class_)
                return fn(*args, **kwargs), code
            return inner
        return wrapper


pythonparser = PythonParser()

def replace(string, a, b):
    if type(a) == list:
        for char in a:
            string = string.replace(char, b)
        return string
    else:
        return string.replace(a, b)

def _assert(stmt, msg, exc=Exception):
    if not stmt:
        raise exc(msg)

def get_params(fn):
    fn_args = []
    fn_kwargs = {}
    params = inspect.signature(fn)
    for param in str(params)[1:-1].split(", "):
        param = param.split("=", 1)
        if len(param) > 1:
            fn_kwargs[param[0]] = param[1]
        else:
            fn_args.append(param[0])

    return fn_args, fn_kwargs

def levenshtein(token1, token2):
    m, n = len(token1), len(token2)
    matrix = np.zeros([m+1, n+1])
    for i in range(m):
        matrix[i][0] = i
    for i in range(n):
        matrix[0][i] = i
    a, b, c = 0, 0, 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            if token1[i-1] == token2[j-1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                a = matrix[i][j - 1]
                b = matrix[i - 1][j]
                c = matrix[i - 1][j - 1]
                if a <= b and a <= c:
                    matrix[i][j] = a + 1
                elif b <= a and b <= c:
                    matrix[i][j] = b + 1
                else:
                    matrix[i][j] = c + 1
    return matrix[m][n]

def most_likely(strings, target, threshold=5):
    scores = [levenshtein(string, target) for string in strings]
    return strings[scores.index(min(scores))] if min(scores) <= threshold else None

def cipher(string=None, key=None, layers=1, _rec_level=0, verbose=False):
    temp_key_ints = [char2int[char] for char in key]

    string = base64.b64encode(string.encode()).decode()

    key_ints = []
    for index, i in enumerate(temp_key_ints):
        if index % 2 == 0:
            key_ints.append(i ** 2)
        else:
            key_ints.append(i * 2)

    string_ints = []
    for index, integer in enumerate([char2int[char] for char in string], 1):
        shift = integer + index + sum(key_ints)
        if shift <= len(chars):
            string_ints.append(shift)
        else:
            if shift % len(chars) == 0:
                string_ints.append(len(chars))
            else:
                string_ints.append(shift % len(chars))

    enc = "".join([int2char[i] for i in string_ints])
    if verbose:
        print(f"Layer {_rec_level + 1} cipher: {enc}")
    if layers > 1:
        _rec_level += 1
        layers -= 1
        enc = cipher(enc, key, layers, _rec_level)
    return enc

def decipher(string=None, key=None, layers=1, _rec_level=0, verbose=False):
    temp_key_ints = [char2int[char] for char in key]

    key_ints = []
    for index, i in enumerate(temp_key_ints):
        if index % 2 == 0:
            key_ints.append(i ** 2)
        else:
            key_ints.append(i * 2)

    string_ints = []
    for index, integer in enumerate([char2int[char] for char in string], 1):
        shift = integer - index - sum(key_ints)
        if shift > 0:
            string_ints.append(shift)
        else:
            if shift % len(chars) == 0:
                string_ints.append(len(chars))
            else:
                string_ints.append(shift % len(chars))

    msg = "".join([int2char[i] for i in string_ints])
    if verbose:
        print(f"Layer {layers} decipher: {msg}")
    msg = base64.b64decode(msg.encode()).decode()
    if layers > 1:
        _rec_level += 1
        layers -= 1
        msg = decipher(msg, key, layers, _rec_level)
    return msg

def lock():
    os.system("Rundll32.exe user32.dll,LockWorkStation")

def get_dependencies(package_name, tree=True, rlevel=1, pre=False):
    try:
        package = pkg_resources.working_set.by_key[package_name]
    except KeyError:
        return None
    deps = [re.split("[^A-Za-z0-9\- ]", str(r))[0] for r in package.requires()]
    if pre:
        return deps
    if rlevel == 1:
        if tree:
            print(package_name + ":" if deps else package_name)
        global global_deps
        global_deps = []
    for dep in deps:
        output = ("  " * rlevel) + "- " + dep
        if tree:
            print(output + ":" if get_dependencies(dep,
                  tree=tree, pre=True) else output)
        global_deps.extend(get_dependencies(dep, tree=tree, rlevel=rlevel+1))

    if rlevel != 1:
        return deps
    return list(set(global_deps))


class FilePreserver:
    paused = False

    @pythonparser.lazy_init
    def __init__(self, path, writable=False):
        self.run()

    @threaded(terminate_after=None)
    def run(self):
        folder = os.path.dirname(self.path)
        file = os.path.basename(self.path)
        if os.path.splitext(file)[1] == "db":
            database = True
        copies = []
        if not self.writable:
            with open(self.path, "rb") as f:
                initial = f.read()
        while True:
            modified = False
            if len(copies) == 3:
                del copies[0]
            items = os.listdir(folder)
            try:
                with open(self.path, "rb") as f:
                    contents = f.read()
                if not self.writable:
                    if contents != initial:
                        modified = True
                        if self.paused:
                            initial = contents
                        continue
            except OSError:
                pass
            if file not in items or modified:
                if modified and not self.writable:
                    print(
                        f"\nFile '{self.path}' has been modified, restoring previous version...")
                    with open(self.path, "wb") as f:
                        f.write(initial)
                else:
                    print(
                        f"\nFile '{self.path}' has been deleted, restoring...")
                    with open(self.path, "wb") as f:
                        f.write(copies[-1])

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class ThreadReturnValue:

    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return f"<{self.__class__.__name__}: {repr(self.obj)}>"

    def get(self):
        return self.obj


class ThreadInProgress: ...
class Thread(_thread):

    @pythonparser.lazy_init
    def __init__(self, target=None, args=(),
                 kwargs={}, name=None, group=None,
                 daemon=None, terminate_after=None, _timer=False
                 ):
        #print(f"Initialising new thread with name '{name}' and target function '{target.__name__}'")
        super(Thread, self).__init__(
            target=target, args=args, kwargs=kwargs, name=name, group=group, daemon=daemon
        )
        self.return_value = None
        self.start()

    # Override run method
    def run(self):
        if self.terminate_after:
            @threaded(terminate_after=0)
            def _fn(thread_to_kill, delay):
                #print(f"Sleeping for {delay}")
                time.sleep(delay + 0.05)
                #print("Killing functional thread")
                thread_to_kill.stop()

            timer_thread = _fn(self, self.terminate_after)
            timer_thread.thread.stop()
        #print(f"Running function '{self.target.__name__}' in thread '{self.name}'")
        self.return_value = ThreadReturnValue(
            self.target(*self.args, **self.kwargs))

    def _get(self):
        if self.return_value:
            return self.return_value.get()
        return ThreadInProgress

    # Get ID of this thread
    def get_id(self):
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    # Kill this thread
    def stop(self, kill_all=False):
        thread_id = self.get_id()
        resp = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
                                                          ctypes.py_object(SystemExit))
        if resp > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            raise ThreadStillAliveError(
                "Thread cannot be stopped!"
            )
        if kill_all:
            os._exit(0)


def _input(msg=""):
    user_input = None
    try:
        user_input = input(msg)
    except KeyboardInterrupt:
        print()
    finally:
        return user_input

def _gen_n(n=1000000000):
    if n is None:
        i = 0
        while True:
            yield i
            i += 1
    else:
        for i in range(n):
            yield i

def replace_builtin_constructors(string):
    replace = {
        "''": "str", '""': "str", "b''": "bytes", 'b""': "bytes",
        "bstr": "bytes", "[]": "list", "\{\}": "dict", "()": "tuple"
    }
    for item in replace.items():
        string = string.replace(*item)
    return string

def fullname(obj):
    _class = obj.__class__
    module = _class.__module__
    if module == "builtins":
        return _class.__name__
    return module + '.' + _class.__name__

def typecheck(arg, param, _type, default_value=False):
    if type(_type) in [list, tuple, set]:
        if len(_type):
            allowed_types = [sub_t.__name__ for sub_t in _type]
            msg = f"Incorrect type for argument '{param}': expected one of {str(allowed_types)[1:-1]}, got type '{type(arg).__name__}' instead"
            _assert(fullname(arg) in allowed_types, msg, TypeError)
    elif hasattr(_type, "allowed_types"):
        msg = f"Incorrect type for argument '{param}': expected one of {str(_type.allowed_types)[1:-1]}, got type '{type(arg).__name__}' instead"
        _assert(fullname(arg) in _type.allowed_types, msg, TypeError)
    else:
        msg = f"Incorrect type for argument '{param}': expected type '{_type.__name__}', got type '{type(arg).__name__}' instead"
        _assert(fullname(arg) == _type.__name__, msg, TypeError)

def statictyped(fn):
    def inner(*args, **kwargs):
        default_params = None
        fn_kwargs = {fn.__code__.co_varnames[i]: args[i] for i in range(len(args))}
        if fn.__defaults__:
            default_params = {param: fn.__defaults__[i] for i, param in enumerate(
                fn.__code__.co_varnames[-len(fn.__defaults__):])}
            default_params.update(**kwargs)
            fn_kwargs.update(default_params)
        args = list(fn_kwargs.values())
        for i, (p, t) in enumerate(fn.__annotations__.items()):
            typecheck(args[i], p, t, default_value=True)
            #print(p, t)
        return fn(**fn_kwargs)
    return inner

def _typecheck(var, expected_type, is_param=False):
    definition = "variable" if not is_param else "parameter"
    actual_type = eval(f"type({var})")
    if hasattr(expected_type, "allowed_types"):
        msg = f"Incorrect type for {definition} '{var}': expected one of {expected_type}, got type '{actual_type.__name__}' instead"
        _assert(actual_type.__name__ in expected_type.allowed_types or fullname(
            eval(var)) in expected_type.allowed_types, msg, TypeError)
    else:
        msg = f"Incorrect type for {definition} '{var}': expected type '{expected_type.__name__}', got type '{actual_type.__name__}' instead"
        _assert(expected_type.__name__ in [
                actual_type.__name__, eval(var)], msg, TypeError)

_gen = _gen_n()

if __name__ == "__main__":

    from customtypes import Text, Iterable, Callable

    x: Text = "Hello World!"
    y: Iterable = np.array([1, 2, 3])
    f: np.ndarray = []
    for k, v in __annotations__.items():
        _typecheck(k, v)
