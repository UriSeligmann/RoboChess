import functools
import time

# =============================================
#               DECORATOR UTILS
# =============================================
def debug_entry_exit_method(level: int = 1):
    """
    A method decorator that logs entering and exiting the decorated method.
    
    Args:
        level (int): Logging level threshold. Only methods with a level 
            <= the debugger's configured level will be logged.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Attempt to log entering if 'debugger' is accessible.
            if hasattr(self, "debugger"):
                self.debugger.log(f"Entering {func.__name__}", level)
            result = func(self, *args, **kwargs)
            # Attempt to log exiting if 'debugger' is accessible.
            if hasattr(self, "debugger"):
                self.debugger.log(f"Exiting {func.__name__}", level)
            return result
        return wrapper
    return decorator

def timeit_method(level: int = 1):
    """
    A method decorator that logs the execution time of the decorated method.
    
    Args:
        level (int): Logging level threshold for timing information.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            elapsed = time.time() - start_time
            if hasattr(self, "debugger"):
                self.debugger.log(
                    f"{func.__name__} executed in {elapsed:.2f} seconds", level
                )
            return result
        return wrapper
    return decorator
