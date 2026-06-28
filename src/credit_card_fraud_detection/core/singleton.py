import threading

class SingletonMeta(type):
    """
    A thread-safe implementation of the Singleton pattern using metaclasses.
    """
    _instances = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # Acquire lock to ensure only one thread enters the creation block
            with cls._lock:
                # Second check (after locking) to prevent race conditions
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]