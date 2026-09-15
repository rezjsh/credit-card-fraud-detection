import threading

class SingletonMeta(type):
    """
    A thread-safe implementation of the Singleton pattern using metaclasses.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            if not hasattr(cls, '_class_lock'):
                cls._class_lock = threading.Lock()
            # Acquire lock to ensure only one thread enters the creation block
            with cls._class_lock:
                # Second check (after locking) to prevent race conditions
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]