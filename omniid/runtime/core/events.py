class EventBus:
    """
    Decouples the Trainer from the Callback system.
    Supports prioritized registration and execution of events.
    """
    def __init__(self):
        self._callbacks = []

    def register(self, callback):
        self._callbacks.append(callback)
        # Sort by priority, highest first
        self._callbacks.sort(key=lambda c: getattr(c, 'priority', 0), reverse=True)

    def dispatch(self, event_name: str, *args, **kwargs):
        for callback in self._callbacks:
            hook = getattr(callback, event_name, None)
            if hook and callable(hook):
                hook(*args, **kwargs)
