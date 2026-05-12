"""Minimal stub of the Serializable base class.

Other classes inherit from Serializable but we strip all actual
serialization logic (save/load/Serializer/Deserializer).
"""


class Serializable:
    """Minimal base class kept so that subclass declarations don't break."""
    pass
