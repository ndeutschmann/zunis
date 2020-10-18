"""Configuration class"""
from dictwrapper import NestedMapping


class Configuration(NestedMapping):
    """Nested dictionary like object with top level access to all leaf-level mappings"""