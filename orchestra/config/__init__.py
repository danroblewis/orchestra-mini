"""The config file (SPEC §1): YAML or JSON, in layers, refusing what it does not know."""

from .build import ConfigError, OldConfig, build_config, parse_duration
from .load import home, layers
from .schema import Config, Machine, Projection, Store

__all__ = ["Config", "ConfigError", "Machine", "OldConfig", "Projection", "Store", "build_config", "home", "layers", "parse_duration"]
