from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


Config = dict[str, Any]


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping at top level: {config_path}")
    data["_config_path"] = str(config_path)
    return data


def write_config(path: str | Path, config: Config) -> None:
    config_to_write = copy.deepcopy(config)
    config_to_write.pop("_config_path", None)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_write, handle, sort_keys=False)


def deep_update(base: Config, updates: Config) -> Config:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_override(raw_override: str) -> Config:
    if "=" not in raw_override:
        raise ValueError(f"Override must use key=value syntax: {raw_override!r}")
    key_path, raw_value = raw_override.split("=", 1)
    keys = [part for part in key_path.split(".") if part]
    if not keys:
        raise ValueError(f"Override has an empty key path: {raw_override!r}")

    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError:
        value = raw_value

    nested: Config = value
    for key in reversed(keys):
        nested = {key: nested}
    return nested


def apply_overrides(config: Config, overrides: list[str] | None) -> Config:
    result = copy.deepcopy(config)
    for raw_override in overrides or []:
        result = deep_update(result, parse_override(raw_override))
    return result


def get_nested(config: Config, path: str, default: Any = None) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def append_flag(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    command.extend([flag, str(value)])


def extend_repeated(command: list[str], flag: str, values: Any) -> None:
    if values is None:
        return
    if isinstance(values, str):
        command.extend([flag, values])
        return
    command.append(flag)
    command.extend(str(value) for value in values)
