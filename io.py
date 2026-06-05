from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .compute import LayerConfig, MaterialTable

PROJECT_SCHEMA_VERSION = 1


def read_material_table(path: Path, skiprows: int) -> MaterialTable:
    rows: list[tuple[float, complex, complex]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < skiprows:
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 5:
                raise ValueError(
                    f"{path}: line {idx + 1} expected 5 columns, found {len(parts)}."
                )
            freq_ghz = float(parts[0])
            # e^{+jwt} convention: loss is a negative imaginary part. The file
            # supplies eps''/mu'' already signed (negative for lossy), so the
            # column values are used directly as the imaginary parts.
            eps = complex(float(parts[1]), float(parts[2]))
            mu = complex(float(parts[3]), float(parts[4]))
            rows.append((freq_ghz, eps, mu))

    if len(rows) < 2:
        raise ValueError(f"{path}: at least two valid data rows are required.")

    rows.sort(key=lambda r: r[0])
    return MaterialTable(
        freq_ghz=[r[0] for r in rows],
        eps_r=[r[1] for r in rows],
        mu_r=[r[2] for r in rows],
    )


def write_output(path: Path, rows: list[tuple[float, float, float]], include_header: bool) -> None:
    with path.open("w", encoding="utf-8") as f:
        if include_header:
            f.write("frequency_GHz z_r z_i\n")
        for freq_ghz, zr, zi in rows:
            f.write(f"{freq_ghz:.12g} {zr:.12g} {zi:.12g}\n")


def layer_config_to_dict(layer: LayerConfig) -> dict[str, Any]:
    d: dict[str, Any] = {
        "thickness_in": layer.thickness_in,
        "anisotropic": layer.anisotropic,
        "file_0deg": layer.file_0deg,
        "file_90deg": layer.file_90deg,
        "polarization_deg": layer.polarization_deg,
    }
    if layer.is_sheet:
        d["is_sheet"] = True
        d["sheet_resistance"] = layer.sheet_resistance
    if layer.inv_t_min_in is not None:
        d["inv_t_min_in"] = layer.inv_t_min_in
    if layer.inv_t_max_in is not None:
        d["inv_t_max_in"] = layer.inv_t_max_in
    if layer.inv_t_steps is not None:
        d["inv_t_steps"] = layer.inv_t_steps
    return d


def layer_config_from_dict(data: dict[str, Any], index: int = 0) -> LayerConfig:
    label = f"Layer {index}" if index > 0 else "Layer"

    is_sheet = bool(data.get("is_sheet", False))
    if is_sheet:
        try:
            sheet_resistance = float(data.get("sheet_resistance", 0.0))
        except Exception as exc:
            raise ValueError(f"{label}: invalid sheet_resistance.") from exc
        if sheet_resistance <= 0:
            raise ValueError(f"{label}: sheet_resistance must be > 0.")
        return LayerConfig(
            thickness_in=0.0,
            anisotropic=False,
            file_0deg="",
            file_90deg="",
            polarization_deg=0.0,
            is_sheet=True,
            sheet_resistance=sheet_resistance,
        )

    try:
        thickness_in = float(data.get("thickness_in", 0.0))
    except Exception as exc:
        raise ValueError(f"{label}: invalid thickness_in.") from exc
    if thickness_in <= 0:
        raise ValueError(f"{label}: thickness_in must be > 0.")

    anisotropic = bool(data.get("anisotropic", False))
    file_0deg = str(data.get("file_0deg", "")).strip()
    file_90deg = str(data.get("file_90deg", "")).strip()
    try:
        polarization_deg = float(data.get("polarization_deg", 0.0))
    except Exception as exc:
        raise ValueError(f"{label}: invalid polarization_deg.") from exc

    if not file_0deg:
        raise ValueError(f"{label}: file_0deg is required.")
    if anisotropic and not file_90deg:
        raise ValueError(f"{label}: file_90deg is required for anisotropic layers.")

    inv_t_min_in = _parse_optional_float(data.get("inv_t_min_in"), label, "inv_t_min_in")
    inv_t_max_in = _parse_optional_float(data.get("inv_t_max_in"), label, "inv_t_max_in")
    inv_t_steps = _parse_optional_int(data.get("inv_t_steps"), label, "inv_t_steps")
    if inv_t_min_in is not None and inv_t_min_in <= 0:
        raise ValueError(f"{label}: inv_t_min_in must be > 0.")
    if inv_t_max_in is not None and inv_t_max_in <= 0:
        raise ValueError(f"{label}: inv_t_max_in must be > 0.")
    if (
        inv_t_min_in is not None
        and inv_t_max_in is not None
        and inv_t_max_in < inv_t_min_in
    ):
        raise ValueError(f"{label}: inv_t_max_in must be >= inv_t_min_in.")
    if inv_t_steps is not None and inv_t_steps < 1:
        raise ValueError(f"{label}: inv_t_steps must be >= 1.")

    return LayerConfig(
        thickness_in=thickness_in,
        anisotropic=anisotropic,
        file_0deg=file_0deg,
        file_90deg=file_90deg,
        polarization_deg=polarization_deg,
        inv_t_min_in=inv_t_min_in,
        inv_t_max_in=inv_t_max_in,
        inv_t_steps=inv_t_steps,
    )


def _parse_optional_float(value: Any, label: str, field: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{label}: invalid {field}.") from exc


def _parse_optional_int(value: Any, label: str, field: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{label}: invalid {field}.") from exc


def save_project_file(path: Path, state: dict[str, Any]) -> None:
    payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "state": state,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_project_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Project file must be a JSON object.")

    schema_version = payload.get("schema_version")
    if schema_version != PROJECT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported project schema version: {schema_version}. "
            f"Expected {PROJECT_SCHEMA_VERSION}."
        )

    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("Project file is missing a valid 'state' object.")
    return state