from __future__ import annotations

import bisect
import cmath
import math
from dataclasses import dataclass
from itertools import product

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False


# Vacuum constants
MU0 = 4.0e-7 * math.pi
EPS0 = 8.854187817e-12
ETA0 = math.sqrt(MU0 / EPS0)
C0 = 1.0 / math.sqrt(MU0 * EPS0)
INCH_TO_M = 0.0254
GHZ_TO_HZ = 1.0e9

@dataclass
class MaterialTable:
    freq_ghz: list[float]
    eps_r: list[complex]
    mu_r: list[complex]


@dataclass
class LayerConfig:
    thickness_in: float
    anisotropic: bool
    file_0deg: str
    file_90deg: str
    polarization_deg: float
    is_sheet: bool = False
    sheet_resistance: float = 0.0  # ohms/square
    # Optional inverse-design thickness overrides; None means "use the global default".
    inv_t_min_in: float | None = None
    inv_t_max_in: float | None = None
    inv_t_steps: int | None = None


@dataclass
class LoadedLayer:
    thickness_m: float
    anisotropic: bool
    polarization_deg: float
    table_0deg: MaterialTable | None
    table_90deg: MaterialTable | None
    is_sheet: bool = False
    sheet_resistance: float = 0.0


@dataclass
class UncertaintyConfig:
    enabled: bool
    thickness_pct: float
    eps_pct: float
    mu_pct: float


@dataclass
class InverseCandidate:
    score_db: float
    nominal_mean_db: float
    worst_mean_db: float
    avg_mean_db: float
    best_mean_db: float
    thickness_in: list[float]
    material_files: list[str]


def make_sweep(f_start: float, f_stop: float, f_step: float) -> list[float]:
    if f_step <= 0:
        raise ValueError("f_step must be > 0.")
    if f_stop < f_start:
        raise ValueError("f_stop must be >= f_start.")
    count = int(math.floor((f_stop - f_start) / f_step + 1e-12)) + 1
    if count <= 0:
        raise ValueError("Frequency sweep is empty.")
    return [f_start + i * f_step for i in range(count)]


def normalize_backing(backing: str) -> str:
    b = backing.strip().lower().replace("_", "-")
    if b == "pec":
        return "pec"
    if b in {"air", "free-space", "freespace"}:
        return "air"
    raise ValueError(f"Unsupported backing: {backing}")


def interp_complex(x: float, xp: list[float], fp: list[complex]) -> complex:
    if x < xp[0] or x > xp[-1]:
        raise ValueError("Interpolation query is out of bounds.")
    i = bisect.bisect_left(xp, x)
    if i == 0:
        return fp[0]
    if i == len(xp):
        return fp[-1]
    if xp[i] == x:
        return fp[i]
    x0 = xp[i - 1]
    x1 = xp[i]
    t = (x - x0) / (x1 - x0)
    return fp[i - 1] + t * (fp[i] - fp[i - 1])


def interp_complex_many(x: list[float], xp: list[float], fp: list[complex]) -> list[complex]:
    if x[0] < xp[0] or x[-1] > xp[-1]:
        raise ValueError("Interpolation query is out of bounds.")
    if NUMPY_AVAILABLE:
        x_arr = np.asarray(x, dtype=float)
        xp_arr = np.asarray(xp, dtype=float)
        fp_arr = np.asarray(fp, dtype=complex)
        re = np.interp(x_arr, xp_arr, fp_arr.real)
        im = np.interp(x_arr, xp_arr, fp_arr.imag)
        return (re + 1j * im).tolist()
    return [interp_complex(v, xp, fp) for v in x]


def validate_sweep_coverage(sweep: list[float], table: MaterialTable, label: str) -> None:
    if sweep[0] < table.freq_ghz[0] or sweep[-1] > table.freq_ghz[-1]:
        raise ValueError(
            f"Sweep [{sweep[0]}, {sweep[-1]}] GHz is outside {label} data range "
            f"[{table.freq_ghz[0]}, {table.freq_ghz[-1]}] GHz."
        )


def mix_anisotropic(
    eps_0: complex,
    mu_0: complex,
    eps_90: complex,
    mu_90: complex,
    polarization_deg: float,
) -> tuple[complex, complex]:
    theta = math.radians(polarization_deg)
    w0 = math.cos(theta) ** 2
    w90 = math.sin(theta) ** 2
    eps_eff = eps_0 * w0 + eps_90 * w90
    mu_eff = mu_0 * w0 + mu_90 * w90
    return eps_eff, mu_eff


def mix_anisotropic_many(
    eps_0: list[complex],
    mu_0: list[complex],
    eps_90: list[complex],
    mu_90: list[complex],
    polarization_deg: float,
) -> tuple[list[complex], list[complex]]:
    if not (len(eps_0) == len(mu_0) == len(eps_90) == len(mu_90)):
        raise ValueError("Anisotropic property vectors must have matching lengths.")

    theta = math.radians(polarization_deg)
    w0 = math.cos(theta) ** 2
    w90 = math.sin(theta) ** 2
    if NUMPY_AVAILABLE:
        eps_eff = np.asarray(eps_0, dtype=complex) * w0 + np.asarray(eps_90, dtype=complex) * w90
        mu_eff = np.asarray(mu_0, dtype=complex) * w0 + np.asarray(mu_90, dtype=complex) * w90
        return eps_eff.tolist(), mu_eff.tolist()
    eps_eff = [e0 * w0 + e90 * w90 for e0, e90 in zip(eps_0, eps_90)]
    mu_eff = [m0 * w0 + m90 * w90 for m0, m90 in zip(mu_0, mu_90)]
    return eps_eff, mu_eff


def build_uncertainty_scales(cfg: UncertaintyConfig) -> list[tuple[float, float, float]]:
    if not cfg.enabled:
        return [(1.0, 1.0, 1.0)]

    dt = cfg.thickness_pct / 100.0
    de = cfg.eps_pct / 100.0
    dm = cfg.mu_pct / 100.0
    if dt < 0 or de < 0 or dm < 0:
        raise ValueError("Uncertainty percentages must be >= 0.")

    t_vals = [1.0 - dt, 1.0 + dt] if dt > 0 else [1.0]
    e_vals = [1.0 - de, 1.0 + de] if de > 0 else [1.0]
    m_vals = [1.0 - dm, 1.0 + dm] if dm > 0 else [1.0]

    scales = sorted(set(product(t_vals, e_vals, m_vals)))
    if (1.0, 1.0, 1.0) not in scales:
        scales.append((1.0, 1.0, 1.0))
    return scales


def is_nominal_scale(t_scale: float, e_scale: float, m_scale: float, tol: float = 1e-12) -> bool:
    return (
        abs(t_scale - 1.0) <= tol
        and abs(e_scale - 1.0) <= tol
        and abs(m_scale - 1.0) <= tol
    )


def layer_properties(layer: LoadedLayer, f_ghz: float) -> tuple[complex, complex]:
    eps_0 = interp_complex(f_ghz, layer.table_0deg.freq_ghz, layer.table_0deg.eps_r)
    mu_0 = interp_complex(f_ghz, layer.table_0deg.freq_ghz, layer.table_0deg.mu_r)
    if not layer.anisotropic:
        return eps_0, mu_0

    assert layer.table_90deg is not None
    eps_90 = interp_complex(f_ghz, layer.table_90deg.freq_ghz, layer.table_90deg.eps_r)
    mu_90 = interp_complex(f_ghz, layer.table_90deg.freq_ghz, layer.table_90deg.mu_r)
    return mix_anisotropic(eps_0, mu_0, eps_90, mu_90, layer.polarization_deg)


def layer_properties_many(layer: LoadedLayer, f_ghz: list[float]) -> tuple[list[complex], list[complex]]:
    eps_0 = interp_complex_many(f_ghz, layer.table_0deg.freq_ghz, layer.table_0deg.eps_r)
    mu_0 = interp_complex_many(f_ghz, layer.table_0deg.freq_ghz, layer.table_0deg.mu_r)
    if not layer.anisotropic:
        return eps_0, mu_0

    assert layer.table_90deg is not None
    eps_90 = interp_complex_many(f_ghz, layer.table_90deg.freq_ghz, layer.table_90deg.eps_r)
    mu_90 = interp_complex_many(f_ghz, layer.table_90deg.freq_ghz, layer.table_90deg.mu_r)
    return mix_anisotropic_many(eps_0, mu_0, eps_90, mu_90, layer.polarization_deg)


def compute_stack_impedance(
    f_ghz: float,
    layers: list[LoadedLayer],
    backing: str,
    thickness_scale: float = 1.0,
    eps_scale: float = 1.0,
    mu_scale: float = 1.0,
) -> complex:
    if backing == "pec":
        z_load = 0.0 + 0.0j
    elif backing == "air":
        z_load = ETA0 + 0.0j
    else:
        raise ValueError(f"Unsupported backing: {backing}")

    f_hz = f_ghz * GHZ_TO_HZ
    omega = 2.0 * math.pi * f_hz
    k0 = omega / C0

    # Cascade from bottom layer to top layer.
    z_next = z_load
    for layer in reversed(layers):
        if layer.is_sheet:
            rs = complex(layer.sheet_resistance, 0.0)
            z_next = (z_next * rs) / (z_next + rs)
            continue
        eps_r, mu_r = layer_properties(layer, f_ghz)
        eps_r *= eps_scale
        mu_r *= mu_scale
        zc = ETA0 * cmath.sqrt(mu_r / eps_r)
        gamma = 1j * k0 * cmath.sqrt(mu_r * eps_r)
        t = cmath.tanh(gamma * layer.thickness_m * thickness_scale)
        z_next = zc * (z_next + zc * t) / (zc + z_next * t)

    return z_next


def compute_stack_impedance_many(
    f_ghz: list[float],
    layers: list[LoadedLayer],
    backing: str,
    thickness_scale: float = 1.0,
    eps_scale: float = 1.0,
    mu_scale: float = 1.0,
) -> list[complex]:
    if backing == "pec":
        z_load = 0.0 + 0.0j
    elif backing == "air":
        z_load = ETA0 + 0.0j
    else:
        raise ValueError(f"Unsupported backing: {backing}")

    if not f_ghz:
        return []

    if NUMPY_AVAILABLE:
        f_arr_ghz = np.asarray(f_ghz, dtype=float)
        f_hz = f_arr_ghz * GHZ_TO_HZ
        k0 = (2.0 * math.pi * f_hz) / C0
        z_next = np.full_like(f_hz, z_load, dtype=complex)

        for layer in reversed(layers):
            if layer.is_sheet:
                rs = complex(layer.sheet_resistance, 0.0)
                z_next = (z_next * rs) / (z_next + rs)
                continue
            eps_r, mu_r = layer_properties_many(layer, f_ghz)
            eps_arr = np.asarray(eps_r, dtype=complex) * eps_scale
            mu_arr = np.asarray(mu_r, dtype=complex) * mu_scale
            zc = ETA0 * np.sqrt(mu_arr / eps_arr)
            gamma = 1j * k0 * np.sqrt(mu_arr * eps_arr)
            t = np.tanh(gamma * layer.thickness_m * thickness_scale)
            z_next = zc * (z_next + zc * t) / (zc + z_next * t)
        return z_next.tolist()

    return [
        compute_stack_impedance(
            f,
            layers,
            backing,
            thickness_scale=thickness_scale,
            eps_scale=eps_scale,
            mu_scale=mu_scale,
        )
        for f in f_ghz
    ]


def normalize_wave_polarization(pol: str) -> str:
    p = pol.strip().lower()
    if p in {"hh", "te", "hh (te)"}:
        return "te"
    if p in {"vv", "tm", "vv (tm)"}:
        return "tm"
    raise ValueError(f"Unsupported wave polarization: {pol}")


def _stable_kz(z: complex) -> complex:
    # Choose branch with non-negative attenuation.
    if z.imag < 0:
        return -z
    if abs(z.imag) < 1e-12 and z.real < 0:
        return -z
    return z


def layer_wave_params(
    f_hz: float,
    theta_deg: float,
    eps_r: complex,
    mu_r: complex,
    wave_pol: str,
) -> tuple[complex, complex]:
    theta = math.radians(theta_deg)
    k0 = 2.0 * math.pi * f_hz / C0
    kx = k0 * math.sin(theta)
    k_layer = k0 * cmath.sqrt(eps_r * mu_r)
    kz = _stable_kz(cmath.sqrt(k_layer * k_layer - kx * kx))
    omega = 2.0 * math.pi * f_hz
    if wave_pol == "te":
        zc = omega * MU0 * mu_r / kz
    elif wave_pol == "tm":
        zc = kz / (omega * EPS0 * eps_r)
    else:
        raise ValueError(f"Unsupported wave polarization: {wave_pol}")
    return zc, kz


def ambient_wave_impedance(theta_deg: float, wave_pol: str) -> complex:
    theta = math.radians(theta_deg)
    c = math.cos(theta)
    if abs(c) < 1e-9:
        c = 1e-9
    if wave_pol == "te":
        return ETA0 / c
    if wave_pol == "tm":
        return ETA0 * c
    raise ValueError(f"Unsupported wave polarization: {wave_pol}")


def cascade_input_impedance(
    f_ghz: float,
    theta_deg: float,
    layers: list[LoadedLayer],
    wave_pol: str,
    z_load: complex,
    thickness_scale: float = 1.0,
    eps_scale: float = 1.0,
    mu_scale: float = 1.0,
) -> complex:
    f_hz = f_ghz * GHZ_TO_HZ
    z_next = z_load
    for layer in reversed(layers):
        if layer.is_sheet:
            rs = complex(layer.sheet_resistance, 0.0)
            z_next = (z_next * rs) / (z_next + rs)
            continue
        eps_r, mu_r = layer_properties(layer, f_ghz)
        eps_r *= eps_scale
        mu_r *= mu_scale
        zc, kz = layer_wave_params(f_hz, theta_deg, eps_r, mu_r, wave_pol)
        t = cmath.tan(kz * layer.thickness_m * thickness_scale)
        z_next = zc * (z_next + 1j * zc * t) / (zc + 1j * z_next * t)
    return z_next


def cascade_abcd(
    f_ghz: float,
    theta_deg: float,
    layers: list[LoadedLayer],
    wave_pol: str,
    thickness_scale: float = 1.0,
    eps_scale: float = 1.0,
    mu_scale: float = 1.0,
) -> tuple[complex, complex, complex, complex]:
    f_hz = f_ghz * GHZ_TO_HZ
    a = 1.0 + 0.0j
    b = 0.0 + 0.0j
    c = 0.0 + 0.0j
    d = 1.0 + 0.0j

    for layer in layers:
        if layer.is_sheet:
            y_s = 1.0 / layer.sheet_resistance
            a, b, c, d = (a + b * y_s, b, c + d * y_s, d)
            continue
        eps_r, mu_r = layer_properties(layer, f_ghz)
        eps_r *= eps_scale
        mu_r *= mu_scale
        zc, kz = layer_wave_params(f_hz, theta_deg, eps_r, mu_r, wave_pol)
        p = kz * layer.thickness_m * thickness_scale
        ai = cmath.cos(p)
        bi = 1j * zc * cmath.sin(p)
        ci = 1j * cmath.sin(p) / zc
        di = ai
        a, b, c, d = (
            a * ai + b * ci,
            a * bi + b * di,
            c * ai + d * ci,
            c * bi + d * di,
        )

    return a, b, c, d


def _db_from_mag(x: complex) -> float:
    mag = max(abs(x), 1e-15)
    return 20.0 * math.log10(mag)


def _db_from_power(x: float) -> float:
    return 10.0 * math.log10(max(x, 1e-15))


def _db_from_power_many(x: "np.ndarray") -> "np.ndarray":
    return 10.0 * np.log10(np.maximum(x, 1e-15))


def _stable_kz_many(z: "np.ndarray") -> "np.ndarray":
    out = np.asarray(z, dtype=complex)
    mask = (out.imag < 0.0) | ((np.abs(out.imag) < 1e-12) & (out.real < 0.0))
    return np.where(mask, -out, out)


def _db_from_mag_many(x: "np.ndarray") -> "np.ndarray":
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-15))


def compute_angle_metrics_many(
    f_ghz: list[float],
    theta_deg: float,
    layers: list[LoadedLayer],
    wave_pol: str,
    thickness_scale: float = 1.0,
    eps_scale: float = 1.0,
    mu_scale: float = 1.0,
) -> dict[str, list[float]]:
    if not f_ghz:
        return {
            "metal_loss_db": [],
            "metal_phase_deg": [],
            "air_loss_db": [],
            "air_phase_deg": [],
            "insertion_loss_db": [],
            "insertion_phase_deg": [],
            "metal_absorption_db": [],
            "air_absorption_db": [],
        }

    if not NUMPY_AVAILABLE:
        rows = [
            compute_angle_metrics(
                f,
                theta_deg,
                layers,
                wave_pol,
                thickness_scale=thickness_scale,
                eps_scale=eps_scale,
                mu_scale=mu_scale,
            )
            for f in f_ghz
        ]
        return {
            "metal_loss_db": [r["metal_loss_db"] for r in rows],
            "metal_phase_deg": [r["metal_phase_deg"] for r in rows],
            "air_loss_db": [r["air_loss_db"] for r in rows],
            "air_phase_deg": [r["air_phase_deg"] for r in rows],
            "insertion_loss_db": [r["insertion_loss_db"] for r in rows],
            "insertion_phase_deg": [r["insertion_phase_deg"] for r in rows],
            "metal_absorption_db": [r["metal_absorption_db"] for r in rows],
            "air_absorption_db": [r["air_absorption_db"] for r in rows],
        }

    theta = math.radians(theta_deg)
    sin_t = math.sin(theta)
    z0 = ambient_wave_impedance(theta_deg, wave_pol)
    f_arr_ghz = np.asarray(f_ghz, dtype=float)
    f_hz = f_arr_ghz * GHZ_TO_HZ
    omega = 2.0 * math.pi * f_hz
    k0 = omega / C0
    kx = k0 * sin_t

    if wave_pol not in {"te", "tm"}:
        raise ValueError(f"Unsupported wave polarization: {wave_pol}")

    # Cache layer wave terms once; they are reused by both reflection cascades and ABCD propagation.
    # For sheet layers, terms entry is None and sheet_rs stores the resistance.
    layer_terms: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] = []
    layer_sheet_rs: list[float] = []
    for layer in layers:
        if layer.is_sheet:
            layer_terms.append(None)
            layer_sheet_rs.append(layer.sheet_resistance)
            continue
        layer_sheet_rs.append(0.0)
        eps_r, mu_r = layer_properties_many(layer, f_ghz)
        eps_arr = np.asarray(eps_r, dtype=complex) * eps_scale
        mu_arr = np.asarray(mu_r, dtype=complex) * mu_scale
        k_layer = k0 * np.sqrt(eps_arr * mu_arr)
        kz = _stable_kz_many(np.sqrt(k_layer * k_layer - kx * kx))
        if wave_pol == "te":
            zc = omega * MU0 * mu_arr / kz
        else:
            zc = kz / (omega * EPS0 * eps_arr)
        p = kz * layer.thickness_m * thickness_scale
        layer_terms.append((zc, np.tan(p), np.sin(p), np.cos(p)))

    def _cascade(z_load: complex) -> np.ndarray:
        z_next = np.full_like(f_hz, z_load, dtype=complex)
        for idx in reversed(range(len(layer_terms))):
            if layer_terms[idx] is None:
                rs = complex(layer_sheet_rs[idx], 0.0)
                z_next = (z_next * rs) / (z_next + rs)
            else:
                zc, t, _si, _co = layer_terms[idx]
                z_next = zc * (z_next + 1j * zc * t) / (zc + 1j * z_next * t)
        return z_next

    zin_metal = _cascade(0.0 + 0.0j)
    gamma_metal = (zin_metal - z0) / (zin_metal + z0)

    zin_air = _cascade(z0)
    gamma_air = (zin_air - z0) / (zin_air + z0)

    a = np.ones_like(f_hz, dtype=complex)
    b = np.zeros_like(f_hz, dtype=complex)
    c = np.zeros_like(f_hz, dtype=complex)
    d = np.ones_like(f_hz, dtype=complex)

    for idx in range(len(layer_terms)):
        if layer_terms[idx] is None:
            y_s = 1.0 / layer_sheet_rs[idx]
            a, b, c, d = (a + b * y_s, b, c + d * y_s, d)
        else:
            zc, _t, si, ai = layer_terms[idx]
            bi = 1j * zc * si
            ci = 1j * si / zc
            di = ai
            a, b, c, d = (
                a * ai + b * ci,
                a * bi + b * di,
                c * ai + d * ci,
                c * bi + d * di,
            )

    den = a + b / z0 + c * z0 + d
    s21 = 2.0 / den

    metal_abs_frac = 1.0 - np.abs(gamma_metal) ** 2
    air_abs_frac = 1.0 - np.abs(gamma_air) ** 2 - np.abs(s21) ** 2

    return {
        "metal_loss_db": _db_from_mag_many(gamma_metal).tolist(),
        "metal_phase_deg": np.degrees(np.angle(gamma_metal)).tolist(),
        "air_loss_db": _db_from_mag_many(gamma_air).tolist(),
        "air_phase_deg": np.degrees(np.angle(gamma_air)).tolist(),
        "insertion_loss_db": _db_from_mag_many(s21).tolist(),
        "insertion_phase_deg": np.degrees(np.angle(s21)).tolist(),
        "metal_absorption_db": _db_from_power_many(metal_abs_frac).tolist(),
        "air_absorption_db": _db_from_power_many(air_abs_frac).tolist(),
    }


def compute_angle_metrics(
    f_ghz: float,
    theta_deg: float,
    layers: list[LoadedLayer],
    wave_pol: str,
    thickness_scale: float = 1.0,
    eps_scale: float = 1.0,
    mu_scale: float = 1.0,
) -> dict[str, float]:
    z0 = ambient_wave_impedance(theta_deg, wave_pol)

    # Metal-backed reflection.
    zin_metal = cascade_input_impedance(
        f_ghz,
        theta_deg,
        layers,
        wave_pol,
        0.0 + 0.0j,
        thickness_scale=thickness_scale,
        eps_scale=eps_scale,
        mu_scale=mu_scale,
    )
    gamma_metal = (zin_metal - z0) / (zin_metal + z0)

    # Air-backed reflection.
    zin_air = cascade_input_impedance(
        f_ghz,
        theta_deg,
        layers,
        wave_pol,
        z0,
        thickness_scale=thickness_scale,
        eps_scale=eps_scale,
        mu_scale=mu_scale,
    )
    gamma_air = (zin_air - z0) / (zin_air + z0)

    # Through transmission in air (insertion).
    a, b, c, d = cascade_abcd(
        f_ghz,
        theta_deg,
        layers,
        wave_pol,
        thickness_scale=thickness_scale,
        eps_scale=eps_scale,
        mu_scale=mu_scale,
    )
    den = a + b / z0 + c * z0 + d
    s21 = 2.0 / den

    return {
        "metal_loss_db": _db_from_mag(gamma_metal),
        "metal_phase_deg": math.degrees(cmath.phase(gamma_metal)),
        "air_loss_db": _db_from_mag(gamma_air),
        "air_phase_deg": math.degrees(cmath.phase(gamma_air)),
        "insertion_loss_db": _db_from_mag(s21),
        "insertion_phase_deg": math.degrees(cmath.phase(s21)),
        "metal_absorption_db": _db_from_power(1.0 - abs(gamma_metal) ** 2),
        "air_absorption_db": _db_from_power(
            1.0 - abs(gamma_air) ** 2 - abs(s21) ** 2
        ),
    }