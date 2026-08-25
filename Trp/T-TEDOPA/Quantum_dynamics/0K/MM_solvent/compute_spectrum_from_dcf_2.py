#!/usr/bin/env python3
"""
Continuous Fourier transform of dipole correlation function (dcf).

Workflow:
1) Load time-domain dcf from .h5/.hdf5 or text (.dat/.txt)
2) Optional exponential dephasing
3) Optional cosine windowing
4) Optional phase unwrapping to correct phase discontinuities
5) Define Energy Grid (based on XLIM settings)
6) Compute FT via Trapezoidal integration: F(E) = Integral [ dcf(t) * exp(iEt) ] dt
7) Save energy-resolved real/imag parts and plot

Usage:
    compute_spectrum_from_dcf.py <input_file> [decay_constant_au]
      input_file: .h5/.hdf5 with datasets under 'data/' or text with 2 cols: time(au) Re
"""

import sys
import os
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= Configuration ================= #

# Unit conversion
HARTREE_TO_EV = 27.2114
FS_TO_AU = 41.3413745758

# Energy-axis shift in eV applied to the PLOT axis.
SHIFT_CENTER = 0.0

# X-axis limits for calculation and plotting in eV
XLIM_LOWER = -3.0
XLIM_UPPER = 6.0

# Number of points in the energy grid. Higher is smoother.
NUM_POINTS = 4500

# Default exponential damping constant in a.u. (if CLI arg omitted)
TAU_AU = None

# Enable phase unwrapping to correct potential phase discontinuities
ENABLE_PHASE_UNWRAP = False

# Enable cosine windowing to have response function decay to zero at t_final (recommended)
APPLY_COSINE_WINDOW = False

# ================= Utilities ================= #
def _coerce_bool(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (bytes, str)):
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        text = text.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return None
    if isinstance(value, (int, float, np.number)):
        return bool(value)
    return None

def _coerce_float(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        else:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _resolve_dynamic_xlims(
    spectrum_axis_mode=None,
    spectrum_center_energy_hartree=None,
    energy_ref_hartree=None,
    use_absolute_energies=None,
    energy_1=None,
):
    axis_mode = None
    if isinstance(spectrum_axis_mode, bytes):
        axis_mode = spectrum_axis_mode.decode("utf-8").strip().lower()
    elif isinstance(spectrum_axis_mode, str):
        axis_mode = spectrum_axis_mode.strip().lower()
    if axis_mode not in {"absolute", "relative"}:
        axis_mode = None

    center_hartree = _coerce_float(spectrum_center_energy_hartree)
    if center_hartree is None:
        center_hartree = _coerce_float(energy_ref_hartree)
    if center_hartree is None:
        center_hartree = _coerce_float(energy_1)

    if axis_mode == "absolute":
        if center_hartree is None:
            return XLIM_LOWER, XLIM_UPPER
        center_ev = center_hartree * HARTREE_TO_EV
        return center_ev - 2.5, center_ev + 2.5
    if axis_mode == "relative":
        return -2.5, 2.5

    use_absolute_energies = _coerce_bool(use_absolute_energies)
    if use_absolute_energies is None:
        return XLIM_LOWER, XLIM_UPPER
    if use_absolute_energies:
        if center_hartree is None:
            return XLIM_LOWER, XLIM_UPPER
        center_ev = center_hartree * HARTREE_TO_EV
        return center_ev - 2.5, center_ev + 2.5
    return -2.5, 2.5

def _read_hdf5_parameters(input_path):
    try:
        with h5py.File(input_path, "r") as f:
            if "parameters" not in f:
                return {}
            params = {}
            for key, ds in f["parameters"].items():
                if isinstance(ds, h5py.Group):
                    continue
                val = ds[()]
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                if isinstance(val, np.ndarray) and val.shape == ():
                    val = val.item()
                params[key] = val
            return params
    except Exception as e:
        print(f"Warning: failed to read parameters from HDF5: {e}")
        return {}

def _load_hdf5_series(f, dataset_name):
    dataset_path = f"data/{dataset_name}"
    re_path = f"data/{dataset_name}-re"
    im_path = f"data/{dataset_name}-im"

    if dataset_path in f:
        dcf_complex = f[dataset_path][:].astype(complex)
        real_dcf = np.real(dcf_complex)
        imag_dcf = np.imag(dcf_complex)
        # print(f"Loaded dcf from '{dataset_path}'")
        return real_dcf, imag_dcf

    if re_path in f and im_path in f:
        real_dcf = f[re_path][:].astype(float)
        imag_dcf = f[im_path][:].astype(float)
        # print(f"Loaded dcf from '{re_path}' and '{im_path}'")
        return real_dcf, imag_dcf

    return None

def load_dcf_series(input_path, dcf_only=False):
    """Load one or more dcf series from either text or HDF5."""
    file_ext = os.path.splitext(input_path)[1].lower()
    if file_ext in (".h5", ".hdf5"):
        return load_hdf5_dcf_series(input_path, dcf_only=dcf_only)
    return load_text_dcf_series(input_path)

def load_data(input_path):
    """Load a single dcf series from either text or HDF5 file."""
    time_au, dcf_series = load_dcf_series(input_path, dcf_only=True)
    _, real_dcf, imag_dcf = dcf_series[0]
    return time_au, real_dcf, imag_dcf

def load_text_dcf_series(input_path):
    """Load a single dcf series from text file with 2 columns: time(au), Re."""
    data = np.loadtxt(input_path, comments="#", dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        print("Error: input must have at least 2 columns: time(au), Re")
        sys.exit(1)
    time_au = data[:, 0].astype(float)
    real_dcf = data[:, 1].astype(float)
    imag_dcf = np.zeros_like(real_dcf)
    return time_au, [("dcf", real_dcf, imag_dcf)]

def load_hdf5_data(input_path):
    """Load dcf data from HDF5 file."""
    time_au, dcf_series = load_hdf5_dcf_series(input_path, dcf_only=True)
    _, real_dcf, imag_dcf = dcf_series[0]
    return time_au, real_dcf, imag_dcf

def load_hdf5_dcf_series(input_path, dcf_only=False):
    """Load one or more dcf series from HDF5."""
    try:
        with h5py.File(input_path, "r") as f:
            if "data/times" not in f:
                print("Error: 'data/times' dataset not found in HDF5 file")
                sys.exit(1)

            time_au = f["data/times"][:].astype(float)
            dcf_series = []

            base_series = _load_hdf5_series(f, "dcf")
            if base_series is None:
                print("Error: Neither 'data/dcf' nor 'data/dcf-re'/'data/dcf-im' datasets found")
                sys.exit(1)
            dcf_series.append(("dcf", base_series[0], base_series[1]))

            if not dcf_only:
                for dataset_name in ("dcf1", "dcf2"):
                    series = _load_hdf5_series(f, dataset_name)
                    if series is not None:
                        dcf_series.append((dataset_name, series[0], series[1]))

            for dataset_name, real_dcf, imag_dcf in dcf_series:
                if len(time_au) != len(real_dcf) or len(time_au) != len(imag_dcf):
                    print(
                        "Error: time, real, and imaginary arrays have different lengths for '{}'".format(
                            dataset_name
                        )
                    )
                    sys.exit(1)

            return time_au, dcf_series

    except Exception as e:
        print("Error reading HDF5 file: {}".format(e))
        sys.exit(1)

def apply_cosine_window(time_au, real_dcf, imag_dcf):
    """Apply a simple cosine window to reduce spectral leakage at the end of the signal."""
    t_final = time_au[-1] - time_au[0]
    if t_final == 0:
        return time_au, real_dcf, imag_dcf
    cosine_window = np.cos(np.pi * time_au / (2 * t_final))
    real_dcf = real_dcf * cosine_window
    imag_dcf = imag_dcf * cosine_window
    return time_au, real_dcf, imag_dcf

def unwrap_phase(time_au, real_dcf, imag_dcf, energy_shift_hartree=None):
    """
    Unwrap phase to correct discontinuities and optionally apply energy shift.
    
    This function converts the dcf to magnitude-phase representation, unwraps
    the phase by detecting jumps > 0.7π, optionally applies an energy shift
    to the phase, and reconstructs the complex signal.
    
    Args:
        time_au: array of time points in atomic units
        real_dcf: real part of dcf
        imag_dcf: imaginary part of dcf
        energy_shift_hartree: optional energy shift in Hartree to apply to phase.
                              If None, no phase-based energy shift is applied.
    
    Returns:
        real_dcf_unwrapped, imag_dcf_unwrapped: corrected real and imaginary parts
    """
    # Convert to magnitude and phase
    dcf_complex = real_dcf + 1j * imag_dcf
    magnitude = np.abs(dcf_complex)
    phase = np.angle(dcf_complex)
    
    # Unwrap phase: detect and correct discontinuities
    phase_fac = 0.0
    for i in range(len(phase) - 1):
        phase[i] = phase[i] + phase_fac
        # Check for discontinuous jump (> 0.7π)
        if abs(phase[i] - phase_fac - phase[i + 1]) > 0.7 * np.pi:
            diff = phase[i + 1] - (phase[i] - phase_fac)
            frac = diff / np.pi
            n = int(round(frac))
            phase_fac = phase_fac - np.pi * n
    
    # Apply phase correction to last point
    phase[-1] = phase[-1] + phase_fac
    
    # Optionally apply energy shift to phase
    if energy_shift_hartree is not None:
        phase = phase - energy_shift_hartree * time_au
    
    # Reconstruct complex signal from magnitude and unwrapped phase
    dcf_unwrapped = magnitude * np.exp(1j * phase)
    
    return np.real(dcf_unwrapped), np.imag(dcf_unwrapped)

def compute_continuous_ft(time_au, signal, energy_grid_ev):
    """
    Compute the continuous Fourier Transform using numerical integration.
    
    F(E) = Integral [ signal(t) * exp(i * omega * t) ] dt
    where omega = E / h_bar.
    
    We use the sign convention exp(+iwt) which corresponds to absorption 
    spectra conventions often used when reversing FFT frequencies.
    
    Args:
        time_au: array of time points in atomic units.
        signal: complex array of signal values.
        energy_grid_ev: array of energy points in eV (the target axis).
        
    Returns:
        spectrum: complex array corresponding to the FT at requested energies.
    """
    # Convert eV to atomic units of frequency (energy)
    omegas_au = energy_grid_ev / HARTREE_TO_EV
    
    spectrum = np.zeros_like(omegas_au, dtype=complex)
    
    # print(f"Integrating {len(time_au)} time points over {len(omegas_au)} energy points...")
    
    # Loop over energies to calculate integral.
    # This is memory efficient (avoids N_time x N_energy matrix).
    for i, omega in enumerate(omegas_au):
        # Phase = omega * t
        # Kernel = exp(i * omega * t)
        kernel = np.exp(1j * omega * time_au)
        
        # Integrand = f(t) * kernel
        integrand = signal * kernel
        
        # Trapezoidal integration over time
        spectrum[i] = np.trapz(integrand, x=time_au)
        
    return spectrum


def _normalize_spectrum_prefactor_mode(mode):
    if mode is None:
        return "none"
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8")
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        if normalized in {"none", "omega", "omega3"}:
            return normalized
    return "none"


def _spectrum_prefactor_values(energy_grid_ev, mode):
    mode = _normalize_spectrum_prefactor_mode(mode)
    clipped = np.clip(np.asarray(energy_grid_ev, dtype=float), 0.0, None)
    if mode == "omega":
        return clipped
    if mode == "omega3":
        return clipped ** 3
    return np.ones_like(clipped)


def _prefactor_series_label(mode, base_label):
    mode = _normalize_spectrum_prefactor_mode(mode)
    if mode == "omega":
        return f"Energy * {base_label}"
    if mode == "omega3":
        return f"Energy^3 * {base_label}"
    return base_label


def compute_spectrum_from_dcf_data(
    times,
    dcf_data,
    output_basename=None,
    spectrum_axis_mode=None,
    spectrum_center_energy_hartree=None,
    energy_ref_hartree=None,
    use_absolute_energies=None,
    energy_1=None,
    overlay_series=None,
    phase_ev=0.0,
    spectrum_prefactor_mode="none",
):
    """
    Compute spectrum (continuous FT of dcf) from in-memory data and optionally save outputs.
    
    Args:
        times: 1D array-like of time points (a.u.).
        dcf_data: Array-like of dcf values. Can be complex 1D, or 2-column real/imag, or NxM (first column used).
        output_basename: Path prefix for outputs. If provided with extension, the extension
                         is preserved for the PDF and a .dat is created alongside.
        spectrum_axis_mode: Optional axis mode ("absolute" or "relative").
        spectrum_center_energy_hartree: Optional Hartree center for absolute axis.
        energy_ref_hartree: Optional Hartree reference used as fallback center for absolute axis.
        use_absolute_energies: Legacy flag for absolute/relative window selection.
        energy_1: Legacy Hartree center used with `use_absolute_energies`.
        overlay_series: Optional list of `(label, series)` to overlay additional spectra.
        phase_ev: Optional static phase in radians used to premultiply dcf by exp(i*phase).
        spectrum_prefactor_mode: Optional mode for FT-axis prefactor scaling:
            "none" (default), "omega" (max(E,0)), "omega3" (max(E,0)^3).
    
    Returns:
        energy_grid_ev, real_ft, imag_ft
    """
    times_arr = np.asarray(times).flatten()
    dcf_arr = np.asarray(dcf_data)

    # Normalize shape
    if dcf_arr.ndim == 2 and dcf_arr.shape[1] >= 2 and not np.iscomplexobj(dcf_arr):
        dcf_arr = dcf_arr[:, 0] + 1j * dcf_arr[:, 1]
    elif dcf_arr.ndim > 1:
        dcf_arr = dcf_arr[:, 0]
    dcf_arr = dcf_arr.astype(complex)

    # Optional preprocessing mirroring CLI defaults
    if APPLY_COSINE_WINDOW:
        times_arr, real_dcf, imag_dcf = apply_cosine_window(times_arr, np.real(dcf_arr), np.imag(dcf_arr))
    else:
        real_dcf, imag_dcf = np.real(dcf_arr), np.imag(dcf_arr)

    if ENABLE_PHASE_UNWRAP:
        phase_energy_shift = SHIFT_CENTER / HARTREE_TO_EV if SHIFT_CENTER != 0.0 else None
        real_dcf, imag_dcf = unwrap_phase(times_arr, real_dcf, imag_dcf, phase_energy_shift)

    dcf_complex = real_dcf + 1j * imag_dcf
    phase_ev = float(phase_ev) if phase_ev is not None else 0.0
    if phase_ev != 0.0:
        dcf_complex = dcf_complex * np.exp(1j * phase_ev)

    xlim_lower, xlim_upper = _resolve_dynamic_xlims(
        spectrum_axis_mode=spectrum_axis_mode,
        spectrum_center_energy_hartree=spectrum_center_energy_hartree,
        energy_ref_hartree=energy_ref_hartree,
        use_absolute_energies=use_absolute_energies,
        energy_1=energy_1,
    )
    e_min = xlim_lower if xlim_lower is not None else -5.0
    e_max = xlim_upper if xlim_upper is not None else 10.0
    n_pts = int(NUM_POINTS) if NUM_POINTS is not None else 3000

    energy_grid_ev = np.linspace(e_min, e_max, n_pts)
    energy_for_computation = energy_grid_ev - SHIFT_CENTER
    ft_complex = compute_continuous_ft(times_arr, dcf_complex, energy_for_computation)
    prefactor = _spectrum_prefactor_values(energy_grid_ev, spectrum_prefactor_mode)
    real_ft = np.real(ft_complex) * prefactor
    imag_ft = np.imag(ft_complex)

    spectra_for_plot = [("dcf", real_ft)]
    if overlay_series:
        for label, series in overlay_series:
            overlay_arr = np.asarray(series)
            if overlay_arr.ndim == 2 and overlay_arr.shape[1] >= 2 and not np.iscomplexobj(overlay_arr):
                overlay_arr = overlay_arr[:, 0] + 1j * overlay_arr[:, 1]
            elif overlay_arr.ndim > 1:
                overlay_arr = overlay_arr[:, 0]
            overlay_arr = overlay_arr.astype(complex)
            if overlay_arr.shape[0] != times_arr.shape[0]:
                print(
                    f"Warning: skipping overlay '{label}' due to length mismatch "
                    f"({overlay_arr.shape[0]} vs {times_arr.shape[0]})."
                )
                continue

            overlay_real = np.real(overlay_arr)
            overlay_imag = np.imag(overlay_arr)
            if APPLY_COSINE_WINDOW:
                _, overlay_real, overlay_imag = apply_cosine_window(times_arr, overlay_real, overlay_imag)
            if ENABLE_PHASE_UNWRAP:
                phase_energy_shift = SHIFT_CENTER / HARTREE_TO_EV if SHIFT_CENTER != 0.0 else None
                overlay_real, overlay_imag = unwrap_phase(
                    times_arr,
                    overlay_real,
                    overlay_imag,
                    phase_energy_shift,
                )
            overlay_complex = overlay_real + 1j * overlay_imag
            if phase_ev != 0.0:
                overlay_complex = overlay_complex * np.exp(1j * phase_ev)
            overlay_ft = np.real(compute_continuous_ft(times_arr, overlay_complex, energy_for_computation))
            overlay_ft = overlay_ft * prefactor
            spectra_for_plot.append((str(label), overlay_ft))

    if output_basename is not None:
        base_root, ext = os.path.splitext(output_basename)
        if ext == "":
            png_path = base_root + ".png"
        else:
            png_path = output_basename
        dat_path = base_root + ".dat"

        arr = np.column_stack([energy_grid_ev, real_ft])
        np.savetxt(dat_path, arr, header="Energy_eV Re_FT")
        plot_spectra_with_dcf(
            energy_grid_ev,
            spectra_for_plot,
            times_arr,
            real_dcf,
            imag_dcf,
            png_path,
            xlim_lower=xlim_lower,
            xlim_upper=xlim_upper,
            spectrum_prefactor_mode=spectrum_prefactor_mode,
        )
        # print(f"Wrote spectrum files: {dat_path}, {png_path}")

    return energy_grid_ev, real_ft, imag_ft

def plot_spectra_with_dcf(
    energy_ev,
    spectra,
    time_au,
    real_dcf,
    imag_dcf,
    out_png,
    xlim_lower=None,
    xlim_upper=None,
    spectrum_prefactor_mode="none",
    envelope_tau_au=None,
):
    """Plot a three-panel figure with one or more spectra and the primary time-domain dcf."""
    fig, (ax_top, ax_mid, ax_bottom) = plt.subplots(3, 1, figsize=(7, 11))

    def _canonical_spectrum_label(label):
        normalized = str(label).strip().lower().replace(" ", "")
        if normalized in ("dcf", "re(ft(dcf))"):
            return "dcf"
        if normalized in ("dcf1", "mu_1", "re(ft(dcf1))", "re(ft(mu_1))"):
            return "dcf1"
        if normalized in ("dcf2", "mu_2", "re(ft(dcf2))", "re(ft(mu_2))"):
            return "dcf2"
        return None

    color_by_series = {"dcf": "k", "dcf1": "r", "dcf2": "b"}
    legend_label_by_series = {
        "dcf": _prefactor_series_label(spectrum_prefactor_mode, "Re(FT(dcf))"),
        "dcf1": _prefactor_series_label(spectrum_prefactor_mode, "Re(FT(dcf1))"),
        "dcf2": _prefactor_series_label(spectrum_prefactor_mode, "Re(FT(dcf2))"),
    }

    # Top panel: spectrum
    for raw_label, real_ft in spectra:
        series_key = _canonical_spectrum_label(raw_label)
        line_color = color_by_series.get(series_key, None)
        line_label = legend_label_by_series.get(series_key, str(raw_label))
        ax_top.plot(energy_ev, real_ft, lw=1.0, color=line_color, label=line_label)
    ax_top.set_xlabel("Energy (eV)")
    ax_top.set_ylabel(_prefactor_series_label(spectrum_prefactor_mode, "Re(FT(dcf))"))
    ax_top.grid(True, alpha=0.3)
    if len(spectra) > 1:
        ax_top.legend(loc="best", fontsize=9)

    # Set limits if defined
    xlim_left = xlim_lower if xlim_lower is not None else XLIM_LOWER
    xlim_right = xlim_upper if xlim_upper is not None else XLIM_UPPER
    if xlim_left is not None:
        ax_top.set_xlim(left=xlim_left)
    if xlim_right is not None:
        ax_top.set_xlim(right=xlim_right)

    # Middle panel: time-domain dcf
    abs_dcf = np.abs(real_dcf + 1j * imag_dcf)
    ax_mid.plot(time_au, real_dcf, lw=1.0, color="b", label="Re dcf")
    ax_mid.plot(time_au, imag_dcf, lw=1.0, color="r", label="Im dcf")
    ax_mid.plot(time_au, abs_dcf, lw=1.0, color="k", label="Abs dcf")
    if envelope_tau_au is not None and envelope_tau_au > 0.0 and abs_dcf.size > 0:
        envelope = abs_dcf[0] * np.exp(-time_au / envelope_tau_au)
        ax_mid.plot(time_au, envelope, lw=1.0, color="0.3", ls="--", label="A * exp(-t/tau)")
    ax_mid.set_xlabel("Time (a.u.)")
    ax_mid.set_ylabel("dcf")
    ax_mid.grid(True, alpha=0.3)
    ax_mid.legend(loc="best", fontsize=9)

    # Bottom panel: normalized absolute dcf on log scale
    abs_dcf_0 = abs_dcf[0] if abs_dcf.size > 0 else 1.0
    if abs_dcf_0 <= 0.0:
        abs_dcf_0 = 1.0
    norm_abs_dcf = abs_dcf / abs_dcf_0
    ax_bottom.plot(time_au, np.maximum(norm_abs_dcf, 1e-18), "k-", lw=1.0, label="Abs dcf / Abs dcf(0)")
    if envelope_tau_au is not None and envelope_tau_au > 0.0 and abs_dcf.size > 0:
        norm_envelope = np.exp(-time_au / envelope_tau_au)
        ax_bottom.plot(
            time_au,
            np.maximum(norm_envelope, 1e-18),
            color="0.3",
            ls="--",
            lw=1.0,
            label="exp(-t/tau)",
        )
    ax_bottom.set_yscale("log")
    ax_bottom.set_xlabel("Time (a.u.)")
    ax_bottom.set_ylabel("Abs dcf / Abs dcf(0)")
    ax_bottom.grid(True, alpha=0.3, which="both")
    ax_bottom.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def plot_spectrum_with_dcf(
    energy_ev,
    real_ft,
    time_au,
    real_dcf,
    imag_dcf,
    out_png,
    xlim_lower=None,
    xlim_upper=None,
    spectrum_prefactor_mode="none",
    envelope_tau_au=None,
):
    """Plot a three-panel figure: Spectrum, time-domain Re/Im dcf, and log-scale |dcf|."""
    plot_spectra_with_dcf(
        energy_ev,
        [("dcf", real_ft)],
        time_au,
        real_dcf,
        imag_dcf,
        out_png,
        xlim_lower=xlim_lower,
        xlim_upper=xlim_upper,
        spectrum_prefactor_mode=spectrum_prefactor_mode,
        envelope_tau_au=envelope_tau_au,
    )

def preprocess_dcf(time_au, real_dcf, imag_dcf, decay_constant_au=None):
    """Apply the configured preprocessing steps to a single dcf series."""
    proc_time_au = np.array(time_au, copy=True)
    proc_real_dcf = np.array(real_dcf, copy=True)
    proc_imag_dcf = np.array(imag_dcf, copy=True)

    if decay_constant_au is not None:
        damping = np.exp(-proc_time_au / decay_constant_au)
        proc_real_dcf = proc_real_dcf * damping
        proc_imag_dcf = proc_imag_dcf * damping

    if APPLY_COSINE_WINDOW:
        proc_time_au, proc_real_dcf, proc_imag_dcf = apply_cosine_window(proc_time_au, proc_real_dcf, proc_imag_dcf)

    if ENABLE_PHASE_UNWRAP:
        phase_energy_shift = SHIFT_CENTER / HARTREE_TO_EV if SHIFT_CENTER != 0.0 else None
        proc_real_dcf, proc_imag_dcf = unwrap_phase(
            proc_time_au, proc_real_dcf, proc_imag_dcf, phase_energy_shift
        )

    return proc_time_au, proc_real_dcf, proc_imag_dcf

def main():
    parser = argparse.ArgumentParser(description="Compute spectrum from dcf data.")
    parser.add_argument("input_file", help="Path to .dat/.txt or .h5/.hdf5 file")
    parser.add_argument("decay_constant_au", nargs="?", type=float, default=None, help="Optional decay constant (a.u.)")
    parser.add_argument(
        "--decay-fs",
        type=float,
        default=None,
        help="Optional decay constant in femtoseconds; converted to a.u. via 1 fs = 41.3413745758 a.u.",
    )
    parser.add_argument("--xlim", nargs=2, type=float, default=None, help="Energy limits in eV for calculation")
    parser.add_argument("--num-pts", type=int, default=None, help="Number of points in the energy grid")
    parser.add_argument(
        "--dcf-only",
        action="store_true",
        help="Only use the primary dcf dataset and ignore optional dcf1/dcf2 datasets",
    )
    parser.add_argument(
        "--phase-ev",
        type=float,
        default=0.0,
        help="Static phase in radians: premultiply dcf by exp(i*phase) before FT",
    )
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument(
        "--scale-omega",
        action="store_true",
        help="Scale Re(FT) by max(E, 0).",
    )
    scale_group.add_argument(
        "--scale-omega-cubed",
        action="store_true",
        help="Scale Re(FT) by max(E, 0)^3.",
    )
    
    args = parser.parse_args()
    if args.scale_omega:
        spectrum_prefactor_mode = "omega"
    elif args.scale_omega_cubed:
        spectrum_prefactor_mode = "omega3"
    else:
        spectrum_prefactor_mode = "none"

    input_path = args.input_file
    if not os.path.isfile(input_path):
        print("Error: file not found: {}".format(input_path))
        sys.exit(1)

    # Optional decay constant (a.u.)
    if args.decay_fs is not None and args.decay_constant_au is not None:
        parser.error("Specify only one decay constant: positional decay_constant_au or --decay-fs.")
    if args.decay_fs is not None:
        if args.decay_fs <= 0.0:
            parser.error("--decay-fs must be > 0.")
        decay_constant_au = float(args.decay_fs) * FS_TO_AU
    else:
        decay_constant_au = args.decay_constant_au
    envelope_tau_au = decay_constant_au if args.decay_fs is not None else None

    # Load data
    time_au, dcf_series = load_dcf_series(input_path, dcf_only=args.dcf_only)

    # Optional exponential damping
    if decay_constant_au is None and TAU_AU is not None:
        decay_constant_au = float(TAU_AU)

    if args.decay_fs is not None:
        print(
            "Applied exponential damping with decay constant: {:.3f} fs ({:.3f} a.u.).".format(
                float(args.decay_fs),
                decay_constant_au,
            )
        )
    elif decay_constant_au is not None:
        print("Applied exponential damping with decay constant: {:.3f} a.u.".format(decay_constant_au))
    if APPLY_COSINE_WINDOW:
        print("Applied cosine windowing")
    if ENABLE_PHASE_UNWRAP:
        phase_energy_shift = SHIFT_CENTER / HARTREE_TO_EV if SHIFT_CENTER != 0.0 else None
        if phase_energy_shift is not None:
            print("Applied phase unwrapping with energy shift: {:.6f} Hartree ({:.3f} eV)".format(
                phase_energy_shift, SHIFT_CENTER))
        else:
            print("Applied phase unwrapping (no energy shift to phase)")
    if args.phase_ev != 0.0:
        print("Applied static phase: exp(i*{:.6f})".format(args.phase_ev))

    # Prepare Energy Grid (eV)
    if args.xlim:
        xlim_lower, xlim_upper = args.xlim
    else:
        # We determine the range to scan based on configuration
        params = {}
        file_ext = os.path.splitext(input_path)[1].lower()
        if file_ext in (".h5", ".hdf5"):
            params = _read_hdf5_parameters(input_path)
        xlim_lower, xlim_upper = _resolve_dynamic_xlims(
            spectrum_axis_mode=params.get("spectrum_axis_mode"),
            spectrum_center_energy_hartree=params.get("spectrum_center_energy_hartree"),
            energy_ref_hartree=params.get("energy_ref_hartree"),
            use_absolute_energies=params.get("use_absolute_energies"),
            energy_1=params.get("energy_1"),
        )
    
    e_min = xlim_lower if xlim_lower is not None else -5.0
    e_max = xlim_upper if xlim_upper is not None else 10.0
    n_pts = args.num_pts if args.num_pts is not None else (int(NUM_POINTS) if NUM_POINTS is not None else 3000)
    
    energy_grid_ev = np.linspace(e_min, e_max, n_pts)

    # Adjust for SHIFT_CENTER
    energy_for_computation = energy_grid_ev - SHIFT_CENTER
    prefactor = _spectrum_prefactor_values(energy_grid_ev, spectrum_prefactor_mode)

    spectra = []
    primary_time_au = None
    primary_real_dcf = None
    primary_imag_dcf = None
    for dataset_name, real_dcf, imag_dcf in dcf_series:
        proc_time_au, proc_real_dcf, proc_imag_dcf = preprocess_dcf(
            time_au, real_dcf, imag_dcf, decay_constant_au=decay_constant_au
        )
        dcf_complex = proc_real_dcf + 1j * proc_imag_dcf
        if args.phase_ev != 0.0:
            dcf_complex = dcf_complex * np.exp(1j * args.phase_ev)

        ft_complex = compute_continuous_ft(proc_time_au, dcf_complex, energy_for_computation)
        spectra.append((dataset_name, np.real(ft_complex) * prefactor))

        if primary_time_au is None:
            primary_time_au = proc_time_au
            primary_real_dcf = proc_real_dcf
            primary_imag_dcf = proc_imag_dcf

    spectra_by_label = {label: real_ft for label, real_ft in spectra}

    # Outputs
    base = os.path.splitext(os.path.basename(input_path))[0]
    mode_suffix = ""
    if spectrum_prefactor_mode == "omega":
        mode_suffix = "_omega"
    elif spectrum_prefactor_mode == "omega3":
        mode_suffix = "_omega3"
    out_dat = f"{base}_ft{mode_suffix}.dat"
    out_png = f"{base}_ft{mode_suffix}.png"

    # Save data (Energy, Re spectra)
    arr = np.column_stack([energy_grid_ev] + [real_ft for _, real_ft in spectra])
    header = "   " + "   ".join(["Energy_eV"] + [f"Re_FT_{label}" for label, _ in spectra])
    np.savetxt(out_dat, arr, header=header)

    # Save per-series data files so dcf1/dcf2 can be consumed directly.
    per_series_dat_paths = []
    for label, real_ft in spectra:
        if label == "dcf":
            continue
        out_dat_series = f"{base}_ft{mode_suffix}_{label}.dat"
        arr_series = np.column_stack([energy_grid_ev, real_ft])
        np.savetxt(out_dat_series, arr_series, header=f"Energy_eV Re_FT_{label}")
        per_series_dat_paths.append(out_dat_series)
    
    # Plot results
    plot_spectra_with_dcf(
        energy_grid_ev,
        spectra,
        primary_time_au,
        primary_real_dcf,
        primary_imag_dcf,
        out_png,
        xlim_lower=xlim_lower,
        xlim_upper=xlim_upper,
        spectrum_prefactor_mode=spectrum_prefactor_mode,
        envelope_tau_au=envelope_tau_au,
    )

    print("Wrote:")
    print("  {}".format(out_dat))
    for out_dat_series in per_series_dat_paths:
        print("  {}".format(out_dat_series))
    print("  {}".format(out_png))

    # Report energy spacing between dcf1 and dcf2 spectral maxima when both exist.
    if "dcf1" in spectra_by_label and "dcf2" in spectra_by_label:
        idx_max_dcf1 = int(np.argmax(spectra_by_label["dcf1"]))
        idx_max_dcf2 = int(np.argmax(spectra_by_label["dcf2"]))
        e_max_dcf1_ev = float(energy_grid_ev[idx_max_dcf1])
        e_max_dcf2_ev = float(energy_grid_ev[idx_max_dcf2])
        delta_e_ev = e_max_dcf1_ev - e_max_dcf2_ev
        print("Maxima energies (eV):")
        print("  dcf1: {:.6f}".format(e_max_dcf1_ev))
        print("  dcf2: {:.6f}".format(e_max_dcf2_ev))
        print("  dcf1 - dcf2: {:.6f} eV".format(delta_e_ev))

if __name__ == '__main__':
    main()

