#!/usr/bin/python3
import os
import sys

_MPSDYNAMICS_ROOT = "/shared/zuehlsdorff/research_software/PyMPSDynamics_4626"
_MPSDYNAMICS_SRC = os.path.join(_MPSDYNAMICS_ROOT, "src")
os.environ["MPSDYN_PATH"] = _MPSDYNAMICS_ROOT
if _MPSDYNAMICS_SRC not in sys.path:
    sys.path.insert(0, _MPSDYNAMICS_SRC)
if _MPSDYNAMICS_ROOT not in sys.path:
    sys.path.insert(0, _MPSDYNAMICS_ROOT)

from MPSDynamics.workflows.singleshot import SingleshotConfig, run_singleshot


def main():
    use_cpu = False
    run_name = "example"

    compute_emission = True
    write_omega_prefactor_spectra = True
    imag_init_for_emission = True
    use_drive = False

    restart_dir = None
    restart_emission = False  # True skips absorption

    abs_save_window_start = None
    abs_save_window_end = None
    abs_save_stride = None

    time_step = 10.0

    energy_1 = 0.18160335982547376
    energy_2 = 0.1908
    coupling = 0.0008322123718324007
    mu_01 = 0.4331887740811826
    mu_02 = 0.871799673071426

    # For adaptive stopping based on |dcf| < threshold * |dcf(t=0)|, use a dict:
    # total_time_absorption = {"smart_dcf": 4e-3, "max_time": 1000.0}
    # total_time_emission = {"smart_dcf": 4e-3, "max_time": 1000.0}
    total_time_absorption = 10.0
    total_time_emission = 10000.0
    temp = 0

    chain_length_1 = 150
    chain_length_2 = 150
    chain_length_12 = 150
    num_fock_1_high = [30]
    num_fock_1_low = 30
    num_fock_2_high = [30]
    num_fock_2_low = 30
    num_fock_12_high = [30]
    num_fock_12_low = 30

    bond_dim_1_large = 80
    bond_dim_1_small = 20
    bond_sigmoid_center_1 = 0.0
    bond_sigmoid_sharpness_1 = 6.0
    bond_dim_2_large = 80
    bond_dim_2_small = 20
    bond_sigmoid_center_2 = 0.0
    bond_sigmoid_sharpness_2 = 6.0
    bond_dim_12_large = 80
    bond_dim_12_small = 20
    bond_sigmoid_center_12 = 0.0
    bond_sigmoid_sharpness_12 = 6.0

    imag_time_step = 25
    max_imag_time = 100000

    # E_amp * cos(omega_carrier * (t - t_pulse_center))
    # * exp(-4 ln 2 * ((t - t_pulse_center)/t_pulse_duration)^2)
    E_amp = 0.1
    omega_carrier = 3.05 / 27.2114
    t_pulse_center = 30
    t_pulse_duration = 10

    savedir = os.getcwd()
    coeffsdir = os.path.join(savedir, "chain_coeffs.hdf5")

    auto_postprocess_after_runsim = True
    compute_singular_val = True
    singular_val_print_steps = 15

    config = SingleshotConfig(
        use_cpu=use_cpu,
        run_name=run_name,
        auto_postprocess_after_runsim=auto_postprocess_after_runsim,
        compute_singular_val=compute_singular_val,
        singular_val_print_steps=singular_val_print_steps,
        compute_emission=compute_emission,
        write_omega_prefactor_spectra=write_omega_prefactor_spectra,
        imag_init_for_emission=imag_init_for_emission,
        use_drive=use_drive,
        time_step=time_step,
        energy_1=energy_1,
        energy_2=energy_2,
        coupling=coupling,
        mu_01=mu_01,
        mu_02=mu_02,
        total_time_absorption=total_time_absorption,
        total_time_emission=total_time_emission,
        temp=temp,
        chain_length_1=chain_length_1,
        chain_length_2=chain_length_2,
        chain_length_12=chain_length_12,
        num_fock_1_high=num_fock_1_high,
        num_fock_1_low=num_fock_1_low,
        num_fock_2_high=num_fock_2_high,
        num_fock_2_low=num_fock_2_low,
        num_fock_12_high=num_fock_12_high,
        num_fock_12_low=num_fock_12_low,
        bond_dim_1_large=bond_dim_1_large,
        bond_dim_1_small=bond_dim_1_small,
        bond_sigmoid_center_1=bond_sigmoid_center_1,
        bond_sigmoid_sharpness_1=bond_sigmoid_sharpness_1,
        bond_dim_2_large=bond_dim_2_large,
        bond_dim_2_small=bond_dim_2_small,
        bond_sigmoid_center_2=bond_sigmoid_center_2,
        bond_sigmoid_sharpness_2=bond_sigmoid_sharpness_2,
        bond_dim_12_large=bond_dim_12_large,
        bond_dim_12_small=bond_dim_12_small,
        bond_sigmoid_center_12=bond_sigmoid_center_12,
        bond_sigmoid_sharpness_12=bond_sigmoid_sharpness_12,
        imag_time_step=imag_time_step,
        max_imag_time=max_imag_time,
        E_amp=E_amp,
        omega_carrier=omega_carrier,
        t_pulse_center=t_pulse_center,
        t_pulse_duration=t_pulse_duration,
        savedir=savedir,
        coeffsdir=coeffsdir,
        abs_save_window_start=abs_save_window_start,
        abs_save_window_end=abs_save_window_end,
        abs_save_stride=abs_save_stride,
        restart_dir=restart_dir,
        restart_emission=restart_emission,
    )

    run_singleshot(config)


if __name__ == "__main__":
    main()
