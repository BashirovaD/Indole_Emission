#!/usr/bin/env python3
import argparse
import copy
import os
import sys
from pathlib import Path

_MPSDYNAMICS_ROOT = "/shared/zuehlsdorff/research_software/PyMPSDynamics_4626"
_MPSDYNAMICS_SRC = os.path.join(_MPSDYNAMICS_ROOT, "src")
os.environ["MPSDYN_PATH"] = _MPSDYNAMICS_ROOT
if _MPSDYNAMICS_SRC not in sys.path:
    sys.path.insert(0, _MPSDYNAMICS_SRC)
if _MPSDYNAMICS_ROOT not in sys.path:
    sys.path.insert(0, _MPSDYNAMICS_ROOT)

LOCAL_CHAIN_COEFFS = (Path(__file__).resolve().parent.parent / "data" / "chain_coeffs.hdf5").resolve()

from MPSDynamics.workflows.phase123.config import (
    BondTaperConfig, ChainConfig, Phase1Config, Phase2Config,
    Phase3Config, Phase123Config, RuntimeConfig, SystemConfig,
)


PHASE123_PARAMS = Phase123Config(
    system=SystemConfig(
        energy_1=0.17596,
        energy_2=0.1748513995307165,
        coupling=0.0011019187119606136,
        mu_01=0.392459421515794,
        mu_02=0.85816551848249,
        temp=300.0,
    ),
    chain=ChainConfig(
        chain_length_1=250,
        chain_length_2=250,
        chain_length_12=250,
        num_fock_1_high=[50],
        num_fock_1_low=50,
        num_fock_2_high=[100,150,140,120,120,120,120,100,100,80,80],
        num_fock_2_low=50,
        num_fock_12_high=[30],
        num_fock_12_low=30,
    ),
    bond_taper=BondTaperConfig(
        bond_dim_1_large=80,
        bond_dim_1_small=40,
        bond_sigmoid_center_1=0.0,
        bond_sigmoid_sharpness_1=6.0,
        bond_dim_2_large=80,
        bond_dim_2_small=50,
        bond_sigmoid_center_2=0.0,
        bond_sigmoid_sharpness_2=6.0,
        bond_dim_12_large=80,
        bond_dim_12_small=30,
        bond_sigmoid_center_12=0.0,
        bond_sigmoid_sharpness_12=6.0,
    ),
    phase1=Phase1Config(
        enabled=False,
        dt=2.0,
        t_end=10.0,
        E_amp=0.01,
        t_pulse_center=1000.0,
        t_pulse_duration=400.0,
    ),
    phase2=Phase2Config(
        dt=10.0,
        t_end=15000.0,
        save_window_start=0.0,
        save_window_end=15000.0,
        save_stride=250.0,
        submit_window_start=5000.0,
        submit_window_end=15000.0,
        submit_stride=1000.0,
    ),
    phase3=Phase3Config(
        dt_emission=10.0,
        smart_dcf=0.0,
        smart_dcf_max_time=4200.0,
    ),
    runtime=RuntimeConfig(
        run_tag="output",
        omegas_ev=[3.0],
        use_cpu=False,
        savedir=".",
        coeffs_h5=str(LOCAL_CHAIN_COEFFS),
        auto_postprocess_after_runsim=True,
        compute_singular_val=True,
        singular_val_print_steps=2,
        write_omega_prefactor_spectra=False,
    ),
)


def print_phase3_defaults(cfg: Phase123Config) -> None:
    phase3 = cfg.phase3
    print(f"dt_emission={float(phase3.dt_emission)!r}")
    print(f"smart_dcf={float(phase3.smart_dcf)!r}")
    print(f"smart_dcf_max_time={float(phase3.smart_dcf_max_time)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phase 1/2 dynamics")
    parser.add_argument("--omega", type=float, default=None, help="Run a single omega (eV) in this job")
    parser.add_argument("--run_tag", type=str, default=None, help="Override run tag prefix for output directories")
    parser.add_argument("--cpu", action="store_true", help="Force the phase 1/2 workflow to use the CPU backend")
    parser.add_argument("--use-driving", action="store_true", default=None, help="Use driving field (phase 1) instead of dipole-weighted initialization")
    parser.add_argument("--print-phase3-defaults", action="store_true", help="Print Phase 3 defaults as key=value lines and exit")
    parser.add_argument("--queue-p3", action="store_true", help="Queue phase 3 snapshots as each checkpoint is written")
    parser.add_argument("--queue-p3-partition", type=str, default=None, help="SLURM partition token forwarded to submit_p3.sh when --queue-p3 is enabled")
    parser.add_argument("--queue-p3-submit-script", type=str, default=None, help="Absolute path to the submit_p3.sh script that should be used for on-the-fly phase 3 queueing")
    parser.add_argument("--queue-p3-arg", dest="queue_p3_args", action="append", default=None, help="Repeat once per raw argument forwarded to submit_p3.sh")
    args = parser.parse_args()

    cfg = copy.deepcopy(PHASE123_PARAMS)
    if args.cpu:
        cfg.runtime.use_cpu = True
    if args.print_phase3_defaults:
        print_phase3_defaults(cfg)
        return

    from MPSDynamics.workflows.phase123 import run_phase12
    from MPSDynamics.workflows.phase123.p3_queue import Phase3QueueConfig

    resolved_run_tag = cfg.runtime.run_tag if args.run_tag is None else args.run_tag
    queue_phase3 = None
    if args.queue_p3:
        if not args.queue_p3_partition:
            parser.error("--queue-p3 requires --queue-p3-partition.")
        queue_phase3 = Phase3QueueConfig(
            partition_token=args.queue_p3_partition,
            run_tag=resolved_run_tag,
            passthrough_args=tuple(args.queue_p3_args or ()),
            submit_script=args.queue_p3_submit_script,
        )
    run_phase12(
        cfg,
        omega=args.omega,
        run_tag=args.run_tag,
        use_driving=args.use_driving,
        queue_phase3=queue_phase3,
    )


if __name__ == "__main__":
    main()
