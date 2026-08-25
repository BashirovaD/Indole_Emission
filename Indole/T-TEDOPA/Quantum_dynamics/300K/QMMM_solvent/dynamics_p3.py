#!/usr/bin/env python3
import argparse
import os
import sys

from phase123_root import get_mpsdynamics_root

_MPSDYNAMICS_ROOT = get_mpsdynamics_root()
_MPSDYNAMICS_SRC = os.path.join(_MPSDYNAMICS_ROOT, "src")
os.environ["MPSDYN_PATH"] = _MPSDYNAMICS_ROOT
if _MPSDYNAMICS_SRC not in sys.path:
    sys.path.insert(0, _MPSDYNAMICS_SRC)
if _MPSDYNAMICS_ROOT not in sys.path:
    sys.path.insert(0, _MPSDYNAMICS_ROOT)

from dynamics_p12 import PHASE123_PARAMS
from MPSDynamics.workflows.phase123 import run_phase3_from_snapshot


DEFAULT_RUN_TAG = PHASE123_PARAMS.runtime.run_tag
DEFAULT_DT_EMISSION = float(PHASE123_PARAMS.phase3.dt_emission)
DEFAULT_SMART_DCF = float(PHASE123_PARAMS.phase3.smart_dcf)
DEFAULT_SMART_DCF_MAX_TIME = float(PHASE123_PARAMS.phase3.smart_dcf_max_time)
DEFAULT_USE_CPU = None
DEFAULT_AUTO_POSTPROCESS = True
DEFAULT_SAVE_DIR = "."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phase 3 emission from a saved phase12 snapshot")
    parser.add_argument("--omega", type=float, required=True, help="omega (eV) used to locate the run directory")
    parser.add_argument("--t", dest="time_dir", required=True, help="snapshot time directory name under phase12/intermediate_saves")
    parser.add_argument("--dt_emission", type=float, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--smart_dcf", type=float, default=None, help="Smart DCF threshold. Set to 0 to force fixed-time propagation.")
    parser.add_argument("--smart_dcf_max_time", type=float, default=None)
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument("--cpu", dest="use_cpu", action="store_true", help="Force the phase-3 workflow to use the CPU backend")
    backend_group.add_argument("--gpu", dest="use_cpu", action="store_false", help="Force the phase-3 workflow to use the GPU backend")
    parser.set_defaults(use_cpu=DEFAULT_USE_CPU)
    args = parser.parse_args()

    run_phase3_from_snapshot(
        omega=args.omega,
        time_dir=args.time_dir,
        dt_emission=DEFAULT_DT_EMISSION if args.dt_emission is None else args.dt_emission,
        run_tag=DEFAULT_RUN_TAG if args.run_tag is None else args.run_tag,
        smart_dcf=DEFAULT_SMART_DCF if args.smart_dcf is None else args.smart_dcf,
        smart_dcf_max_time=(DEFAULT_SMART_DCF_MAX_TIME if args.smart_dcf_max_time is None else args.smart_dcf_max_time),
        savedir=DEFAULT_SAVE_DIR,
        use_cpu=args.use_cpu,
        auto_postprocess_after_runsim=DEFAULT_AUTO_POSTPROCESS,
    )


if __name__ == "__main__":
    main()
