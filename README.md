This repository contains computational supporting information for the manuscript "Computational Elucidation of System-Dependent Excited-State Relaxation Mechanisms and Stokes Shifts in Indole-Based Chromophores" by D. Bashirova, E. Lambertson, B. Dawson, and T. J. Zuehlsdorff.

```bash
.
├── 2CNI
│   ├── MD # contains topology and parameter files, as well as excitation energies along the QM/MM simulation 
│   │   ├── 2cni_opt.xyz
│   │   ├── MM
│   │   ├── QM:MM
│   │   └── TDDFT
│   └── T-TEDOPA 
│       ├── Adiabaitc_data # contains adiabatic energies and dipole moments
│       ├── Diabatic_data # contains diabatic energies, dipole moments, and couplings
│       ├── Quantum_dynamics # contains the input file, chain coefficients, resulting optical spectra, and population dynamics
│       └── SD # contains diabatic spectral densities
├── Indole
│   ├── MD # contains topology and parameter files, as well as excitation energies along the QM/MM simulation
│   │   ├── indole_opt.xyz
│   │   ├── MM
│   │   ├── QM:MM
│   │   └── TDDFT
│   └── T-TEDOPA
│       ├── Adiabaitc_data # contains adiabatic energies and dipole moments
│       ├── Diabatic_data # contains diabatic energies, dipole moments, and couplings
│       ├── Quantum_dynamics # contains the input file, chain coefficients, resulting optical spectra, and population dynamics
│       └── SD
├── Python_scripts
│   ├── compute_sd.py
│   ├── diabatize.py
│   └── spectral_dens_to_chain.py
└── Trp
    ├── MD # contains topology and parameter files, as well as excitation energies along the QM/MM simulation
    │   ├── MM
    │   ├── QM:MM
    │   ├── TDDFT
    │   └── trp_gs_opt.xyz
    └── T-TEDOPA
        ├── Adiabaitc_data # contains adiabatic energies and dipole moments
        ├── Diabatic_data # contains diabatic energies, dipole moments, and couplings
        ├── Quantum_dynamics # contains the input file, chain coefficients, resulting optical spectra, and population dynamics
        └── SD
