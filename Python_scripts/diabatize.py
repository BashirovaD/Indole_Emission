#! /usr/bin/env python

# A script to take adiabtic TDDFT data and find the diabatic coupling and 
# states.
# Run with the data file names as command line arguments:
#   Each file can contain the data for one or more excitations as additional 
#   columns of data, so 5 * n_states columns.

import sys
from numpy.linalg import eigh as solve
from math import copysign

# Some math
eV_to_ha = 0.03674932217565437
def svprod(scalar, vector):
    return [scalar * v for v in vector]
def vsum(*args):
    return [sum(elements) for elements in zip(*args)]
def dot(vec1, vec2): return sum(v1 * v2 for v1, v2 in zip(vec1, vec2)) 
def mtrans(m):
    return [
        [m[icol][irow] for icol in range(len(m[irow]))] 
        for irow in range(len(m))
    ]
def mprod(m1, m2):
    if len(m1[0]) == len(m2):
        return [
            [
                dot(m1_row, [m2_row[im2_col] for m2_row in m2]) 
                for im2_col in range(len(m2[0]))
            ] 
            for m1_row in m1
        ]

# Read a space or tab separated file
def read_dat(file_name):
    return [
        list(map(float, line.strip().split())) 
        for line in open(file_name) 
        if len(line.strip()) > 0 and line.strip()[0].isnumeric()
    ]

def subotnik_diabatization(adiabatic_states):
    # Use the dipole overlaps to find orthogonal diabatic dipoles
    overlap_matrix = [
        [
            dot(adiabatic_states[2:5], adiabatic_states[2:5]), 
            dot(adiabatic_states[2:5], adiabatic_states[7:10])
        ], 
        [0, dot(adiabatic_states[7:10], adiabatic_states[7:10])]
    ]
    overlap_matrix[1][0] = overlap_matrix[0][1]
    # Solve for rotation matix
    diabatic_overlaps, rotation_matrix = solve(overlap_matrix)
    # Calculate diabatic oscillator strengths
    diabatic_oscillator_strengths = [
        float(diabatic_overlaps[0]) * adiabatic_states[0] * eV_to_ha * 2 / 3,
        float(diabatic_overlaps[1]) * adiabatic_states[5] * eV_to_ha * 2 / 3
    ]
    # Rotate excitation energies
    energy_matrix = [
        [adiabatic_states[0], 0], 
        [0, adiabatic_states[5]]
    ]
    diabatic_energies = mprod(
        mtrans(rotation_matrix), 
        mprod(energy_matrix, rotation_matrix)
    )
    # Assign S1 & S2 based on oscillator strength only
    if diabatic_oscillator_strengths[0] < diabatic_oscillator_strengths[1]:
        diabatic_oscillator_strengths = diabatic_oscillator_strengths[::-1]
        diabatic_energies[0][0], diabatic_energies[1][1] = (
            diabatic_energies[1][1], diabatic_energies[0][0].copy())
        diabatic_energies[0][1] = diabatic_energies[1][0] = (
            diabatic_energies[0][1] * -1
        )

    return [
        # diabatic S1
        float(diabatic_energies[0][0]), diabatic_oscillator_strengths[0], 
        # diabatic S2
        float(diabatic_energies[1][1]), diabatic_oscillator_strengths[1], 
        # diabatic S1-S2 coupling
        float(diabatic_energies[0][1])
    ]


def n_state_subotnik_diabatization(ad_states):
    # Use the dipole overlaps to find orthogonal diabatic dipoles
    num_states = len(ad_states)
    ad_overlap_matrix = [
        [dot(state1[2:5], state2[2:5]) for state2 in ad_states] 
        for state1 in ad_states
    ]
    # Solve for rotation matix
    d_overlaps, transformation_matrix = solve(ad_overlap_matrix)
    # Rotate excitation energies
    ad_energy_matrix = [[0] * num_states for _ in range(num_states)]
    for istate in range(num_states):
        ad_energy_matrix[istate][istate] = ad_states[istate][0]
    d_energies = mprod(
        mtrans(transformation_matrix), 
        mprod(ad_energy_matrix, transformation_matrix)
    )
    # Calculate diabatic oscillator strengths
    d_oscillator_strengths = [
        # Why `abs`? For tiny dipole moments, the eigensolver might return a 
        # negative number, which is unphysical, and will cause problems later.
        # `abs` might not be the most correct way to fix this though.
        abs(float(d_overlaps[istate])) * d_energies[istate][istate] 
        * eV_to_ha * 2 / 3  
        for istate in range(num_states)
    ]
    # Calculate the diabatic transition dipole moments
    ad_dipoles = [ad_state[2:5] for ad_state in ad_states]
    d_dipoles = [[]] * num_states
    for istate in range(num_states):
        d_dipoles[istate] = vsum(*[
            svprod(coef ** 2, dipole) for dipole, coef 
            in zip(ad_dipoles, transformation_matrix[:, istate]) 
        ])

    d_data_out = [
        float(d_energies[istate1][istate2]) 
        for istate1 in range(num_states) 
        for istate2 in range(num_states) 
        if istate2 >= istate1
    ] + d_oscillator_strengths
    return d_data_out, d_dipoles

def fix_signs_quad(coupling : list[float]) -> list[float]:
    '''Change the signs of a couplings to remove discontinuities

    Quadratic regression and extrapolation to remove incorrect sign changes in 
    a list of floats that are assumed to be a time ordered sequence fluctuating 
    smoothly.
    '''
    # Regression the first time
    first_der = [coupling[1] - coupling[0], coupling[2] - coupling[1]]
    extrapo_val = coupling[2] + 2 * first_der[1] - first_der[0]
    # Loop over all of them
    for icoup in range(3,len(coupling)):
        if coupling[icoup] != copysign(coupling[icoup], extrapo_val):
             coupling[icoup] = -1.0 * coupling[icoup]
        first_der = [first_der[1], coupling[icoup] - coupling[icoup - 1]] 
        extrapo_val = coupling[icoup] + 2 * first_der[1] - first_der[0] 
    return coupling

def fix_signs_linear(coupling : list[float]) -> list[float]:
    '''Change the signs of a couplings to remove discontinuities

    Linear regression and extrapolation to remove incorrect sign changes in 
    a list of floats that are assumed to be a time ordered sequence fluctuating 
    smoothly.
    '''
    # Regression the first time
    extrapo_val = 2 * coupling[1] - coupling[0]
    # Loop over all of them
    for icoup in range(2,len(coupling)):
        if coupling[icoup] != copysign(coupling[icoup], extrapo_val):
             coupling[icoup] = -1.0 * coupling[icoup]
        extrapo_val = 2 * coupling[icoup] - coupling[icoup - 1] 
    return coupling

def check_dipoles(data, dipoles):
    # Find the coupling columns
    num_states = len(dipoles[0])
    all_columns = set(range(num_states))
    energy_gap_columns = set([ 
        int(state * num_states - state * (state - 1) / 2)  
        for state in range(num_states)
    ])
    oscillator_strength_columns = set(range(
        len(data[0]) - num_states, 
        len(data[0])
    ))
    # A list of the indices of `data` that are columns of coupling values
    coupling_columns = all_columns - (
        energy_gap_columns | oscillator_strength_columns
    )
    # A list of lists of which states correspond to each column of couplings
    coupling_states = [ 
        [num_states - y + 1, num_states - z + 1] 
        for y in range(num_states + 1, 1, -1) for z in range(y - 1, 1, -1) 
    ]

    # Check for bad sign flips: keep it simple, the diabatic dipoles should be 
    # boring?
    for istate in range(num_states):
        # A list of the indices of the couplings for this state
        couplings = [
            coupling for icoupling, coupling in enumerate(coupling_columns) 
            if istate in coupling_states[icoupling]
        ]
        for idipole in range(1, len(dipoles[istate])):
            if dot(dipoles[istate][idipole], dipoles[istate][idipole - 1]) < 0:
                dipoles[istate][idipole] = svprod(-1, dipoles[istate][idipole])
                for coupling in couplings:
                    data[idipole][coupling] *= -1
    # It needs something more complex I think: for each diabatic dipole, find 
    # the adiabatic dipole that is most similar to it, perhaps by balancing 
    # magnitude and direction using the distance between them. Then if the 
    # adiabatic dipole is facing the oposite direction, flip it and change the 
    # sign of the coupling?
    return data

def write_dat(data, file_name):
    file_root = file_name.split('.')[0]
    output_file = open(f'diabatic_{file_root}.dat', 'w')
    output_file.write('\n'.join([
        ('{:19.12e} ' * len(line)).strip().format(*line) 
        for line in data
    ]))
    

def main(cmd_line_args):
    # Read in adiabatic data: 
    # A list of (states) a list of (snapshots) a list of (energy, oscillator 
    # strength, dipole X, dipole Y, dipole Z) floats
    cmd_line_args.pop(0)
    adiabatic_data = []
    for file_name in cmd_line_args:
        adiabatic_data.append(read_dat(file_name))
        while len(adiabatic_data[-1][0]) > 5:
            adiabatic_data.extend([
                [line[0:5], line[5:]] for line in adiabatic_data.pop()
            ])
    # Do the rotation thing
    num_lines = len(adiabatic_data[0])
    diabatic_data = [[]] * num_lines
    d_dipoles = [[]] * num_lines
    for iline in range(num_lines):
        diabatic_data[iline], d_dipoles[iline] = n_state_subotnik_diabatization(
            [state[iline] for state in adiabatic_data]
        )
    # Check for bad sign flips in the couplings 
    diabatic_data = check_dipoles(diabatic_data, d_dipoles)

    # Write the output
    write_dat(diabatic_data, cmd_line_args[-1])
    d_dipoles = list(zip(*d_dipoles))
    for istate in range(len(d_dipoles)):
        write_dat(d_dipoles[istate], f'dipoles_S{istate + 1}.dat')

main(sys.argv)

# Swapping S1 and S2 before diabatizing: same diabatic states, but the coupling changes sign
# Changing the sign of one of the states: same diabatic states, but the coupling changes sign
# Proposed algorithm:
#   1 Diabatize the data
#   2 Sort the crossings of the diabatic data
#     2.1 Sort by regression of oscillator strength?
#       2.1.1 If the diabatization is good, the oscillator strengths mmight never
#         cross
#   3 Sort the adiabatic data using the diabatic data and smooth the dipoles again
#   4 Diabatize the adiabatic data again
# Alternative:
#   1 Diabatize the data
#   2 Use regression to correct the sign of the coupling in an arbitrary way
# Alternative 2:
#   1 Diabatize the data
#   2 Hope that the states can be separated by their oscillator strength
#   3 Calculate the dipole moments of the diabatic states
#   4 Use the dipoles to correct the sign of the couplings



