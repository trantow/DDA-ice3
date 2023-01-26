"""
This script executes the DDA-ice algorithm, to run (in terminal):
    1. set permission:      chmod +x run_dda-ice.sh 
    2. execute:             ./run_dda-ice.sh

Before you run it, make sure to read the README in order to make sure you have set all the 
correct parameters in this script, along with the output location in DDA_main (look for :outdir:)

"""

# Standard parmater combinations
weak_beam_params="-s 4 -u 1 -a 1 -q .6 -k 0.2 -R 5 -r 1 -t neg25 --pix 50,10 -Q 0.5"
simplGreen_params="-s 3 -u 1 -a 1 -q .5 -k 1 -R 5 -r 2 -t neg25 --pix 50,10 -Q 0.5"
double_dens_params="-s 3,5 -u 1,1 -a 1,0.5 -q 0.5,0 -k 1,0.1 -R 5,8 -r 2,1 -t neg25 --pix 50,10 -Q 0.5,0.05"
bif_params="-s 1 -u 1 -a 100 -q .05 -k 1 -R 10 -r 2 -t neg25 --pix 50,10 -m 1 -v 0.2 -z 25"

optimal_ice1_params="-s 3 -u 1 -a 4 -q 0.5 -k 2.25 -R 5 -r 2 --slab-thickness 50 -S 1.75 -Q 0.5"

# Set you runname
runname=a_very_descriptive_runname

# Set input h5 data and choose the channel. Start and end times are optional but encouraged.
#-----
fp=path_to_ATL03.h5_file
ch=gt1l  #channel/beam for the algorithm to look at from the h5 data of the form: 'gt1l' or 'gt1r'
tstart=0  # Use show_atl03.m to get a start time
tend=0  # Use show_atl03.m to get an end time
#-----
params=$optimal_ice1_params

# -p: make plots?, -P: parallelized with multiple processors?
python3 DDA_main.py  -n $runname -c $ch $params -i ATLAS -p T -P T $fp 
# --time-start X --time-start Y --> if subsetting by delta_time
# --track-start X --track-start Y --> if subsetting by along track distance (relative to the raw data)
