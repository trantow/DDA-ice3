# DDA-ice3

Density Dimension Algorithm for Ice Surfaces (python3)

## Introduction and References

The Density Dimension Algorithm for Ice (DDA-Ice) is designed to detect cryospheric signals in the photon data provided NASA's ICESat-2 satellite. The algorithm and its applications are described in the following manuscripts:

(1) Herzfeld, U. C., Trantow, T. M., Harding, D., & Dabney, P. W. (2017). Surface-height determination of crevassed glaciers—Mathematical principles of an autoadaptive density-dimension algorithm and validation using ICESat-2 simulator (SIMPL) data. IEEE Transactions on Geoscience and Remote Sensing, 55(4), 1874-1896.

(2) Herzfeld, U. C., Trantow, T., Lawson, M., Hans, J., & Medley, G. (2021). Surface heights and crevasse morphologies of surging and fast-moving glaciers from ICESat-2 laser altimeter data-Application of the density-dimension algorithm (DDA-ice) and evaluation using airborne altimeter and Planet SkySat data. Science of Remote Sensing, 3, 100013.


## Dependencies

The packages listed below are the only ones that will need to be installed in your python environment. The `scikit-learn` dependencies should cover all other necessary packages.

```diff
+ Python 3
+ h5py
+ json
+ scipy
+ numpy
+ shapely
+ matplotlib
+ optparse
+ utm
+ sklearn
```
> **Note:** Sometimes you will get a runtime error (OSError: Could not find lib geos_c.dll or load any of its variants []) when shapely is installed with conda. To fix it, you must 1) remove the conda version of shapely (conda uninstall shapely), 2) install the conda version of "geos" (conda install geos), 3) install shapely via pip (pip install shapely). You should not get this error anymore.

## Code

The DDA-ice code is built using python-3 and is most effeciently run using a bash script. The code in this repository is as follows:

**`DDA_main.py`**
This is the main python file for the DDA-ice code. It first parses the command line arguments into readable parameters and formats the output directories. This file also produces all logging information throughout execution of the code, and saves output data at specific locations.
<br></br>

**`DDA_func.py`**
This file defines all the *computation* functions for DDA-ice. They are functions like `load_photon_data`, `compute_density`, and `interpolate_ground`.
<br></br>

**`DDA_vis.py`**
This file contains all the *plotting* functions for DDA-ice. They are functions like `plot_raw_data`, `plot_density`, and `plot_ground_estimate`.
<br></br>

**`run_dda-ice.sh`**
This is the bash script which sets input parameters for the python code and calls the python code with these parameters passed in as command line arguments. Three standard parameter sets (`weak_beam_params`, `simplGreen_params`, and `double_dens_params`) are all included at the top of the shell script for convenience. Some of the parameters will still need to be changed regularly: `fp`, `ch`, `runname`, `tstart`, and `tend`. 
<br></br>

## Parameters
| **Symbol** | **Alias: Meaning** | **SIMPL green** | **Double-Density** | **Weak-Beam** | **Bifurcate** |
| --- | --- | :---: | :---: | :---: | :---:| 
| **`s`** | sigma: standard deviation for the gaussian RBF | 3 | 3,5 | 4 |-------------|
| **`u`** | cutoff: # of standard deviations before RBF is cutoff | 1 | 1,1 | 1 | ------------- |
| **`a`** | aniso: anisotropy factor for the RBF | 1 | 1,0.5 | 1 | ------------- |
| **`q`** | quantile: Quantile for threshold function | 0.5 | 0.5,0 | 0.6 | ------------- |
| **`k`** | offset: offset for thresholding before taking the quantile | 1 | 1,0.1 | 1 | ------------- |
| **`l`** | slab-thickness: slab thickness for defining signal and noise slab | 200 | 200 | 200 | ------------- |
| **`R`** | interp-res: initial resolution of interpolated surface | 5 | 5,8 | 5 | ------------- |
| **`r`** | interp-factor: Reduction factor for interpolation in certain areas | 2 | 2,1 | 2 | ------------- |
| **`Q`** | crev-depth-quantile: determines how deep a crevasse is from it's deepest measured point | 0.5 | 0.5,0.05 | 0.5 | ?|
| **`--pix`** | pixel resoluion for slab detection (xres,yres) | 50,10 | 50,10 | 50,10 | 50,10 |
| **`S`** | std-dev: standard deviation of thresholded signal to trigger interpolation reduction | 1.75 | 1.75 | 1.75 | 1.75 |
| **`p`** | plot: Boolean for plotting | --------- | ----------- | -------- | ------------- |
| **`P`** | parallel: Boolean for using parallel computing options |--------- | ----------- | -------- | ------------- |
| **`b`** | binsize: width (m) of the bins for calculated thresholds | 5 | 5 | 5 | 5 |
| **`i`** | instrument: Instrument that collected the data | i | i | i | i |
| **`c`** | channel: instrument channel | --------- | -----------| -------- | ------------- |
| **`n`** | name: the name of the run used for storing the output | --------- | ----------- | -------- | ------------- |
| **`t`** | cloud-tolerance: tolerance for the maximum curvature that will be accepted as ground (Not used in assign slab by histogram) | neg25 | neg25 | neg25 | neg25 |
| **`m`** | meltpond-bool: Boolean for melt pond run | False | False | False | True |
| **`d`** | melt-pond-quantile: determines how deep a melt pond is from its deepest measured point | --------- | ----------- | --------| 0.75 |
| **`w`** | density-weight: density weighting via power law for ground follower in DDA-bif | --------- | ----------- |-------- | ? |
| **`v`** | vbin: vertical bin size for DDA-bif | --------- | ----------- | -------- | 0.2 |
| **`z`** | hbin: horizontal bin size for DDA-bif | --------- | ----------- | -------- | 25 |
| **`time-start`** | time at which to start subsetting | --------- | -----------| -------- | ------------- |
| **`time-end`** | time at which to stop subsetting | --------- | ----------- | -------- | ------------- |
| **`track-start`** | along track distance at which to start subsetting | --------- | ----------- | -------- | ------------- |
| **`track-end`** | along track distance at which to stop subsetting | --------- | ----------- | -------- | ------------- |


## Input Data

The DDA-ice takes the ATL03 Global Geolocated Photon Data from ICESat-2 as input, which is in HDF5 format. Data can be downloaded after making a free account at https://www.earthdata.nasa.gov/. It is best to downsample the data using the variables **`time-start`**, **`time-end`**, **`track-start`** and/or **`track-end`** in order for run times to be reasonable.

## Output
DDA-ice-py3 will create a folder in the current directory called `output/<runname>/` which will contain everything generated during runtime (including the log file). `output/<runname>/plots` will contain all plots generated during runtime.
