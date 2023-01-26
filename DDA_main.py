from DDA_func import *
from DDA_vis import *
from optparse import OptionParser
import multiprocessing
import threading
import logging
import time
import os
import gc


def main():
	'''
	Set up the command line parser to take flagged options like
	python surface.py [options] args
	'''

	####################################################################################################
	####################################################################################################
	############################                                    ####################################
	############################               SETUP                ####################################
	############################                                    ####################################
	####################################################################################################
	####################################################################################################

	usage = "usage: %prog [options] filepath"
	cmd_parser = OptionParser(usage=usage)
	cmd_parser.add_option(
		'-n', '--name', dest='name',
		default=None,
		help='Name of the run used for storing output')
	cmd_parser.add_option(
		'-c', '--channel', dest='channel',
		default='2',
		help='Channel number')
	cmd_parser.add_option(
		'-s', '--sigma', dest='sigma',
		default='3',
		help='Sigma, the standard deviation for the Gaussian RBF')
	cmd_parser.add_option(
		'-u', '--cutoff', dest='cutoff',
		default='1',
		help='Cutoff for the RBF (# of standard deviations)')
	cmd_parser.add_option(
		'-a', '--aniso', dest='aniso',
		default='5',
		help='Anisotropy factor for the RBF')
	cmd_parser.add_option(
		'-i', '--instrument', dest='instrument',
		default='ATLAS',
		help='Instrument from which the data was generated')
	cmd_parser.add_option(
		'-b', '--binsize', dest='binsize',
		default='5',
		help='Size (width) of the bins for calculated thresholds (m)')
	cmd_parser.add_option(
		'-q', '--quantile', dest='quantile',
		default='.75',
		help='Quantile for threshold calculation')
	cmd_parser.add_option(
		'-k', '--offset', dest='threshold_offset',
		default='0',
		help='Offset for thresholding before taking the quantile')
	cmd_parser.add_option(
		'-p', '--plot', dest='plot_TF',
		default='False',
		help='Boolean for plotting, True or False')
	cmd_parser.add_option(
		'--track-start', dest='track_start',
		default='0',
		help='Along track distance to start from')
	cmd_parser.add_option(
		'--track-end', dest='track_end',
		default='1E99',
		help='Along track distance at which to stop')
	cmd_parser.add_option(
		'--time-start', dest='time_start',
		default='0',
		help='Time at which to start')
	cmd_parser.add_option(
		'--time-end', dest='time_end',
		default='1E99',
		help='Time at which to stop')
	cmd_parser.add_option(
		'--pix', dest='pixel_dimensions',
		default='50,10',
		help='Pixel resolution for slab detection input as xres,yres.')
	cmd_parser.add_option(
		'-R', '--interp-res', dest='interp_res',
		default='5',
		help='The initial resolution of the interpolated surface. Gets reduced by a factor of interp_factor for crevassed surfaces.')
	cmd_parser.add_option(
		'-r', '--interp-factor', dest='interp_factor',
		default='5',
		help='Reduction factor for interpolation in variable areas (crevassed areas).')
	cmd_parser.add_option(
		'-t', '--cloud-tolerance', dest='cloud_tolerance',
		default='neg30',
		help='Tolerance for the maximum curvature (least negative) that will be accepted as ground signal. For negative values use neg5 rather than -5.')
	cmd_parser.add_option(
		'-l', '--slab-thickness', dest='slab_thickness',
		default='200',
		help='Slab thickness (or height) for defining noise and signal slabs')
	cmd_parser.add_option(
		'-S', '--std-dev', dest='std_dev',
		default='1.75',
		help='Standard deviation threshold of thresholded signal to trigger small step size in ground follower (meters)')
	cmd_parser.add_option(
		'-Q', '--crev-depth-quantile', dest='crev_depth_quantile',
		default='0.5',
		help='Quantile controlling depth of crevasse, Q=0 corresponds to crevasse depth equal to the lowest photon in thresholded signal in second pass')
	cmd_parser.add_option(
		'-m', '--meltpond-bool', dest='meltpond_bool',
		default='False',
		help='Boolean for DDA-Bifurcate (melt pond) run (default: false)')
	cmd_parser.add_option(
		'-z', '--hbin', dest='mp_bin_h',
		default='25',
		help='Horizontal Bin Size for DDA-Bifurcate Option (meltponds)')
	cmd_parser.add_option(
		'-v', '--vbin', dest='mp_bin_v',
		default='0.2',
		help='Vertical Bin Size for DDA-Bifurcate Option (meltponds)')
	cmd_parser.add_option(
		'-w', '--density-weight', dest='den_weight',
		default='1',
		help='Density weighting, via a power law, for Ground Follower in the DDA-Bifurcate Option (meltponds, bottom only)')
	cmd_parser.add_option(
		'-d', '--melt-pond-quantile', dest='mp_quantile',
		default='0.75',
		help='Elevation (or depth) quantile for melt pond bottoms (The higher the value, the shallower the melt pond)')
	cmd_parser.add_option(
		'-P', '--parallel', dest='isParallel',
		default='True',
		help='Boolean for using the multiprocessing functions for parallel computing (True or False)')
	cmd_parser.add_option(
		'-L', '--location', dest='location',
		default='None',
		help='location keyword for subsetting by geojson shapefile ("Negri", "Jak", "Greenland", ...)')
	cmd_parser.add_option(
		'-D', '--density-histo', dest='density_histo_bool',
		default='False',
		help='Boolean for using density histograms in the DDA-Bifurcate (instead of Elevation histos) (True or False)')
	cmd_parser.add_option(
		'-H', '--plot-histo', dest='histo_plot_bool',
		default='False',
		help='Boolean for plotting histograms for the DDA-Bifurcate (True or False)')
	cmd_parser.add_option(
		'-M', '--min-peak', dest='min_peak',
		default=3,
		help='Minimum peak for histogram peaks in the DDA-Bifurcate ')
	cmd_parser.add_option(
		'-O', '--min-prom', dest='min_prom',
		default=3,
		help='Minimum prominence for histogram peaks in the DDA-Bifurcate ')
	cmd_parser.add_option(
		'--thresh-class', dest='thresh_class',
		default=False,
		help='Bool for if you want to save photon classifications (default=False b/c of memory allocation)')
	cmd_parser.add_option(
		'-g', '--segment-length', dest='segment_length',
		default=1000,
		help='Along-track length of plots in meters (default 1000m)')
	cmd_parser.add_option(
		'-C', '--chunk-size', dest='chunk_size',
		default=5000,
		help='Size of data chunk in along-track meters for more effcient computation')
	cmd_parser.add_option(
		'--lon-start', dest='lon_start',
		default=-180,
		help='Longtitude at which to start'
		)
	cmd_parser.add_option(
		'--lon-end', dest='lon_end',
		default=180,
		help='Longtitude at which to end'
		)
	cmd_parser.add_option(
		'--lat-start', dest='lat_start',
		default=-90,
		help='Latitude at which to start'
		)
	cmd_parser.add_option(
		'--lat-end', dest='lat_end',
		default=90,
		help='Latitude at which to end'
		)
	cmd_parser.add_option(
		'-f', '--cloud-filter', dest='cloud_filter',
		default='False',
		help='Boolean control param used for integrating the cloud filter algorithm (filter ground estimate)'
	)

	algo_start = time.time()

# Parse options and arguments given via command line
	options, filepath = cmd_parser.parse_args()  # Because no flag for filepath
	filepath = filepath[0]

	# Assign vector values
	run_name = options.name
	sigmas = [float(el) for el in options.sigma.split(',')]
	cutoffs = [float(el) for el in options.cutoff.split(',')]
	aniso_factors = [float(el) for el in options.aniso.split(',')]
	binsizes = [float(el) for el in options.binsize.split(',')]
	quantiles = [float(el) for el in options.quantile.split(',')]
	threshold_offsets = [float(el) for el in options.threshold_offset.split(',')]
	pixel_dimensions = [float(el) for el in options.pixel_dimensions.split(',')]
	slab_thickness = int(options.slab_thickness)

	# Assign scalar values
	track_start = float(options.track_start)
	track_end = float(options.track_end)
	time_start = float(options.time_start)
	time_end = float(options.time_end)
	lon_start = float(options.lon_start)
	lon_end = float(options.lon_end)
	lat_start = float(options.lat_start)
	lat_end= float(options.lat_end)
	interp_ress = [float(el) for el in options.interp_res.split(',')]
	interp_factors = [float(el) for el in options.interp_factor.split(',')]
	std_devs = [float(el) for el in options.std_dev.split(',')]
	crev_depth_quantiles = [float(el) for el in options.crev_depth_quantile.split(',')]  # (bin_width, vertical_bin_width)
	segment_length = float(options.segment_length)  # Size of each segment of plot in meters
	chunk_size = int(options.chunk_size) # in meters

	# Get correct channel number for a givin instrument
	instrument = options.instrument.lower()
	if instrument != 'atlas':
		channel = '0' * (3 - len(options.channel)) + options.channel  # needs to be '00X' for simpl and mabel
	else:
		channel = options.channel

	# Assign boolean values with conditional checks
	plot_TF = handle_bool_params(options.plot_TF, 'plot_TF')
	meltpond_bool = handle_bool_params(options.meltpond_bool, 'meltpond_bool')
	isParallel = handle_bool_params(options.isParallel, 'isParallel')
	cloud_filter = handle_bool_params(options.cloud_filter, 'cloud_filter')
	density_histo_bool = handle_bool_params(options.density_histo_bool, 'density_histo_bool')
	histo_plot_bool = handle_bool_params(options.histo_plot_bool, 'histo_plot_bool')

	location = None if options.location.lower() == 'none' else options.location
	cloud_tolerance = (-1 * float(options.cloud_tolerance[3:])) if options.cloud_tolerance[0:3] == 'neg' else float(options.cloud_tolerance)

	# Assign DDA-bif specific values
	mp_bin_h = float(options.mp_bin_h)
	mp_bin_v = float(options.mp_bin_v)
	den_weight = float(options.den_weight)
	mp_quantile = float(options.mp_quantile)
	min_peak = float(options.min_peak)
	min_prom = float(options.min_prom)

	# Duplicate single parameters if more than 1 pass (DDA-ice-2)
	num_passes = max([len(sigmas), len(cutoffs), len(aniso_factors), len(binsizes), len(quantiles),
		len(threshold_offsets), len(std_devs)])
	if num_passes > 1:
		if len(sigmas) == 1:
			sigmas = sigmas * num_passes
		if len(cutoffs) == 1:
			cutoffs = cutoffs * num_passes
		if len(aniso_factors) == 1:
			aniso_factors = aniso_factors * num_passes
		if len(binsizes) == 1:
			binsizes = binsizes * num_passes
		if len(quantiles) == 1:
			quantiles = quantiles * num_passes
		if len(threshold_offsets) == 1:
			threshold_offsets = threshold_offsets * num_passes
		if len(interp_ress) == 1:
			interp_ress = interp_ress * num_passes
		if len(interp_factors) == 1:
			interp_factors = interp_factors * num_passes
		if len(std_devs) == 1:
			std_devs = std_devs * num_passes
		if len(crev_depth_quantiles) == 1:
			crev_depth_quantiles = crev_depth_quantiles * num_passes

# Setup output directories and run_name
	if run_name is None:
		fileName = filepath.split('/')  # split into parts
		fileName = fileName[-1]  # Take the last part
		fileName = fileName[:-3]  # Remove the file extension
		run_name = '{}_{}'.format(fileName, channel)

	outdir = os.path.join('../output', run_name)  # output directory for logs, data, plot directory

	plotdir = os.path.join(outdir, 'plots')  # plot directory

	# Make output directories
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	else:
		print('A directory already exists with this same outdir: {}'.format(outdir))
		return
	if not os.path.exists(plotdir): os.makedirs(plotdir)

	# Check strength of beam from corresponding channel. Change weak beams into strong beams in same beam pair
	beamStrength = determine_beam_strength(filepath, channel)
	forceStrong = False
	if forceStrong is True:
		if beamStrength == 'weak':
			oldBeamStrength = beamStrength
			oldChannel = channel
			if channel[-1] == 'r': channel = channel[:3] + 'l'  # Just change the last letter of the channel
			elif channel[-1] == 'l': channel = channel[:3] + 'r'  # Just change the last letter of the channel
			beamStrength = determine_beam_strength(filepath, channel)

	# Set up logging at default level DEBUG to .log file and with specified level (WARNING or DEBUG) in stream output
	log_filename = run_name + '.log'
	logger = logging.getLogger('DDA_ice')
	logger.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s - %(message)s')  # Format each line with as "time - mesage"

	# Set up a filehandler to write to the .log file at DEBUG level
	fh = logging.FileHandler(os.path.join(outdir, log_filename), mode='w')
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(formatter)

	# Set up a streamhandler to write to the screen during the run at DEBUG level
	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	ch.setFormatter(formatter)

	# add log handlers
	logger.addHandler(fh)
	logger.addHandler(ch)


	# Log all input parameters
	logger.info('INPUT PARAMETERS')
	logger.info('File name: ' + __file__)
	logger.info('Directory: ' + os.getcwd())
	logger.info('Run name: ' + run_name)
	logger.info('Output directory: ' + outdir)
	logger.info('Plots directory: ' + plotdir)
	logger.info('Instrument: ' + instrument)
	logger.info('Input filepath: ' + filepath)
	logger.info('Channel: ' + channel + ' (' + beamStrength + ')')
	logger.info('Number of passes: {}'.format(num_passes))
	if isParallel:
		logger.info('Parallel Computing using multiprocessing?: {}, {} processors'.format(str(isParallel), multiprocessing.cpu_count()))
	else:
		logger.info('Parallel Computing using multiprocessing?: {}, {} processor'.format(str(isParallel), 1))
	logger.info('Time Start: {}'.format(time_start))
	logger.info('Time End: {}'.format(time_end))
	logger.info('Pixel size for histogram-based slabs: {}'.format(pixel_dimensions))
	logger.info('Curvature threshold for cloud filtering: {}'.format(cloud_tolerance))
	logger.info('Bin width for thresholding: {}'.format(binsizes))
	logger.info('Gaussian RBF StDev: {}'.format(sigmas))
	logger.info('Gaussian RBF cutoff: {}'.format(cutoffs))
	logger.info('Anisotropy factor: {}'.format(aniso_factors))
	logger.info('Threshold quantile: {}'.format(quantiles))
	logger.info('Threshold offset: {}'.format(threshold_offsets))
	logger.info('Resolution of interpolated ground: {}'.format(interp_ress))
	logger.info('Factor for resolution increase over rough surface: {}'.format(interp_factors))
	logger.info('Slab thickness: {}'.format(slab_thickness))
	logger.info('Standard deviation parameter: {}'.format(std_devs))
	logger.info('Quantile for  crevasse depth: {}'.format(crev_depth_quantiles))
	logger.info('DDA_Bifurcate (Melt Pond) Run?: {}'.format(meltpond_bool))
	logger.info('Horizontal Bin Size (DDA_Bif): {}'.format(mp_bin_h))
	logger.info('Vertical Bin Size (DDA_Bif): {}'.format(mp_bin_v))
	logger.info('Density weight for Lower Surface (DDA_Bif): {}'.format(den_weight))
	logger.info('Meltpond Quantile (DDA_Bif): {}'.format(mp_quantile))
	logger.info('Using Density Histograms? (DDA_Bif): {}'.format(density_histo_bool))
	logger.info('Plotting histograms? (DDA_Bif): {}'.format(histo_plot_bool))
	logger.info('Minimum peak in histograms (DDA_Bif): {}'.format(min_peak))
	logger.info('Minimum prominence in histograms (DDA_Bif): {}'.format(min_prom))
	logger.info('Data chunk size (in along-track meters): {}'.format(chunk_size))
	if cloud_filter:
		logger.info('Cloud Filter: True')
	else:
		logger.info('Cloud Filter: False')

	#Calculate kernel size
	kernel_width = 2*aniso_factors[0]*sigmas[0]*cutoffs[0]
	kernel_height = 2*sigmas[0]*cutoffs[0]
	logger.info('Kernel width in meters (pass0): {}'.format(kernel_width))
	logger.info('kernel height in meters (pass0): {}'.format(kernel_height))
	if num_passes >1:
		kernel_width1 = 2*aniso_factors[1]*sigmas[1]*cutoffs[1]
		kernel_height1 = 2*sigmas[1]*cutoffs[1]
		logger.info('Kernel width in meters (pass1): {}'.format(kernel_width1))
		logger.info('Kernel height in meters (pass1): {}'.format(kernel_height1))


	####################################################################################################
	####################################################################################################
	############################                                    ####################################
	############################             Algo Start             ####################################
	############################                                    ####################################
	####################################################################################################
	####################################################################################################

	# initialize DDA-ice class
	DDA = DDAice(filepath, channel, instrument, pixel_dimensions, slab_thickness, segment_length, location, logger, outdir)

	################################################
	########  Step 1: load raw photon data  ########
	################################################

	# Load raw data into DDA-ice class
	DDA.load_photon_data()
	logger.info('Photon data read as: ' + DDA.data_fmt)
	logger.info('Velocity data read as: [delta_time, x_waVelocity, y_waVelocity]')

	# Compute along-track distance from velocity data; store in DDA-ice class
	DDA.distance_from_velocity()
	logger.info('Photon data appended with distance along track: [delta_time, lon, lat, elev, distance]')

	# calculate how long the track is
	logger.info('Total track length: {}'.format(np.max(DDA.photon_data[:, 4])))

	# PRE FILTER THE DATA
	if not DDA.pre_filter(): return

	# subset data over region of interest (if desired)
	if DDA.subset_data_over_area_of_interest(track_start, track_end, time_start, time_end) is None:
		return
	elif location is not None:
		plot_subset_granule_over_region(DDA)

	#subset data over bounding box (if desired)
	if DDA.subset_data_from_bounding_box(lat_start, lat_end, lon_start, lon_end) is None:
		return

	# FILTER BY PODPPD_FLAG (after subset b/c of compute time involved)
	DDA.filter_photons_by_podppd()
    
	# make array of plot segments to use for all plotting functions (based on segment length)
	DDA.compute_plot_segments()

	logger.info('Plot segment length: {}'.format(segment_length))
	logger.info('Number of segments for plotting: {}'.format(len(DDA.plot_segments)))


	if isParallel:  # Save in a separate thread to continue with code execution in the foreground
		save_thread = threading.Thread(target=np.savetxt, args=(os.path.join(outdir, 'raw_photon_data.txt'), DDA.photon_data))
		save_thread.start()
	else:
		np.savetxt(os.path.join(outdir, 'raw_photon_data.txt'), DDA.photon_data)


	# Plot the raw data (Fig 1)
	if plot_TF is True:
		plot_raw_data(DDA)
		# 	rawPlotProcess = multiprocessing.Process(target=plot_raw_data, args=(DDA))
		# 	rawPlotProcess.start()
		# else:
		# 	plot_raw_data(DDA)  # Original

	#################################################
	#### Step 2: Calculate signal and noise slab ####
	#################################################

	start = time.time()
	DDA.assign_slabs_by_histogram_max_bin()
	logger.info('Time elapsed assigning slabs: {}'.format(time.time() - start))

	# filter noise and signal photons
	# result: DDA.photon_signal and DDA.photon_noise
	DDA.signal_noise_separation()

	# Plot the Signal and Noise slabs
	if plot_TF is True:
		plot_slabs(DDA)  # Original


	#######################################
	###### Step 3: Calculate density ######
	#######################################

	## Start of for-loop with each loop corresponding to one pass of the algo

	# For each pass, get the corresponding set of parameters
	for pass_num, (sigma, cutoff, aniso_factor, binsize, quantile, threshold_offset, interp_res,
		interp_factor, std_dev, crev_depth_quantile) in enumerate(zip(sigmas, cutoffs,
		aniso_factors, binsizes, quantiles, threshold_offsets, interp_ress, interp_factors,
		std_devs, crev_depth_quantiles)):

		logger.info('Beginning pass {}'.format(pass_num))

		if pass_num == 0:
			start = time.time()
			if isParallel:
				DDA.compute_density_parallel_3(sigma, cutoff, aniso_factor)
				logger.info('Time elapsed computing density (parallel): {} seconds'.format(time.time() - start))
			else:
				DDA.compute_density(sigma, cutoff, aniso_factor, chunk_size)

				logger.info('Time elapsed computing density: {} seconds'.format(time.time() - start))

		else:
			# photon_signal_orig --> photon_signal_all --> all_photons: all photons
			# photon_signal_pass0 --> pass0_final_photons: photons that show up in final ground estimate for pass0

			# TODO: update DDA.signal_mask in thresholding, 
			photon_noise = photon_noise[:, :5]
			photon_signal = np.array([pho for pho in DDA.photon_data[:, 0:5] if pho not in photon_signal[:, 0:5]])

			# photon_signal_pass0 = photon_signal
			# all_photons = [tuple(x) for x in photon_signal_all[:, 0:5]]
			# pass0_final_photons = [tuple(x) for x in photon_signal[:, 0:5]]  # photon signal from previous pass

			# Remove all pass0 final signal photons (we don't want to use them in pass 2)
			# photon_signal = np.array([pho for pho in all_photons if pho not in pass0_final_photons])
			# photon_signal = np.array([True if pho not in photon_signal[:, 0:5] else False for pho in DDA.photon_data[:, 0:5]])

			# TODO: eliminate pass 2 computation of noise_density -- same compute density function for pass 1 and pass 2
			start = time.time()
			if isParallel:
				signal_density = compute_density_parallel_2(photon_signal, sigma, cutoff, aniso_factor, logger)
				noise_density = compute_density_parallel_2(photon_all[noise_mask], sigma, cutoff, aniso_factor, logger)
				logger.info('Time elapsed computing density (parallel): {} seconds'.format(time.time() - start))
			else:
				signal_density = compute_density(photon_signal, sigma, cutoff, aniso_factor, logger)
				noise_density = compute_density(photon_all[noise_mask], sigma, cutoff, aniso_factor, logger)
				logger.info('Time elapsed computing density: {} seconds'.format(time.time() - start))



			signal_density.shape = (len(signal_density), 1)  # modify shape for appending
			noise_density.shape = (len(noise_density), 1)  # modify shape for appending
			photon_noise = np.append(photon_noise[:, 0:5], noise_density, axis=1)
			del photon_all
			# gc.collect()
			all_density = np.append(signal_density, noise_density)

		logger.info('Signal and noise data appended with density dimension: [delta_time, lon, lat, elev, distance, density]')

		# Plot Density (Fig 3s)
		if plot_TF is True:
			# if isParallel:  # Plot as a separate process to continue with code execution in the current process
				# densityPlotProcess = multiprocessing.Process(target=plot_density, args=(DDA), kwargs={'ax': None})
				# densityPlotProcess.start()
			# else:
			plot_density(DDA, ax=None)

		logger.info('Finished plotting density...')

		# weighted photon format: [delta_time, lon, lat, elev, distance, density]
		if isParallel:  # Save in a separate thread to continue with code execution in the foreground
			save_thread = threading.Thread(target=np.savetxt, args=(os.path.join(outdir, 'weighted_photons_pass{}.txt'.format(pass_num)), DDA.photon_signal))
			save_thread.start()
		else:
			np.savetxt(os.path.join(outdir, 'weighted_photons_pass{}.txt'.format(pass_num)), DDA.photon_signal)  # Original



		################################################################
		###### Step 4: Thresholding -- Finding the signal photons ######
		################################################################

		start = time.time()
		DDA.compute_thresholds(threshold_offset, quantile, binsize)
		logger.info('Time elapsed computing thresholds: {}'.format(time.time() - start))

		if plot_TF is True:
			plot_thresholds(DDA, ax=None)  # Original

		if options.thresh_class:
			# if we want to save the threshold-based classifications of every photon
			DDA.classify_photons_from_thresholding()
		else:
			DDA.set_signal_threshold_photons()
		

		####################################################
		###### Optional Step 4b: DDA-Bifurcate Option ######
		####################################################

		if meltpond_bool:

			## Give signal photons from thresholding to the BIF code to find top-surface signal photons and bottom-surface signal photons (9/3/21)
			# Note that the histogram plot code is within the following function (Figure 4b's)
			logger.info('Starting Bifurcate Code')
			DDA.compute_thresholds_melt_pond(threshold_offset, quantile, binsize, mp_bin_h, mp_bin_v, density_histo_bool, histo_plot_bool, chunk_size)
			# signal_mask = np.logical_or(signal_mask_bot, signal_mask_top)

			# Creating self.photon_signal_top & self.photon_signal_bot
			DDA.set_top_and_bottom_signal()


		########################################
		####### Step 5: Ground following #######
        ########################################

        # Option 1: No bifuraction, i.e., regular DDA-ice code, only one surface to find and plot
		if meltpond_bool is False:
			# interpolate_ground produces output in the following format
			# [bin_lon, bin_lat, bin_elev, bin_distance, bin_elev_stdev, bin_density_mean, bin_weighted_stdev]
			start = time.time()
			DDA.interpolate_ground_tom(interp_res, interp_factor, std_dev, crev_depth_quantile, meltpond_bool, mp_quantile)
			logger.info('Time elapsed interpolating ground: {}'.format(time.time() - start))

			# TODO: make better control for this
			interpolate_missing_data_TF = True
			if interpolate_missing_data_TF:
				DDA.interpolate_missing_data(interp_res)

			if cloud_filter:
				logger.info('Filtering cloudy data from ground estimate...')
				DDA.integrate_cloud_filter_mask()

			if isParallel:  # Save in a separate thread to continue with code execution in the foreground
				save_thread = threading.Thread(target=np.savetxt, args=(os.path.join(outdir, 'ground_estimate_pass{}.txt'.format(pass_num)), DDA.ground_estimate))
				save_thread.start()
			else:
				np.savetxt(os.path.join(outdir, 'ground_estimate_pass{}.txt'.format(pass_num)), DDA.ground_estimate)  # Original

			logger.info('Final ground estimate saved as ground_estimate_pass' + str(pass_num) + '.txt')
			logger.info('Format of ground estimate:\n[bin_lon, bin_lat, bin_elev, bin_distance, bin_time, bin_elev_stdev, bin_density_mean, bin_weighted_stdev]')

			if plot_TF is True:
				# TODO: adapt plot_all() method
				# plot_ground_estimate_no_interpolate(DDA, ax=None)
				plot_ground_estimate(DDA, meltpond_bool, ax=None)  # Original
				# plot_all(photon_data, plot_slabs_photon_signal, plot_slabs_photon_noise, full_signal, photon_noise, plot_thresholds_photon_signal, plot_thresholds_signal_mask, threshold_mask, photon_signal, ground_estimate, plot_segments, run_name, pass_num, plotdir)  # Original

		# Option 2: Bifurcation code (equivalent to melt-pond finding at this point), finds and plots two surfcaes
		else:
			# Find TOP surface 
			DDA.interpolate_ground_tom(interp_res, interp_factor, std_dev, crev_depth_quantile, meltpond_bool, mp_quantile, top_bool=True) 
			logger.info('Finished interpolating ground...')

			# Find BOTTOM surface
			DDA.interpolate_ground_tom(interp_res, interp_factor, std_dev, crev_depth_quantile, meltpond_bool, mp_quantile)

			# Correct ponds by removing false positives. TODO: Correct pond edges and account for saturation better than using a simple depth threshold
			logger.info('Correcting ponds')
			DDA.correct_ponds()

			##### Plot and save both surfaces #####
			# TOP surface
			if plot_TF is True:
				# Plot top surface (red) Fig 5 pass0
				plot_ground_estimate(DDA, meltpond_bool, topBool=True)

			if isParallel:  # Save in a separate thread to continue with code execution in the foreground
				save_thread = threading.Thread(target=np.savetxt, args=(os.path.join(outdir, 'ground_estimate_pass_top.txt'), DDA.ground_estimate_top))
				save_thread.start()
			else:
				np.savetxt(os.path.join(outdir, 'ground_estimate_pass_top.txt'), DDA.ground_estimate_top)  # Original

			logger.info('Top Final ground estimate saved as ground_estimate_pass_top.txt')
			logger.info('Format of top ground estimate:\n[bin_lon, bin_lat, bin_elev, bin_distance, bin_time, bin_elev_stdev, bin_density_mean, bin_weighted_stdev]')

			# BOTTOM surface
			if isParallel:  # Save in a separate thread to continue with code execution in the foreground
				save_thread = threading.Thread(target=np.savetxt, args=(os.path.join(outdir, 'ground_estimate_pass_bot.txt'), DDA.ground_estimate_bot))
				save_thread.start()
			else:
				np.savetxt(os.path.join(outdir, 'ground_estimate_pass_bot.txt'), DDA.ground_estimate_bot)  # Original

			logger.info('Bottom Final ground estimate saved as ground_estimate_pass_bot.txt')
			logger.info('Format of bottom ground estimate:\n[bin_lon, bin_lat, bin_elev, bin_distance, bin_time, bin_elev_stdev, bin_density_mean, bin_weighted_stdev]')
			
			if plot_TF is True:
				# Plot bottom surface (green) Fig 5 pass1
				plot_ground_estimate(DDA, meltpond_bool, ax=None)

			# Save pond characteristics
			if isParallel:  # Save in a separate thread to continue with code execution in the foreground
				save_thread = threading.Thread(target=np.savetxt, args=(os.path.join(outdir, 'pond_chars.txt'), DDA.pond_edges))
				save_thread.start()
			else:
				np.savetxt(os.path.join(outdir, 'pond_chars.txt'), DDA.pond_edges)  # Original

			logger.info('Pond edges and characteristics saved as pond_chars.txt')
			logger.info('Format of pond_chars:\n[at_left_edge, at_right_edge, pond_width, mean_pond_depth]')


		########## END PASS LOOP ##########
		DDA.increment_pass_num()


	if (num_passes > 1) or (meltpond_bool is True):
		if plot_TF is True:
			# Plot both surfaces (top and bottom) Fig 6
			plot_ground_estimate_both(DDA, ax=None)
			logger.info('Final ground estimate with both surfaces')

	if meltpond_bool is True and plot_TF is True:  # Make the MegaPlot with both ground estimates
		plot_all_both_grounds(DDA)  # original


	logger.info('Total time elapsed for algorithm: ' + str(time.time() - algo_start))

	return




if __name__ == "__main__":
	main()

