import h5py
import time
import scipy.signal
import ctypes
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, KDTree
from scipy.stats import norm as gaussian
from skimage import filters
import geopandas as gpd
# from memory_profiler import profile
import gc
import os

# import ray  # Not needed unless using compute_density_parallel_ray
# ray.init()

class DDAice:

	def __init__(self, filepath, channel, instrument, pixels, slab_thick, segLength, location, logger, outputDirectory):

		self.filepath = filepath
		self.channel = channel
		self.instrument = instrument
		self.pixel_dimensions = pixels
		self.slab_thickness = slab_thick
		self.logger = logger
		self.outdir = outputDirectory
		self.plotdir = os.path.join(self.outdir, 'plots')
		self.pass_num = 0

		# data attributes
		self.data_fmt = None
		self.photon_data = None
		self.velocity_data = None
		self.density = None
		self.photon_signal = None
		self.photon_noise = None
		self.photon_signal_thresh = None # only signal photons that pass thresholding

		# mask attributes
		self.signal_mask = None
		self.noise_mask = None
		self.threshold_mask = None

		# location attributes
		self.location = location
		self.shape_polygon = None
		self.crs = None

		# ground estimate attributes
		self.ground_estimate = np.empty(shape=(0, 8)) 

		# plotting attributes
		self.plot_segments = None
		self.segment_length = segLength

		# DDA Bif attributes
		self.signal_mask_top = None
		self.signal_mask_bot = None
		self.pond_edges = None
		self.photon_signal_top = None
		self.photon_signal_bot = None
		self.ground_estimate_top = np.empty(shape=(0, 8)) 
		self.ground_estimate_bot = np.empty(shape=(0, 8)) 

	def load_photon_data(self):
		'''
		Loading function for H5 data
		'''
		if self.instrument == 'simpl':
			f = h5py.File(self.filepath, 'r')
			self.photon_data = np.array([
				f['/photon/channel' + self.channel + '/delta_time'][()],
				f['/photon/channel' + self.channel + '/longitude'][()],
				f['/photon/channel' + self.channel + '/latitude'][()],
				f['/photon/channel' + self.channel + '/elev'][()]
				]).T
			self.velocity_data = np.array([
				f['/ins/delta_time'][()],
				f['/ins/location/x_waVelocity'][()],
				f['/ins/location/y_waVelocity'][()]
				]).T
			self.data_fmt = '[dt, lon, lat, elev]'
		elif self.instrument == 'mabel':
			f = h5py.File(self.filepath, 'r')
			self.photon_data = np.array([
				f['/channel' + self.channel + '/photon/delta_time'][()],
				f['/channel' + self.channel + '/photon/ph_longitude'][()],
				f['/channel' + self.channel + '/photon/ph_latitude'][()],
				f['/channel' + self.channel + '/photon/ph_h'][()]
				]).T
			self.velocity_data = np.array([
				f['/novatel_ins/delta_time'][()],
				f['/novatel_ins/velocity/ins_v_east'][()],
				f['/novatel_ins/velocity/ins_v_north'][()]
				]).T
			self.data_fmt = '[dt, lon, lat, elev]'
		elif self.instrument == 'atlas':  # channel = gt1l, gt1r, gt2l, gt2r, gt3l, or gt3r
			f = h5py.File(self.filepath, 'r')
			self.photon_data = np.array([
				f['/' + self.channel + '/heights/delta_time'][()],
				f['/' + self.channel + '/heights/lon_ph'][()],
				f['/' + self.channel + '/heights/lat_ph'][()],
				f['/' + self.channel + '/heights/h_ph'][()]
				]).T
			self.velocity_data = np.array([
				f['/' + self.channel + '/geolocation/delta_time'][()],
				f['/' + self.channel + '/geolocation/velocity_sc'][:, 0],
				f['/' + self.channel + '/geolocation/velocity_sc'][:, 1]
				]).T
			self.dem = np.array([
				f['/' + self.channel + '/geophys_corr/dem_h'],
				f['/' + self.channel + '/geophys_corr/delta_time']
				]).T
			self.quality_ph = np.array([f['/' + self.channel + '/heights/quality_ph']])[0]
			self.signal_conf_ph = np.array([f['/' + self.channel + '/heights/signal_conf_ph']])
			# print(self.quality_ph.shape)
			self.set_podppd_flag(np.array(f['/' + self.channel + '/geolocation/podppd_flag']))
			self.data_fmt = '[dt, lon, lat, elev]'
		else:
			raise ParamValueError(self.instrument)

	def increment_pass_num(self):
		self.pass_num += 1

	def compute_plot_segments(self):
		self.min_distance = np.floor(np.nanmin(self.photon_data[:, 4]) / 1000.) * 1000
		self.max_distance = np.ceil(np.nanmax(self.photon_data[:, 4]) / 1000.) * 1000

		self.plot_segments = np.array([np.arange(self.min_distance, self.max_distance, self.segment_length),
			np.arange(self.min_distance + self.segment_length, self.max_distance + self.segment_length, self.segment_length)]).T

	def integrate_cloud_filter_mask(self):
		'''
		Apply cloud filter mask to ground estimate, where the cloud filter mask is set in assign_slabs function
		'''
		# attach along-track distances to cloud_filter_mask
		start = np.min(self.photon_data[:, 4])
		dist = [(i * self.pixel_dimensions[0]) + start for i in range(len(self.cloud_filter_mask))]
		self.cloud_filter_mask = np.c_[self.cloud_filter_mask, np.array(dist)]

		# [bin_lon, bin_lat, bin_elev, bin_distance, bin_time, bin_elev_stdev, bin_density_mean, bin_weighted_stdev]
		for elem in self.cloud_filter_mask:
			# if mask tells us their IS ground, do nothing
			if elem[0]: continue
			start = elem[1]
			end = start + self.pixel_dimensions[0]
			self.ground_estimate[np.logical_and(self.ground_estimate[:, 3] >= start, self.ground_estimate[:, 3] < end), :] = np.NaN

	def set_podppd_flag(self, podppd):
		"""
		using delta_time to map /geolocation/podppd_flag to photon data
		See ATL03 ATBD for more info on podppd_flag...
		"""
		# First, check if necessary, or if all the data is fine
		self.skip_dataset = False
		all_zeros = not np.any(podppd)
		if all_zeros:
			# print("all zeros!")
			# if only 0's in podppd_flag: good data
			self.filter_by_podppd = False

		elif 0 not in podppd:
			# print("all bad data!")
			# No 0's = bad track, skip this dataset
			self.skip_dataset = True
		else:
			# print("mix!")
			# mix of bad and good data, utilize mask
			self.filter_by_podppd = True # this should be TRUE

			# geolocation delta_time indices where podppd flag indicates bad data
			# these get filtered out after the data is subset by area of interest
			self.podppd_bad_idxs = np.where(podppd != 0.)[0]

	def filter_photons_by_podppd(self):
		'''
		Filter out bad data identified by podppd_flag (on product) before DDA computations
		to avoid off pointing or other bad data affecting the DDA
		'''
		if not self.filter_by_podppd: return

		dt_geo = self.velocity_data[:, 0] # /geolocation/delta_time
		dt_heights = self.photon_data[:, 0] # /heights/delta_time
		min_dt, max_dt = min(dt_heights), max(dt_heights)
		self.podppd_flag = np.zeros(len(self.photon_data))
		# print(len(self.podppd_bad_idxs))

		for i in self.podppd_bad_idxs:
			if min_dt > dt_geo[i]: continue
			if max_dt < dt_geo[i]: break
			print(i)

			start = np.argmax(dt_heights > dt_geo[i])
			end = np.argmax(dt_heights > dt_geo[min(i+1, len(dt_geo)-1)])
			self.podppd_flag[start:end] = 1.

		self.photon_data[self.podppd_flag == 1., :] = np.NaN

	def pre_filter(self):
		'''
		Handler for determining if entire granule is bad or if there are areas pre-identified on product where data is unusable
		This function filters out bad data at the product level
		Later, we do DDA based filtering for cloudy and/or sparse data
		'''
		if self.skip_dataset:
			self.logger.info('podppd_flag indicates this granule is unusable... Exiting...')
			return False
		
		if self.filter_by_podppd:
			self.logger.info('########## Filtering photon data by podppd_flag...')
			# Moved filter photons by podppd functionality to AFTER data subsetting b/c compute time it takes
			# self.filter_photons_by_podppd()
		else:
			self.logger.info('NOT filtering by podppd_flag...')

		if not np.any(self.quality_ph):
			self.logger.info('NOT filtering by quality_ph...')
		else:
			self.logger.info('########## Filtering photon data by quality_ph...')
			if not self.filter_photons_by_quality_ph():
				self.logger.info('quality_ph indicates this granule is unusable... Exiting...')
				return False

		if -2 not in self.signal_conf_ph:
			self.logger.info('NOT filtering by signal_conf_ph...')
		else:
			self.logger.info('########## Filtering photon data by signal_conf_ph...')
			if not self.filter_photons_by_signal_conf_ph(): 
				self.logger.info('signal_conf_ph indicates this granule is unusable... Exiting...')
				return False

		# print(self.photon_data)
		return True

	def filter_photons_by_quality_ph(self):
		if 0 not in self.quality_ph: return False
		self.photon_data[self.quality_ph != 0, :] = np.NaN
		return True

	def filter_photons_by_signal_conf_ph(self):
		invalid = np.argwhere(self.signal_conf_ph == -2.)
		invalid_row_idxs = np.unique(invalid[:,0])
		# check if ALL data is low confidence, if so skip dataset
		if len(invalid_row_idxs) == len(self.photon_data): return False
		self.photon_data[invalid_row_idxs] = np.NaN
		return True

	def distance_from_velocity(self):
		'''
		Calculates along-track distance from interpolating and integrating velocity
		'''
		velocity_magnitude = np.sqrt(self.velocity_data[:, 1]**2 + self.velocity_data[:, 2]**2)
		tdiffs = self.photon_data[1:, 0] - self.photon_data[0:-1, 0]  # time between photon returns
		# tdiffs = np.subtract(self.photon_data[1:, 0].filled(0), self.photon_data[0:-1, 0].filled(0))  # time between photon returns
		if self.instrument == 'atlas':
			velocity_good = velocity_magnitude[velocity_magnitude < 10000]  # remove erroneous velocities faster than 10000 m/s
			mean_velocity = np.mean(velocity_good)
		else:
			velocity_at_returns = np.interp(x=self.photon_data[:, 0], xp=self.velocity_data[:, 0], fp=velocity_magnitude)
			mean_velocity = np.mean([velocity_at_returns[0:-1], velocity_at_returns[1:]], axis=0)  # take mean velocity during those times
		distance = np.zeros(len(self.photon_data))  # set up array to hold distance values
		distance[1:] = np.cumsum(tdiffs * mean_velocity)  # take cumulative sum to find distance at each point
		distance.shape = (len(distance), 1)  # modify shape for appending
		self.photon_data = np.append(self.photon_data, distance, axis=1)
	
	def subset_data_over_area_of_interest(self, track_start, track_end, time_start, time_end):

		max_distance = np.max(self.photon_data[:, 4])
		# min_delta_time, max_delta_time = np.min(self.photon_data[:,0]), np.max(self.photon_data[:,0])
		min_delta_time, max_delta_time = np.nanmin(self.photon_data[:,0]), np.nanmax(self.photon_data[:,0])

		# Chop data down by input start and end distances
		if track_start <= 0.:
			track_start = 0.
			self.logger.info('Default track start point used (0): {}'.format(track_start))
		else:
			self.logger.info('User selected track start point: {}'.format(track_start))

		if track_end > max_distance:
			track_end = max_distance
			self.logger.info('Default track end point used (max_distance): {}'.format(track_end))
		else:
			self.logger.info('User selected track end point: {}'.format(track_end))
		max_distance = track_end
		data_range = np.logical_and(self.photon_data[:, 4] >= track_start, self.photon_data[:, 4] <= track_end)

		# Chop data down by input start and end times
		self.logger.info('Total track time: {}'.format(max_delta_time))
		if time_start < min_delta_time or time_start == 0:
			time_start = min_delta_time
			self.logger.info('Default delta time start point used (min_delta_time): {}'.format(time_start))
		else:
			self.logger.info('User selected delta time start point: {}'.format(time_start))

		if time_end > max_delta_time or time_end <= time_start:
			time_end = max_delta_time
			self.logger.info('Default delta time end point used (max_delta_time): {}'.format(time_end))
		else:
			self.logger.info('User selected delta time end point used: {}'.format(time_end))

		# Chop down the data to the user specified delta_time range
		data_range = np.logical_and(data_range, np.logical_and(self.photon_data[:, 0] >= time_start, self.photon_data[:, 0] <= time_end))
		self.photon_data = self.photon_data[data_range, :]

		# Chop data down by the user specified geojson shape file region
		if self.location is not None:
			self.shape_polygon, self.crs = location_to_polygon(self.location)
			if self.shape_polygon is None:  # location does not match a valid polygon
				self.logger.error('Stopping: Could not associate location "{}" with a polygon'.format(self.location))
				return None
			else:  # Normal case where everything works
				self.logger.info('Subsetting granule with location: {} and crs: {}'.format(self.location, self.crs))
				self.photon_data = subset_data_from_polygon(self.shape_polygon, self.photon_data, self.crs)
			if self.photon_data is None:
				self.logger.warning('Stopping: This granule has no overlap the given location: {}'.format(self.location))
				return None
		return True

	def subset_data_from_bounding_box(self, lat_start, lat_end, lon_start, lon_end):
		lon_mask = np.logical_and(self.photon_data[:,1]>=lon_start, self.photon_data[:,1]<=lon_end) 
		lat_mask = np.logical_and(self.photon_data[:,2]>=lat_start, self.photon_data[:,2]<=lat_end)
		bounding_box_mask = np.logical_and(lon_mask,lat_mask)
		
		self.photon_data = self.photon_data[bounding_box_mask,:]

		if self.photon_data is None:
			return None

		min_delta_time, max_delta_time = np.min(self.photon_data[:,0]), np.max(self.photon_data[:,0])

		self.logger.info('Delta time start point after subset by bounding box: {}'.format(min_delta_time))
		self.logger.info('Delta time end point after subset by bounding box: {}'.format(max_delta_time))
		return True

	def assign_slabs_by_histogram_max_bin(self):
		'''
		Assigns every photon to either the signal slab or noise slab. Uses the maximum elevation bin of the elevation-ATD histogram.
		'''
		dem_ssc_tol_dist = 200 # max tolerance (vertical meters) distance between DEM and signal slab center

		# Get average DEM height for the corresponding horizontal along-track bin determined by delta_time
		def get_dem(time_start, time_end):
			# subset DEM for input time interval
			dem_temp = self.dem[np.where((self.dem[:,1]>=time_start) & (self.dem[:,1]<=time_end))]
			return np.mean(dem_temp[:,0])

		photon_bin_thresh = 35 # min photon count used to filter out cloudy data
		# Format of photon data: [delta_time, longitude, latitude, elevation, distance]
		pix_x, pix_y = self.pixel_dimensions
		horizontal_edges = np.arange(min(self.photon_data[:, 4]), max(self.photon_data[:, 4]) + pix_x, pix_x)  # New
		vertical_edges = np.arange(min(self.photon_data[:, 3]), max(self.photon_data[:, 3]) + pix_y, pix_y)
		H = np.histogram2d(self.photon_data[:, 3], self.photon_data[:, 4], bins=(vertical_edges, horizontal_edges))[0]

		index_of_max_vertical_bins = np.argmax(H, axis=0)  # Each element is the index of the max vertical bin (one element for every horizontal bin)
		signal_slab_center = vertical_edges[index_of_max_vertical_bins] + (pix_y / 2)  # Turn the index of bins into the elevation at the center of that bin
		# Create the empty arrays that will be filled in incrementally as we progress through the hbins
		self.signal_mask = np.array([False] * len(self.photon_data))
		self.noise_mask = np.array([False] * len(self.photon_data))

		# initialize cloud filter mask
		self.cloud_filter_mask = np.array([True] * len(index_of_max_vertical_bins))

		for hbin, (hbin_start, hbin_end) in enumerate(zip(horizontal_edges[:-1], horizontal_edges[1:])):

			# Get bin endpoints quickly from the "sorted" signal and noise arrays
			hbin_endpoints = np.searchsorted(self.photon_data[:, 4], [hbin_start, hbin_end])
			# Cut down photon_data to only the data in the current horizontal bin (hbin)
			hbin_subset = self.photon_data[hbin_endpoints[0]:hbin_endpoints[1], :]
			if len(hbin_subset) <= 0: continue

			# print("hbin: {}, start: {}, end: {}, signal slab center: {}, dem: {}".format(hbin, hbin_start, hbin_end,signal_slab_center[hbin],get_dem(np.nanmin(hbin_subset[:,0]), np.nanmax(hbin_subset[:,0]))))

			# Mask out along-track segments where signal slab has too few photons (i.e. no ground or bad signal)
			# ALSO mask out areas where signal slab and DEM are too far apart for signal slab to actually be ground
			if H[index_of_max_vertical_bins[hbin], hbin] <= photon_bin_thresh:
				self.cloud_filter_mask[hbin] = False
			elif np.absolute(signal_slab_center[hbin] - get_dem(np.nanmin(hbin_subset[:,0]), np.nanmax(hbin_subset[:,0]))) > dem_ssc_tol_dist:
				self.cloud_filter_mask[hbin] = False

			# Signal points are points that are 30% of slab thickness above the slab center and 70% of slab thickness below the slab center
			# Ex: slab_center = 150m; points between 150+(150*0.3) and 150-(150*0.7) --> (195 meters to 45 meters) are considered noise
			# This is because crevasses, ponds, water, ... is usually below the slab center, so we shift the signal slab 20% down.
			percentAbove = 0.30
			percentBelow = 0.70
			# Previously (characterizes noise using photons above and below signal slab):
			# hbin_signal_mask = np.logical_or(hbin_subset[:, 3] - signal_slab_center[hbin] < percentAbove * self.slab_thickness, signal_slab_center[hbin] - hbin_subset[:, 3] < percentBelow * self.slab_thickness)
			# hbin_noise_mask = np.logical_or(hbin_subset[:, 3] - signal_slab_center[hbin] >= percentAbove * self.slab_thickness, signal_slab_center[hbin] - hbin_subset[:, 3] >= percentBelow * self.slab_thickness)

			# Want noise characterized by just the photons above the signal bin!!! 
			center_bin_adj = signal_slab_center[hbin] - (percentBelow - 0.5)*self.slab_thickness
			hbin_signal_mask = np.abs(hbin_subset[:, 3] - center_bin_adj) < .5 * self.slab_thickness
			# For the conditionals below, 0.5 and 1.5 will put the noise slab directly on-top of signal slab
			hbin_noise_mask = np.logical_and((hbin_subset[:, 3] - center_bin_adj) >= 0.5 * self.slab_thickness, 
											(hbin_subset[:, 3] - center_bin_adj) < 1.5 * self.slab_thickness)

			self.signal_mask[hbin_endpoints[0]:hbin_endpoints[1]] = hbin_signal_mask
			self.noise_mask[hbin_endpoints[0]:hbin_endpoints[1]] = hbin_noise_mask

	def signal_noise_separation(self):
		'''
		initialize photon_signal and photon_noise with signal/noise masks from assign_slabs_by_histogram_max_bin()
		'''
		self.photon_signal = self.photon_data[self.signal_mask, :]
		self.photon_noise = self.photon_data[self.noise_mask, :]

	def compute_density(self, sigma, cutoff, aniso_factor, chunk_size):
		'''
		Computes the density for every single photon point. Uses a RBF with a kernel described by sigma, cutoff, and aniso_factor.
		'''
		start_dens = time.time()
		self.density = np.array([])
		half_kernel_width = aniso_factor*sigma*cutoff
		total_chunks = np.ceil((self.photon_data[-1,4]-self.photon_data[0,4])/chunk_size)
		self.logger.info('Number of Chunks: {}'.format(total_chunks))
		counter = -1
		for i in np.arange(self.photon_data[0,4],self.photon_data[-1,4],chunk_size):
			start = time.time()
			# Extended data set (based on kerenel size) needed to calculate density for the actual data chunk
			data_extend_temp_bool = (self.photon_data[:,4] >= (i-half_kernel_width)) & (self.photon_data[:,4] < (i + chunk_size+ half_kernel_width))
			data_extend_temp  = self.photon_data[data_extend_temp_bool,:]
			# Bool for actual data in chunk which we are calculating density for
			data_bool = (data_extend_temp[:,4] >= i) & (data_extend_temp[:,4] < (i + chunk_size))
			self.logger.info('Time elapsed downsampling data to chunk: {}'.format(time.time() - start))

			# Data comes as [delta_time, lon, lat, elev, distance]
			DE = np.array([data_extend_temp[:, 4] / aniso_factor, data_extend_temp[:, 3]]).T  # Distance and Elevation
			start = time.time()
			tree = cKDTree(DE, balanced_tree=False)  # make the KDTree to use for rangesearch. Balanced_tree=False has faster query
			counter = counter +1
			self.logger.info('Chunk Number: {}'.format(counter))
			self.logger.info('Time elapsed creating cKDTree: {}'.format(time.time() - start))
			start = time.time()
			neighbors = tree.query_ball_tree(tree, r=sigma*cutoff, p=2)  # calculate neighbors within radius
			self.logger.info('Time elapsed finding neighbors within kernel: {}'.format(time.time() - start))

			start = time.time()
		
			all_dens_temp = np.array([])
			for pt in range(len(DE)):
				idx = np.array(neighbors[pt])[np.array(neighbors[pt]) != pt]  # take all neigbors except for pt!
				distances = np.linalg.norm(DE[idx, :] - DE[pt, :], axis=1)  # Euclidean distance norm along axis=1
				weight_sum = np.sum(gaussian.pdf(distances, loc=0, scale=sigma))  # sum up pdf values
				all_dens_temp = np.append(all_dens_temp, weight_sum)
			self.density = np.append(self.density, all_dens_temp[data_bool]) # keep only densities in chunnk (not extended borders)
			self.logger.info('Time elapsed calculating density for each point: {}'.format(time.time() - start))
		self.logger.info('Time elapsed calculating density for all points: {}'.format(time.time() - start_dens))
		self.density.shape = (len(self.photon_data),1)

		# append signal/noise density to their respective photon arrays
		self.photon_signal = np.append(self.photon_signal, self.density[self.signal_mask, :], axis=1)
		self.photon_noise = np.append(self.photon_noise, self.density[self.noise_mask, :], axis=1)
		
	def compute_density_parallel_3(self, sigma, cutoff, aniso_factor):
		'''
		Computes the density for every single photon point. Uses a RBF with a kernel described by sigma, cutoff, and aniso_factor.
		Then updates photon_signal and photon_noise arrays with computed density.
		'''
		# Data comes as [delta_time, lon, lat, elev, distance]
		DE = np.array([self.photon_data[:, 4] / aniso_factor, self.photon_data[:, 3]]).T  # Distance and Elevation
		shared_array_base = mp.Array(ctypes.c_double, DE.flatten(), lock=False)
		DE_shared = np.ctypeslib.as_array(shared_array_base)
		DE_shared = DE_shared.reshape(DE.shape)

		start = time.time()
		tree = KDTree(DE, balanced_tree=False)  # make the KDTree to use for rangesearch. Balanced_tree=False has faster query times
		self.logger.info('Time elapsed creating KDTree: {}'.format(time.time() - start))

		start = time.time()
		with mp.Pool(processes=mp.cpu_count()) as pool:  # Create the multiprocessing pool. Each processor kills itself and restarts after maxtasksperchild iterations
			worker_args_iter = [(pt, sigma, cutoff, DE, tree) for pt in range(len(DE))]  # Flag Make this DE_shared?
			density = pool.starmap(density_worker_3, worker_args_iter)
		self.density = np.asarray(density)
		self.density.shape = (len(self.photon_data),1)
		self.logger.info('Time elapsed calculating density for each point: {}'.format(time.time() - start))

		# append signal/noise density to their respective photon arrays
		self.photon_signal = np.append(self.photon_signal, self.density[self.signal_mask, :], axis=1)
		self.photon_noise = np.append(self.photon_noise, self.density[self.noise_mask, :], axis=1)

	def compute_thresholds(self, threshold_offset, quantile, binsize):
		'''
		Removes photons that do not pass as signal.
		It removes photons that have a density less than max_density_of_noise + threshold_offset.
		It also removes all photons that are less than quantile percent of the signal density.
		'''
		# Data comes as [delta_time, lon, lat, elev, distance, density]
		max_distance = int(np.ceil(np.max(self.photon_signal[:, 4])))
		stepsize = int(binsize)

		self.signal_mask = np.array([False] * len(self.photon_signal))  # stores values that passed thresholding and quantile
		self.threshold_mask = np.array([False] * len(self.photon_signal))  # stores values that passed thresholding but not quantile ("blue points")
		prev_threshold = 0.  # store in case we have an empty signal_bool array

		# Sort them once, then we can use "searchsorted" (much faster than logical_and()) inside the loop

		for b in range(0, max_distance, stepsize):
			# Get bin endpoints quickly from the "sorted" signal and noise arrays
			signal_bin_endpoints = np.searchsorted(self.photon_signal[:, 4], [b, b + stepsize])
			noise_bin_endpoints = np.searchsorted(self.photon_noise[:, 4], [b, b + stepsize])

			# Cut down photon_signal and photon_noise to bin_signal and bin_noise
			bin_signal = self.photon_signal[signal_bin_endpoints[0]:signal_bin_endpoints[1], :]
			bin_noise = self.photon_noise[noise_bin_endpoints[0]:noise_bin_endpoints[1], :]

			signal_mask = np.array([False] * len(bin_signal))  # stores values that passed thresholding and quantile
			threshold_mask = np.array([False] * len(bin_signal))  # stores values that passed thresholding but not quantile ("blue points")

			# threshold value for current bin is max noise density value + offset
			if bin_noise.size == 0:
				# if nothing in noise slab, use threshold offset (NOT previous threshold)
				threshold = threshold_offset
			else:
				threshold = np.max(bin_noise[:, 5]) + threshold_offset

			# also add constraint that the signal must be above threshold
			signal_bool = bin_signal[:, 5] > threshold

			# OR existing threshold mask with signal_bool
			threshold_mask = np.logical_or(threshold_mask, signal_bool)

			# calculate the value of the q*100% quantile
			if signal_bool.sum() == 0:
				quantile_value = 100000
			else:
				quantile_value = np.quantile(bin_signal[signal_bool, 5], quantile)

			# take only values that pass the quantile test
			if quantile == 0:
				signal_bool = np.logical_and(signal_bool, bin_signal[:, 5])
			else:
				signal_bool = np.logical_and(signal_bool, bin_signal[:, 5] > quantile_value)

			# OR the existing mask (starts as False) and the current signal boolean
			signal_mask = np.logical_or(signal_mask, signal_bool)

			self.signal_mask[signal_bin_endpoints[0]:signal_bin_endpoints[1]] = signal_mask
			self.threshold_mask[signal_bin_endpoints[0]:signal_bin_endpoints[1]] = threshold_mask

	def classify_photons_from_thresholding(self):
		
		photon_offset = self.photon_signal[~(self.threshold_mask | self.signal_mask), :]
		photon_threshold_only = self.photon_signal[(self.threshold_mask & ~self.signal_mask), :]
		photon_thresholded = self.photon_signal[self.threshold_mask, :]  # Points that passed hard thresholds but not quantile ("blue points")
		self.photon_signal_thresh = self.photon_signal[self.signal_mask, :]  # points that passed hard thresholds AND quantile - used in ground computation

		# Data for saving, add class flag (int): 1-noise (red), 2-points below offset but above max noise (green), 3 - points below quantile but above offset (blue) 4-final signal (light green)
		photon_noise_save = np.hstack((self.photon_noise, 1 * np.ones((len(self.photon_noise), 1), dtype=int)))
		photon_offset_save = np.hstack((photon_offset, 2 * np.ones((len(photon_offset), 1), dtype=int)))
		photon_quantile_save = np.hstack((photon_threshold_only, 3 * np.ones((len(photon_threshold_only), 1), dtype=int)))
		photon_signal_save = np.hstack((self.photon_signal_thresh, 4 * np.ones((len(self.photon_signal_thresh), 1), dtype=int)))

		photon_save_all = np.vstack((photon_noise_save, photon_offset_save, photon_quantile_save, photon_signal_save))


		np.savetxt(os.path.join(self.outdir, 'photons_class_pass{}.txt'.format(self.pass_num)), photon_save_all)
		self.logger.info('Save photon classification: [delta_time, lon, lat, elev, distance, density, class]')
	
	def set_signal_threshold_photons(self):
		# sets the data attribute containing all signal photons that pass the threshold
		self.photon_signal_thresh = self.photon_signal[self.signal_mask, :]

	def interpolate_ground_tom(self, interp_res, interp_factor, std_dev, crev_depth_quantile, meltpond_bool, mp_quantile, top_bool=False):
		'''
		Interpolates the ground in bins of interp_res meters.
		It the standard deviation of the current height bin is more than std_dev, the resolution gets multiplied by interp_factor.
		The elevation of a regular bin is simply a weighted average of elevation, weighted by density. (Denser photons have a stronger say in the final elevation).
		The elevation of a small bin is calculated by taking a percentile of elevation in that bin. The crev_depth_quantile percentile is the bin's final elevation.

		The process is different for DDA-bif (melt pond runs).
		'''
		# photon_signal = [delta_time, lon, lat, elev, distance, density]
		
		# Adjust density value by den_weight
		# photon_signal[:, 5] = photon_signal[:, 5]**den_weight # deprecated

		# Determine correct signal to use for interpolation
		if not meltpond_bool:
			# Normal DDA-ice-1 interpolation
			photon_signal = self.photon_signal_thresh
		else:
			# DDA bifurcate interpolation
			if top_bool:
				# interpolating the TOP surface
				photon_signal = self.photon_signal_top
			else:
				# interpolating the BOTTOM surface
				photon_signal = self.photon_signal_bot

		# Option for using maximum density for the bottom ground follower (not useful enough to include as an input param)
		max_dens_bool = False

		# We interpolate to points at every bin center. e.g. if interp_res is 5, we interpolate at 2.5, 7.5, 12.5,...
		min_photon_count = 1

		for b in np.arange(0, self.max_distance, interp_res):
			new_ground_points = np.empty(shape=(0, 8))  # array to fill and later append to ground_estimate
			bin_signal_endpoints = np.searchsorted(photon_signal[:, 4], [b, b + interp_res])
			bin_signal = photon_signal[bin_signal_endpoints[0]:bin_signal_endpoints[1], :]
			if len(bin_signal) < min_photon_count:  # if there fewer than 3 photons in this bin, skip it and move on to next bin
				continue

			bin_elev_stdev = np.std(bin_signal[:, 3])  # take standard deviation of elevation

			if bin_elev_stdev > std_dev:  # if the surface could be rough (crevasses)
				mini_bins = np.linspace(b, b + interp_res, num=int(interp_factor), endpoint=False)
				mini_interp_res = interp_res / interp_factor
				for mb in mini_bins:  # for each mini bin within the larger bin
					# mini_bin_signal = photon_signal[np.logical_and(photon_signal[:, 4] >= mb, photon_signal[:, 4] < (mb + mini_interp_res)), :]

					mini_bin_signal_endpoints = np.searchsorted(photon_signal[:, 4], [mb, mb + mini_interp_res])
					mini_bin_signal = photon_signal[mini_bin_signal_endpoints[0]:mini_bin_signal_endpoints[1], :]

					# New outlier detection code
					percentile = 10  # granularity of the threshold search
					while np.std(mini_bin_signal[:, 3]) > (std_dev * 2.5):  # check if there is a possibility for erroneous outliers
						(percentileLow, percentileHigh) = np.percentile(mini_bin_signal[:, 3], (percentile, 100 - percentile))  # filter out anomalous points far below the actual centroid we
						# filter out points significantly far below the target
						lowThreshold = mini_bin_signal[np.where(mini_bin_signal[:, 3] > percentileLow)]
						highThreshold = mini_bin_signal[np.where(mini_bin_signal[:, 3] < percentileHigh)]
						stdLow = np.std(lowThreshold[:, 3])
						stdHigh = np.std(highThreshold[:, 3])

						# choose the threshold that minimizes the variance
						if stdLow < stdHigh:
							mini_bin_signal = lowThreshold
						else:
							mini_bin_signal = highThreshold
						percentile /= 2  # reduce the percentile overtime to avoid overshooting

					if len(mini_bin_signal) < min_photon_count:  # if there are fewer than 3 photons in this mini_bin skip and move on to next mini_bin
						continue

					mini_bin_lon = np.average(mini_bin_signal[:, 1], weights=mini_bin_signal[:, 5] / np.sum(mini_bin_signal[:, 5]))
					mini_bin_lat = np.average(mini_bin_signal[:, 2], weights=mini_bin_signal[:, 5] / np.sum(mini_bin_signal[:, 5]))

					if meltpond_bool:
						if max_dens_bool:  # If the maximum density option is on
							max_dens_i = np.argmax(bin_signal[:, 5])  # max_density bin
							mini_bin_elev = mini_bin_signal[max_dens_i, 3]  # Elevation of photon with largest density value
						else:
							mini_bin_elev = np.percentile(mini_bin_signal[:, 3], mp_quantile * 100)

					else:
						mini_bin_elev = np.percentile(mini_bin_signal[:, 3], crev_depth_quantile * 100)

					mini_bin_time = np.average(mini_bin_signal[:, 0], weights=mini_bin_signal[:, 5] / np.sum(mini_bin_signal[:, 5]))

					mini_bin_distance = mb + (mini_interp_res / 2.)  # distance is center of mini_bin
					mini_bin_elev_stdev = np.std(mini_bin_signal[:, 3])  # stdev of elevations before weighting
					mini_bin_density_mean = np.mean(mini_bin_signal[:, 5])  # mean of mini_bin densities used in weighting
					# calculate a weighted standard deviation
					# stdev_weights = alpha*(e**(-beta*mini_bin_signal[:,5]))/np.sum(alpha*(e**(-beta*mini_bin_signal[:,5])))
					stdev_weights = (mini_bin_signal[:, 5]**(-1)) / np.sum(mini_bin_signal[:, 5]**(-1))  # normalized to 1
					mini_bin_weighted_stdev = np.sqrt(np.abs(np.sum(stdev_weights * (mini_bin_signal[:, 3] - np.mean(mini_bin_signal[:, 3]))**2) / (float(len(mini_bin_signal)) - 1)))
					# formula from http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf

					mini_bin_point = np.array([mini_bin_lon, mini_bin_lat, mini_bin_elev,
						mini_bin_distance, mini_bin_time, mini_bin_elev_stdev, mini_bin_density_mean,
						mini_bin_weighted_stdev])
					mini_bin_point.shape = (1, 8)
					# append to the new ground_points array (later appended to ground_estimate)
					# [lon, lat, elev, distance, elev_stdev, density_mean]
					new_ground_points = np.append(new_ground_points, mini_bin_point, axis=0)

			else:  # if elev_stdev is small (low topographic roughness)
				# bin_elev, bin_lon and bin_lat are weighted averages of the points used, weighted by density
				bin_lon = np.average(bin_signal[:, 1], weights=bin_signal[:, 5] / np.sum(bin_signal[:, 5]))
				bin_lat = np.average(bin_signal[:, 2], weights=bin_signal[:, 5] / np.sum(bin_signal[:, 5]))
				if meltpond_bool:
					if max_dens_bool:
						# Maximum density option
						max_dens_i = np.argmax(bin_signal[:, 5])  # max_density bin
						bin_elev = bin_signal[max_dens_i, 3]  # Elevation of photon with largest density value
					else:
						# Crevasse quantile option (default) on non-crevassed regions for dda-bif
						bin_elev = np.percentile(bin_signal[:, 3], mp_quantile * 100)

				else:
					# Regular dda-ice weights ground follower by density
					bin_elev = np.average(bin_signal[:, 3], weights=bin_signal[:, 5] / np.sum(bin_signal[:, 5]))

				bin_time = np.average(bin_signal[:, 0], weights=bin_signal[:, 5] / np.sum(bin_signal[:, 5]))

				bin_distance = b + (interp_res / 2.)  # distance is center of bin b
				bin_elev_stdev = np.std(bin_signal[:, 3])  # stdev of elevations used in weighting
				bin_density_mean = np.mean(bin_signal[:, 5])  # mean of bin densities used in weighting

				# calculate a weighted standard deviation
				# stdev_weights = alpha*(e**(-beta*bin_signal[:,5]))/np.sum(alpha*(e**(-beta*bin_signal[:,5])))
				stdev_weights = (bin_signal[:, 5]**(-1)) / np.sum(bin_signal[:, 5]**(-1))  # normalized to 1
				bin_weighted_stdev = np.sqrt(np.abs(np.sum(stdev_weights * (bin_signal[:, 3] - np.mean(bin_signal[:, 3]))**2) / (float(len(bin_signal)) - 1)))
				# formula from http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf

				bin_point = np.array([bin_lon, bin_lat, bin_elev, bin_distance, bin_time, bin_elev_stdev, bin_density_mean, bin_weighted_stdev])
				bin_point.shape = (1, 8)
				# append to new_ground_points (actually point in this case) array with
				# [lon, lat, elev, distance, elev_stdev, density_mean]
				new_ground_points = np.append(new_ground_points, bin_point, axis=0)

			# once we are done with bin b (possibly divided up into minibins),
			# append the new point (or possibly points) to the ground_estimate
			if not meltpond_bool:
				self.ground_estimate = np.append(self.ground_estimate, new_ground_points, axis=0)
			else:
				if top_bool:
					self.ground_estimate_top = np.append(self.ground_estimate_top, new_ground_points, axis=0)
				else:
					self.ground_estimate_bot = np.append(self.ground_estimate_bot, new_ground_points, axis=0)

	def interpolate_missing_data(self, interp_res):
		# TODO: refactor this function to NOT initialize empty array interpolated_ground_estimate
		# Instead, we should interpolate the points directly into self.ground_estimate
		prev_i = 0
		interpolated_ground_estimate = np.empty((0, self.ground_estimate.shape[1]))  # Setup the empty array 

		for i in range(self.ground_estimate.shape[0] - 1):  # For each self.ground_estimate point
			dist2nextPoint = self.ground_estimate[i + 1, 3] - self.ground_estimate[i, 3]
			if (dist2nextPoint > interp_res*2) and (dist2nextPoint < 200):  # If there are gaps larger than 'interp_res * 2' in the data (but less than 200 meters)
				interpolated_ground_estimate = np.append(interpolated_ground_estimate, self.ground_estimate[prev_i:i, :], axis=0)  # Add in the section before the gap
				prev_i = i

				# Interpolate the along track distance between endpoints of the gap by 'interp_res' meters
				endpoints_distance = [self.ground_estimate[i, 3], self.ground_estimate[i+1, 3]]
				interpolated_dist = np.arange(self.ground_estimate[i, 3], self.ground_estimate[i+1, 3], interp_res)

				# Not interpolate lon, lat, and elevation at those interpolated distances
				interpolated_lon = np.interp(interpolated_dist, endpoints_distance, [self.ground_estimate[i, 0], self.ground_estimate[i+1, 0]])  # lon
				interpolated_lat = np.interp(interpolated_dist, endpoints_distance, [self.ground_estimate[i, 1], self.ground_estimate[i+1, 1]])  # lat
				interpolated_elev = np.interp(interpolated_dist, endpoints_distance, [self.ground_estimate[i, 2], self.ground_estimate[i+1, 2]])  # elev

				# Fill bin_time, bin_elev_stdev, bin_density_mean, bin_weighted_stdev with nans because they can't be meaningfully interpolated
				interpolated_nans = np.empty((len(interpolated_dist), 4)) * np.nan

				# Stack all of the individual columns into a matrix. This interpolated_section matrix contains all data for the gap
				interpolated_section = np.hstack((interpolated_lon[:, None], interpolated_lat[:, None], interpolated_elev[:, None], interpolated_dist[:, None], interpolated_nans))

				# Append the interpolated_section to the previous data
				interpolated_ground_estimate = np.append(interpolated_ground_estimate, interpolated_section, axis=0)

		self.ground_estimate = np.append(interpolated_ground_estimate, self.ground_estimate[prev_i:self.ground_estimate.shape[0], :], axis=0)  # Add on the last segment

	def compute_thresholds_melt_pond(self, threshold_offset, quantile, binsize, mp_bin_h, mp_bin_v, density_histo_bool, histo_plot_bool, chunk_size):

		# photon_signal = self.photon_signal_thresh
		# photon_noise = self.photon_noise

		total_chunks = np.ceil((self.photon_signal_thresh[-1,4]-self.photon_signal_thresh[0,4])/chunk_size)
		self.logger.info('Number of Chunks: {}'.format(total_chunks))

		# Data comes as [delta_time, lon, lat, elev, distance, density]
		signal_mask = np.array([False] * len(self.photon_signal_thresh))  # stores values that passed thresholding and quantile
		self.signal_mask_top = np.array([False] * len(self.photon_signal_thresh))  # stores values that passed thresholding but not quantile ("blue points")
		self.signal_mask_bot = np.array([False] * len(self.photon_signal_thresh))

		prev_threshold = 0.  # store in case we have an empty signal_bool array

		# Boolens to track when yyou are in a pond 	and mark edges
		inpond_bool = 0
		start_bool = 1

		#Preallocate pond start and end arrays
		pond_edge_start = np.array([])
		pond_edge_end = np.array([])

		counter = -1 #chunk counter
		# Start chunking loop
		for i in np.arange(self.photon_signal_thresh[0,4],self.photon_signal_thresh[-1,4],chunk_size):

			start = time.time()
			# Bool for signal and noise in chunk 
			signal_bool = (self.photon_signal_thresh[:,4] >= i) & (self.photon_signal_thresh[:,4] < (i + chunk_size))
			noise_bool = (self.photon_noise[:,4] >= i) & (self.photon_noise[:,4] < (i + chunk_size))

			counter = counter +1
			self.logger.info('Chunk Number: {}'.format(counter))

			# If there is no signal in the hbin then continue
			if signal_bool.sum() == 0:
				continue

			# Downsample data for chunk
			photon_signal_temp = self.photon_signal_thresh[signal_bool,:]
			photon_noise_temp = self.photon_noise[noise_bool,:]
			signal_mask_temp = np.array([False] * len(photon_signal_temp))  # stores values that passed thresholding and quantile in chunck
			signal_mask_top_temp = np.array([False] * len(photon_signal_temp))  # stores values that passed thresholding but not quantile ("blue points") in chunk
			signal_mask_bot_temp = np.array([False] * len(photon_signal_temp))

			# STEP 1: Binning (by elevation or density)

			# Density histogram or elevation histogram?
			if density_histo_bool:
				# Determine if there are 2 surfaces to be found (Using Density Histo)
				f = photon_signal_temp[:, 5]
				neg_dens_bool = f <= 0
				f[neg_dens_bool] = 0.000000001  # min density for logarithm
				photon_signal_temp[:, 5] = np.log(photon_signal_temp[:, 5])  # use log of dens
				mp_bin_v = max(photon_signal_temp[:,5])/50 #50 density bins
				horizontal_edges = np.arange(min(photon_signal_temp[:, 4]), max(photon_signal_temp[:, 4]) + mp_bin_h, mp_bin_h)
				vertical_edges = np.arange(min(photon_signal_temp[:, 5]), max(photon_signal_temp[:, 5]) + 2 * mp_bin_v, mp_bin_v)  # The times 2 is to avoid edge effects
				H = np.histogram2d(photon_signal_temp[:, 4], photon_signal_temp[:, 5], bins=(horizontal_edges, vertical_edges))[0]
				H_filt = H.copy()
			else:
				# Determine if there are 2 surfaces to be found (Using Elev Histo)
				horizontal_edges = np.arange(min(photon_signal_temp[:, 4]), max(photon_signal_temp[:, 4]) + mp_bin_h, mp_bin_h)
				vertical_edges = np.arange(min(photon_signal_temp[:, 3]), max(photon_signal_temp[:, 3]) + 2 * mp_bin_v, mp_bin_v)# The times 2 is to avoid edge effects
				H = np.histogram2d(photon_signal_temp[:, 4], photon_signal_temp[:, 3], bins=(horizontal_edges, vertical_edges))[0]
				H_filt = H.copy()

			self.logger.info('Time elapsed creating histogram: {}'.format(time.time() - start))
			
			hlen = H.shape[0]
			vlen = H.shape[1]

			V_elev = vertical_edges[0:-1] + mp_bin_v / 2  # elevation refernce for each vertical bin
			start_d = photon_signal_temp[0, 4]

			start = time.time()
			for hbin in range(hlen):
				
				# STEP 2: Binomial filter: smooth out high-frequency signals
				for vbin in range(vlen):
					if vbin == 0:
						H_filt[hbin, vbin] = H[hbin, vbin] * 0.375 + H[hbin, vbin + 1] * 0.25 + H[hbin, vbin + 2] * 0.0625
					elif vbin == 1:
						H_filt[hbin, vbin] = H[hbin, vbin - 1] * 0.25 + H[hbin, vbin] * 0.375 + H[hbin, vbin + 1] * 0.25 + H[hbin, vbin + 2] * 0.0625
					elif vbin == vlen - 2:
						H_filt[hbin, vbin] = H[hbin, vbin - 2] * 0.0625 + H[hbin, vbin - 1] * 0.25 + H[hbin, vbin] * 0.375 + H[hbin, vbin + 1] * 0.25
					elif vbin == vlen - 1:
						H_filt[hbin, vbin] = H[hbin, vbin - 2] * 0.0625 + H[hbin, vbin - 1] * 0.25 + H[hbin, vbin] * 0.375
					else:
						H_filt[hbin, vbin] = H[hbin, vbin - 2] * 0.0625 + H[hbin, vbin - 1] * 0.25 + H[hbin, vbin] * 0.375 + H[hbin, vbin + 1] * 0.25 + H[hbin, vbin + 2] * 0.0625


				# STEP 3: Find peaks (via sigmaveg algo step 2c)
				ipks, properties = scipy.signal.find_peaks(H_filt[hbin, :], height=2, prominence=2)
				pks_sort = np.argsort(properties["peak_heights"])
				ipks_sort = ipks[pks_sort]
				plen = len(ipks)


				# Plot histograms?
				if histo_plot_bool:
					if density_histo_bool:  # density histos
						seg = np.round(start_d + hbin * mp_bin_h)
						plt.figure('Histogram')

						ax = plt.subplot(111)
						box = ax.get_position()
						ax.set_position([.05, .15, .91, .8])

						# Plot only the two largest peaks
						if len(ipks_sort) > 2:
							ipks_2 = ipks_sort[-2:]
						else:
							ipks_2 = ipks_sort

						ax.barh(V_elev, H_filt[hbin, :])
						ax.plot(H_filt[hbin, ipks_2], V_elev[ipks_2], 'r.', markersize=12, label='Peaks')
						plt.title('Log-Density Histogram (Start: ' + str(seg) + ' m, width: ' + str(mp_bin_h) + ' m)', fontsize=25)
						plt.xlabel('Counts', fontsize=20)
						plt.ylabel('Logarithm of Density', fontsize=20)
						ax.tick_params(labelsize=20)

						plt.savefig(os.path.join(self.plotdir,'Fig4_density_histogram_{}_{}.png'.format(seg,mp_bin_h)), bbox_inches='tight')
						plt.clf()
						plt.close()

					else: # elevation histo
						seg = np.round(start_d + hbin * mp_bin_h)
						plt.figure('Histogram')

						ax = plt.subplot(111)
						box = ax.get_position()
						ax.set_position([.05, .15, .91, .8])

						# Plot only the two largest peaks
						if len(ipks_sort) > 2:
							ipks_2 = ipks_sort[-2:]
						else:
							ipks_2 = ipks_sort

						ax.barh(V_elev, H_filt[hbin, :],height=0.1)
						#ax.plot(H_filt[hbin, ipks_2], V_elev[ipks_2], 'r.', markersize=12, label='Peaks') # plot 2 peaks
						ax.plot(H_filt[hbin, ipks], V_elev[ipks], 'r.', markersize=12, label='Peaks') # plot all peaks
						plt.title('Vertical Histogram (Start: ' + str(seg) + ' m, width: ' + str(mp_bin_h) + ' m)', fontsize=25)
						plt.xlabel('Photon Counts', fontsize=20)
						plt.ylabel('Elevation (m)', fontsize=20)
						ax.tick_params(labelsize=20)

						plt.savefig(os.path.join(self.plotdir,'Fig4_elevation_histogram_{}_{}.png'.format(seg,mp_bin_h)), bbox_inches='tight')
						plt.clf()
						plt.close()

				if density_histo_bool:
					photon_signal_temp[:, 5] = f

				# If there are two surfaces
				if plen > 1:

					# 2 largset peaks
					pks_max = ipks_sort[plen - 1]
					pks_max2 = ipks_sort[plen - 2]
					i1 = np.minimum(pks_max, pks_max2)  # bottom surface
					i2 = np.maximum(pks_max, pks_max2)  # top surface

					# Identify slabs
					pk_min = i1 + np.argmin(H_filt[hbin, i1:i2])  # Find min index between peaks
					bin_above = i2 + (pk_min - i1)
					bin_below = i1

					while H_filt[hbin, bin_below] > 1:
						bin_below = bin_below - 1
						if bin_below < 0:
							bin_below = 0
							break

					if bin_above >= vlen:
						bin_above = vlen - 1

					for b in range(int(hbin * mp_bin_h), int((hbin + 1) * mp_bin_h), int(binsize)):
						# make boolean arrays to subset into horizontal bins
						signal_bool1 = np.logical_and(photon_signal_temp[:, 4] >= start_d + b, photon_signal_temp[:, 4] < (start_d + b + binsize))

						if density_histo_bool:
							## Density
							signal_bool2 = np.logical_and(photon_signal_temp[:, 5] <= V_elev[bin_above], photon_signal_temp[:, 5] > V_elev[pk_min])
							signal_bool3 = np.logical_and(photon_signal_temp[:, 5] > V_elev[bin_below], photon_signal_temp[:, 5] <= V_elev[pk_min])
						else:
							# Elevation
							signal_bool2 = np.logical_and(photon_signal_temp[:, 3] <= V_elev[bin_above], photon_signal_temp[:, 3] > V_elev[pk_min])
							signal_bool3 = np.logical_and(photon_signal_temp[:, 3] > V_elev[bin_below], photon_signal_temp[:, 3] <= V_elev[pk_min])

						# Signal photon for the 2 Layers
						signal_bool_top_temp = np.logical_and(signal_bool1, signal_bool2)
						signal_bool_bot_temp = np.logical_and(signal_bool1, signal_bool3)

						# OR the existing mask (starts as False) and the current signal boolean
						signal_mask_top_temp = np.logical_or(signal_mask_top_temp, signal_bool_top_temp)
						signal_mask_bot_temp = np.logical_or(signal_mask_bot_temp, signal_bool_bot_temp)

				
					#Mark the start of the pond
					if start_bool:
						pond_edge_start = np.append(pond_edge_start,start_d + int(hbin * mp_bin_h)-binsize)
						start_bool = 0
					inpond_bool = 1


				else:  # 1 surface (regular thresholding procedure)
				
					if inpond_bool:
						pond_edge_end = np.append(pond_edge_end, start_d+int(hbin * mp_bin_h))
						inpond_bool = 0
					start_bool = 1

					for b in range(int(hbin * mp_bin_h), int((hbin + 1) * mp_bin_h), int(binsize)):
						# make boolean arrays to subset into horizontal bins
						signal_bool2 = np.logical_and(photon_signal_temp[:, 4] >= start_d + b, photon_signal_temp[:, 4] < (start_d + b + binsize))
						noise_bool2 = np.logical_and(photon_noise_temp[:, 4] >= b, photon_noise_temp[:, 4] < (b + binsize))

						bin_signal = photon_signal_temp[signal_bool2, :]
						bin_noise = photon_noise_temp[noise_bool2, :]

						# threshold value for current bin is max noise density value + offset
						if bin_noise.size == 0:
							# if nothing in noise slab, use threshold offset (NOT previous threshold)
							threshold = threshold_offset
						else:
							threshold = np.max(bin_noise[:, 5]) + threshold_offset

						# also add constraint that the signal must be above threshold
						signal_bool2 = np.logical_and(signal_bool2, photon_signal_temp[:, 5] > threshold)


						# calculate the value of the q*100% quantile
						if signal_bool2.sum() == 0:
							quantile_value = 100000
						else:
							quantile_value = np.percentile(photon_signal_temp[signal_bool2, 5], quantile * 100)

						# take only values that pass the quantile test
						signal_bool2 = np.logical_and(signal_bool2, photon_signal_temp[:, 5] > quantile_value)

						# OR the existing mask (starts as False) and the current signal boolean
						signal_mask_temp = np.logical_or(signal_mask_temp, signal_bool2)

				# Combine masks (2 layers + 1 layer)
				signal_mask_top_temp = np.logical_or(signal_mask_top_temp, signal_mask_temp)
				signal_mask_bot_temp = np.logical_or(signal_mask_bot_temp, signal_mask_temp)

			# Combine chunk masks with full mask
			signal_mask[signal_bool] = signal_mask_temp
			self.signal_mask_top[signal_bool] = signal_mask_top_temp
			self.signal_mask_bot[signal_bool] = signal_mask_bot_temp

			self.logger.info('Time elapsed identifying signal: {}'.format(time.time() - start))
		# If the end of the pond is cutoff then use last pond as pond end
		if len(pond_edge_start) > len(pond_edge_end):
			pond_edge_start = pond_edge_start[:-1]

		# Combine edge starts and ends into a single variable
		self.pond_edges = np.column_stack((pond_edge_start, pond_edge_end))

	def set_top_and_bottom_signal(self):

		self.photon_signal_top = self.photon_signal_thresh[self.signal_mask_top, :]
		self.photon_signal_bot = self.photon_signal_thresh[self.signal_mask_bot, :]

	def correct_ponds(self):
		# Format of top ground estimate:\n[bin_lon, bin_lat, bin_elev, bin_distance, bin_time, bin_elev_stdev, bin_density_mean, bin_weighted_stdev]

		# PREVIOUSLY: saturation_data (as a parameter) --> NOW: self.velocity_data

		# Will also add pond statistics here (width and mean depth) and add to pond_edges
		pond_widths = np.zeros([len(self.pond_edges),1])
		pond_mean_depths = np.zeros([len(self.pond_edges),1])
		pond_max_depths = np.zeros([len(self.pond_edges),1])
		mean_sats = np.zeros([len(self.pond_edges),1])
		good_pond_index = np.empty(0,dtype=int) #Indexes to delete from pond edges when bad pond is found

		# Correct pond edges
		for i in range(len(self.pond_edges)):
			
			# Find pond edge indices
			start_temp = self.pond_edges[i,0]
			abs_top_left_temp = np.abs(self.ground_estimate_top[:,3]-start_temp)
			index_top_left_temp = abs_top_left_temp.argmin()
			abs_bot_left_temp = np.abs(self.ground_estimate_bot[:,3]-start_temp)
			index_bot_left_temp = abs_bot_left_temp.argmin()

			end_temp = self.pond_edges[i,1]
			abs_top_right_temp = np.abs(self.ground_estimate_top[:,3]-end_temp)
			index_top_right_temp = abs_top_right_temp.argmin()
			abs_bot_right_temp = np.abs(self.ground_estimate_bot[:,3]-end_temp)
			index_bot_right_temp = abs_bot_right_temp.argmin()

			# Get rid of false positive ponds. 3 Cases.
			# Case 1: Top surface is a spike in elev. Remove if mean top is higher than mean pond edges (over 20cm = delta_elev)
			surf_temp = self.ground_estimate_top[index_top_left_temp +1:index_top_right_temp,2]
			mean_top = np.mean(surf_temp)
			mean_edges = (np.mean(self.ground_estimate_top[index_top_left_temp-2:index_top_left_temp+1,2])+ np.mean(self.ground_estimate_top[index_top_right_temp:index_top_right_temp+3,2]))/2
			delta_elev = 0.2 #meters. Threshold parameter for significant difference in individual pond surface height from mean pond surface height
			# Case 2: The two pond edges are too different in height (delta_diff) - small ``ponds" only (less than 4 points)
			edge_diff = np.abs(self.ground_estimate_top[index_top_left_temp,2] - self.ground_estimate_top[index_top_right_temp,2])
			delta_diff = 1 #meters. THreshold parameter for significant difference in pond edge heights

			# Check the two cases that involve the pond surface height. 
			# If true replace top surface (flase signal) with bottom surface (true signal)
			if np.logical_or(mean_top > (mean_edges + delta_elev), edge_diff > delta_diff):
				# Get bottom surface across the false pond segment
				bot_pond_temp = self.ground_estimate_bot[index_bot_left_temp:index_bot_right_temp+1,:]

				# Replace previous pond top with pond bottom pond bottom
				self.ground_estimate_top = np.delete(self.ground_estimate_top,slice(index_top_left_temp,index_top_right_temp+1),axis=0)
				self.ground_estimate_top = np.insert(self.ground_estimate_top,index_top_left_temp,bot_pond_temp,0)

				continue #removed pond so skip to the next one

			# Case 3: minimum width to be considered a pond
			min_w_i = 4 # value of 4 implies at least 3 depth measurements (with 2 edges)
			if (index_top_right_temp-index_top_left_temp) <= min_w_i:
				# Set bot equal to top in the case where the pond is not wide enough
				top_pond_temp = self.ground_estimate_top[index_top_left_temp:index_top_right_temp+1,:]
				self.ground_estimate_bot = np.delete(self.ground_estimate_bot,slice(index_bot_left_temp,index_bot_right_temp+1),axis=0)
				self.ground_estimate_bot = np.insert(self.ground_estimate_bot,index_bot_left_temp,top_pond_temp,0)
				continue

			# Mean pond surface height (interior points only), with removed outliers via pond elevation greater than delta_top off mean surf height
			delta_top = 0.25 #m
			surf_temp2 = surf_temp[np.abs(surf_temp - mean_top) < delta_top]
			pond_top_temp = np.mean(surf_temp2)

			# Make all pond top heights the same (make pond surface)
			self.ground_estimate_top[index_top_left_temp+1:index_top_right_temp,2] = pond_top_temp


			### Incomplete Code below to correct the pond edges ###
			# Correct left side of pond
			
			# Alternate height correcttion using two adjacent heights to edge
			#pond_top_left_temp = np.mean([self.ground_estimate_top[index_top_temp +1,2],self.ground_estimate_top[index_top_temp +2,2]])
			#pond_top__right_temp = np.mean([self.ground_estimate_top[index_top_temp -1,2],self.ground_estimate_top[index_top_temp -2,2]])
			
			#Extend top to edge by looking for edge of pond where elevation rises above the pond top
			#while self.ground_estimate_top[index_top_left_temp,2] < pond_top_temp and index_top_left_temp >= 0:
			#	self.ground_estimate_top[index_top_left_temp,2] = pond_top_temp
			#	index_top_left_temp = index_top_left_temp-1
			# Save corrected edge
			#start_index = index_top_left_temp +2
			#self.pond_edges[i,0] = self.ground_estimate_top[start_index,3]

			# Correct right side of pond
			#Extend top to edge by looking for edge of pond where elevation rises above the pond top
			#while self.ground_estimate_top[index_top_right_temp,2] < pond_top_temp and index_top_right_temp < len(self.ground_estimate_top)-1:
			#	self.ground_estimate_top[index_top_right_temp,2] = pond_top_temp
			#	index_top_right_temp = index_top_right_temp+1

			# Save corrected edge
			#end_index = index_top_right_temp
			#self.pond_edges[i,1] = self.ground_estimate_top[end_index,3]

			# Add edge points to bottom surface too (if applicable)
			#Left end
			#abs_bot_left = np.abs(self.ground_estimate_bot[:,3]-self.ground_estimate_top[start_index,3])
			#index_bot_left = abs_bot_left.argmin()
			# if np.min(abs_bot_left) != 0:
			# 	if self.ground_estimate_bot[index_bot_left,3] >  self.ground_estimate_top[start_index,3]:
			# 		add_index = index_bot_left
			# 	else:
			# 		add_index = index_bot_left + 1

			# 	self.ground_estimate_bot = np.insert(self.ground_estimate_bot,add_index,self.ground_estimate_top[start_index,:],0)
			# else:
			# 	self.ground_estimate_bot[index_bot_left,:] = self.ground_estimate_top[start_index,:]


			#Right end
			#abs_bot_right = np.abs(self.ground_estimate_bot[:,3]-self.ground_estimate_top[end_index,3]) 
			#index_bot_right = abs_bot_right.argmin()
			# if np.min(abs_bot_right) != 0:
			# 	if self.ground_estimate_bot[index_bot_right,3] >  self.ground_estimate_top[end_index,3]:
			# 		add_index = index_bot_right
			# 	else:
			# 		add_index = index_bot_right + 1

			# 	self.ground_estimate_bot = np.insert(self.ground_estimate_bot,add_index,self.ground_estimate_top[end_index,:],0)
			# else:
			# 	self.ground_estimate_bot[index_bot_right,:] = self.ground_estimate_top[end_index,:]

			### END Incomplete edge code ###

			# Add width and depth
			if self.ground_estimate_bot[index_bot_left_temp:index_bot_right_temp+1,2].size > 0:
				pond_widths[i] = self.ground_estimate_top[index_top_right_temp,3] - self.ground_estimate_top[index_top_left_temp,3]
				pond_mean_depths[i] = pond_top_temp - np.mean(self.ground_estimate_bot[index_bot_left_temp+1:index_bot_right_temp,2]) # Only for points on interior (with non-zero depth). Use mean surf height since surfs have different resolutions
				pond_max_depths[i] = pond_top_temp - np.amin(self.ground_estimate_bot[index_bot_left_temp+1:index_bot_right_temp,2]) # Only for points on interior (with non-zero depth). Use mean surf height since surfs have different resolutions

				#Calculate sat flag mean
				#sat_bool = np.logical_and((saturation_data[:,0] >= self.ground_estimate_bot[index_bot_left+1,4]), (saturation_data[:,0]< self.ground_estimate_bot[index_bot_right,4]))
				#if sat_bool.sum() == 0:
				#	mean_sat = 0
				#else:
				#	mean_sat = np.max(saturation_data[sat_bool,3])
				#sat_tol = 0.8 # Tolerance for saturation
				#mean_sats[i] = mean_sat

				# Mark that this was a good pond if depth > min_depth (after pulse quick fix) and the edge difference tol still holds
				min_depth = 0.5
				if (pond_max_depths[i] > min_depth): 
					good_pond_index = np.append(good_pond_index,i) # good_pond_index tracks the indices of valid ponds. Only ponds that made it this far are considered valid.

				else:
					# Set bottom surface equal to top surface in this case of afterpulse effects
					top_pond_temp = self.ground_estimate_top[index_top_left_temp-1:index_top_right_temp+1,:]
					self.ground_estimate_bot = np.delete(self.ground_estimate_bot,slice(index_bot_left_temp-1,index_bot_right_temp+2),axis=0)
					self.ground_estimate_bot = np.insert(self.ground_estimate_bot,index_bot_left_temp,top_pond_temp,0)

		# append pond stats to self.pond_edges structure as columns
		self.pond_edges = np.hstack((self.pond_edges,pond_widths))
		self.pond_edges = np.hstack((self.pond_edges,pond_mean_depths))
		self.pond_edges = np.hstack((self.pond_edges,pond_max_depths))
		self.pond_edges = np.hstack((self.pond_edges,mean_sats))

		# Remove bad ponds officially based on the good_pond_index
		if good_pond_index.size ==0:
			self.pond_edges = np.empty(0)
		else:
			self.pond_edges = self.pond_edges[good_pond_index,:]
			# Remove nan ponds
			self.pond_edges = self.pond_edges[~np.isnan(self.pond_edges[:,3])]





class ParamValueError(Exception):
	def __init__(self, value):
		self.value = value
		print('Invalid parameter value:', self.value)

def initialize_func_globals():
	pass

def determine_beam_strength(filepath, channel):
	'''
	Determines the strength of the given channel for the given filepath. This only works for ATLAS.
	'''
	f = h5py.File(filepath, 'r')
	channelInfo = f['/' + channel]
	beamStrength = channelInfo.attrs['atlas_beam_type']
	return beamStrength.decode('UTF-8')


def subset_data_from_polygon(poly, photon_data, crs):
	'''
	Chops the photon_data down to only inside the polygon shape.
	'''
	# Turn photon data into geopandas dataframe
	downsampleRate = 1000  # Downsample photons at start and upsample indicies at end
	photonLine = gpd.GeoDataFrame(geometry=gpd.points_from_xy(photon_data[0::downsampleRate, 1], photon_data[0::downsampleRate, 2]))
	photonLineLatLon = photonLine.set_crs('EPSG:4326')  # Set as regular lat/lon for WGS84
	photonLineProjected = photonLineLatLon.to_crs(crs)  # Project to proper coordinate reference system (determined from location_to_polygon)

	isIntersect = []
	for point in photonLineProjected.iterrows():  # Iterate through the rows (and preserve data type)
		isIntersect.append(poly.contains(point[1][0]).bool())  # Does polygon contain point? (stupid formatting due to geopandas and shapely)

	if any(isIntersect):
		intersectionInds = [ind for ind, value in enumerate(isIntersect) if value is True]  # Get a list of indicies of true values
		startInd = intersectionInds[0] * downsampleRate
		endInd = intersectionInds[-1] * downsampleRate
		return photon_data[startInd:endInd, :]
	else:
		return None


def location_to_polygon(location):
	'''
	Takes a string input of a location and returns the corresponding shapely polygon object.
	'''
	if location in ['negri', 'Negri', 'negribreen', 'Negribreen']:
		file = 'geojson/Negribreen_poly.JSON'
		crs = 'EPSG:32633'
	elif location in ['jak', 'Jak', 'jakobshavn', 'Jakobshavn']:
		file = 'geojson/Jakobshavn_poly.JSON'
		crs = 'EPSG:32622'
	elif location in ['bering', 'Bering', 'bbgs', 'BBGS']:
		file = 'geojson/BBGS_poly.JSON'
		crs = 'EPSG:32606'
	elif location in ['colorado', 'Colorado', 'co', 'CO', 'rockies', 'Rockies']:
		file = 'geojson/Colorado_mountains_poly.JSON'
		crs = 'EPSG:32613'
	elif location in ['greenland', 'Greenland']:
		file = 'geojson/Greenland_poly.JSON'
		crs = 'EPSG:3413'
	elif location in ['Lincoln','lincoln','seaice','MYI2020']:
		file = 'geojson/MYIRegion20Aug_poly.JSON'
		crs = 'EPSG:3413'
	elif location in ['Peterman', 'peterman', 'Pman', 'pman']:
		file = 'geojson/Peterman.JSON'
		crs = 'EPSG:3413'
	elif location in ['Muldrow','muld','mul','muldrow','mdw']:
		file = 'geojson/Muldrow.JSON'
		crs = 'EPSG:3338'
	elif location in ['Nathorstbreen','Nath','nathbreen','nath','nathorstbreen','nathor']:
		file = 'geojson/Nathorstbreen.JSON'
		crs = 'EPSG:23031'
	elif location in ['Greenland_small', 'greenland_small', 'GS', 'gs']:
		file = 'geojson/Greenland_center_poly.JSON'
		crs = 'EPSG:32622'
	else:
		return None, None

	regionShapeFile = gpd.read_file(file)
	projectedRegionShapeFile = regionShapeFile.to_crs(crs)
	return projectedRegionShapeFile, crs


def density_worker_3(pt, sigma, cutoff, DE_shared, tree):
	neighbors = tree.query_ball_point(DE_shared[pt, :], r=sigma * cutoff, p=2, workers=1)  # calculate neighbors within radius
	neighbors = np.array(neighbors)
	idx = neighbors[neighbors != pt]
	distances = np.linalg.norm(DE_shared[idx, :] - DE_shared[pt, :], axis=1)  # Euclidean distance norm along axis=1
	weight_sum = np.sum(gaussian.pdf(distances, loc=0, scale=sigma))  # sum up pdf values
	return weight_sum

def handle_bool_params(value, name):

	if value.lower() == 'true' or value.lower() == 't' or value == '1':
		return True
	elif value.lower() == 'false' or value.lower() == 'f' or value == '0':
		return False
	else:
		raise Exception('{} must be "True" or "False"'.format(name))








######################### OLD / UNUSED functions below #########################


"""
def assign_slabs_by_ridge_detection(photon_data, pixel_dimensions, slab_thickness, run_name, cloud_tolerance):
	'''
	Assigns signal slab and noise slab by finding ridges (better for bad data).
	'''
	# Data format: delta_time, longitude, latitude, elevation, distance
	# 				  0				1		2			3		4
	pix_x, pix_y = pixel_dimensions
	horizontal_edges = np.arange(min(photon_data[:, 4]), max(photon_data[:, 4]) + pix_x, pix_x)
	vertical_edges = np.arange(min(photon_data[:, 3]), max(photon_data[:, 3]) + pix_y, pix_y)
	H = np.histogram2d(photon_data[:, 3], photon_data[:, 4], bins=(vertical_edges, horizontal_edges))[0]

	for col in range(H.shape[1]):
		subset = H[:, col]
		ind = np.where(subset != 0)[0]
		if np.size(ind) != 0:
			first, last = ind[0], ind[-1]
			if last - first <= 5:
				sb = 0
			else:
				sb = 5
			H[:(first + sb), col] = H[(first + sb), col]
			H[(last - sb):, col] = H[(last - sb), col]

	Hyy = find_ridges(H, pix_y, sigma=(0.5, 0.75))  # Hyy is the second derivative in the vertical direction
	# Plot the histogram and associated second derivative
	X, Y = np.meshgrid(horizontal_edges, vertical_edges)

	slab_center_locations = []
	for i in range(len(horizontal_edges) - 1):
		slab_center_locations .append((horizontal_edges[i] + horizontal_edges[i + 1]) / 2)

	plt.figure(figsize=(20, 10))
	histax = plt.subplot(211)
	histax.set_position([.05, .525, .85, .45])
	histoplot = plt.pcolormesh(X, Y, H, cmap='jet')
	plt.title('Histogram of Dataset')
	plt.xlim([np.min(X), np.max(X)])
	plt.ylim([np.min(Y), np.max(Y)])
	hist_cbax = plt.axes([.92, .525, .02, .45])
	plt.colorbar(histoplot, cax=hist_cbax)
	hyyax = plt.subplot(212)
	hyyax.set_position([.05, .025, .85, .45])
	hyyplot = plt.pcolormesh(X, Y, Hyy, cmap='jet')
	plt.title(r'Second Derivative $H_{yy}(x,y)$')
	plt.xlim([np.min(X), np.max(X)])
	plt.ylim([np.min(Y), np.max(Y)])
	hyy_cbax = plt.axes([.92, .025, .02, .45])
	plt.colorbar(hyyplot, cax=hyy_cbax)
	plt.savefig('output/' + str(run_name) + '/histogram_Hyy.png')

	min_vertical_bins = np.argmin(Hyy, axis=0)  # minimum second derivative bin along track
	vertical_bin_centers = (vertical_edges[1:] + vertical_edges[0:-1]) / 2.
	signal_slab_center = vertical_bin_centers[min_vertical_bins]

	signal_slab_center[np.min(Hyy, axis=0) > cloud_tolerance] = -9999  # replace locations with bad cloud cover with -9999 (high Hyy values)

	signal_mask = False * len(photon_data)
	noise_mask = False * len(photon_data)

	for hbin in range(len(horizontal_edges) - 1):
		hbin_subset = np.logical_and(photon_data[:, 4] > horizontal_edges[hbin], photon_data[:, 4] < horizontal_edges[hbin + 1])

		hbin_signal_mask = np.logical_and(hbin_subset, np.abs(photon_data[:, 3] - signal_slab_center[hbin]) < .5 * slab_thickness)
		hbin_noise_mask = np.logical_and(hbin_subset, np.abs(photon_data[:, 3] - signal_slab_center[hbin] - slab_thickness) < .5 * slab_thickness)

		# hbin_noise_mask = np.logical_and(hbin_subset, np.abs(photon_data[:,3] - signal_slab_center[hbin]) > 1.5*slab_thickness)

		signal_mask = np.logical_or(signal_mask, hbin_signal_mask)
		noise_mask = np.logical_or(noise_mask, hbin_noise_mask)

	return signal_mask, noise_mask



# ### Starmap without neighbors (FAST!!!), but high memory usage ###
def compute_density_parallel_original(data, sigma, cutoff, aniso_factor, logger):
	'''
	Computes the density for every single photon point. Uses a RBF with a kernel described by sigma, cutoff, and aniso_factor.
	'''
	# Data comes as [delta_time, lon, lat, elev, distance]
	DE = np.array([data[:, 4] / aniso_factor, data[:, 3]]).T  # Distance and Elevation

	start = time.time()
	tree = cKDTree(DE, balanced_tree=False)  # make the KDTree to use for rangesearch. Balanced_tree=False has faster query times
	logger.info('Time elapsed creating cKDTree: {}'.format(time.time() - start))
	start = time.time()
	neighbors = tree.query_ball_tree(tree, r=sigma * cutoff, p=2)  # calculate neighbors within radius
	logger.info('Time elapsed finding neighbors within sigma*cutoff: {}'.format(time.time() - start))

	start = time.time()
	density = np.array([])

	idxs = [np.array(neighbors[pt])[np.array(neighbors[pt]) != pt] for pt in range(len(DE))]
	with mp.Pool(processes=mp.cpu_count() - 1) as pool:  # Create the multiprocessing pool
		worker_args_iter = [(pt, idx, sigma, DE) for pt, idx in zip(range(len(DE)), idxs)]
		density = pool.starmap(density_worker_original, worker_args_iter)
		density = np.asarray(density)
		logger.info('Time elapsed calculating density for each point: {}'.format(time.time() - start))
		return density


def density_worker_original(pt, idx, sigma, DE):
	distances = np.linalg.norm(DE[idx, :] - DE[pt, :], axis=1)  # Euclidean distance norm along axis=1
	weight_sum = np.sum(gaussian.pdf(distances, loc=0, scale=sigma))  # sum up pdf values
	return weight_sum


# Helpful multiprocessing shared memory questions
# https://stackoverflow.com/questions/423379/using-global-variables-in-a-function
# https://stackoverflow.com/questions/39322677/python-how-to-use-value-and-array-in-multiprocessing-pool
# https://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-multiprocessing/5550156#5550156
# https://stackoverflow.com/questions/10721915/shared-memory-objects-in-multiprocessing


# ### Starmap with shared array and without neighbors (FAST!!! and memory stable!!!!!) ###
def compute_density_parallel_2(data, sigma, cutoff, aniso_factor, logger):
	'''
	Computes the density for every single photon point. Uses a RBF with a kernel described by sigma, cutoff, and aniso_factor.
	'''
	# Data comes as [delta_time, lon, lat, elev, distance]
	DE = np.array([data[:, 4] / aniso_factor, data[:, 3]]).T  # Distance and Elevation

	start = time.time()
	tree = cKDTree(DE, balanced_tree=False)  # make the KDTree to use for rangesearch. Balanced_tree=False has faster query times
	logger.info('Time elapsed creating cKDTree: {}'.format(time.time() - start))

	start = time.time()
	neighbors = tree.query_ball_tree(tree, r=sigma * cutoff, p=2)  # calculate neighbors within radius
	logger.info('Time elapsed finding neighbors within sigma*cutoff: {}'.format(time.time() - start))

	start = time.time()
	shared_array_base = mp.Array(ctypes.c_double, DE.flatten(), lock=False)
	DE_shared = np.ctypeslib.as_array(shared_array_base)
	DE_shared = DE_shared.reshape(DE.shape)

	idxs = [np.array(neighbors[pt])[np.array(neighbors[pt]) != pt] for pt in range(len(DE))]
	density = np.array([])
	with mp.Pool(processes=mp.cpu_count() - 1, maxtasksperchild=10) as pool:  # Create the multiprocessing pool. Each processor kills itself and restarts after maxtasksperchild iterations
		worker_args_iter = [(pt, idx, sigma, DE_shared) for pt, idx in zip(range(len(DE)), idxs)]
		density = pool.starmap(density_worker_2, worker_args_iter)
		density = np.asarray(density)
		logger.info('Time elapsed calculating density for each point: {}'.format(time.time() - start))
		return density


def density_worker_2(pt, idx, sigma, DE_shared):
	distances = np.linalg.norm(DE_shared[idx, :] - DE_shared[pt, :], axis=1)  # Euclidean distance norm along axis=1
	weight_sum = np.sum(gaussian.pdf(distances, loc=0, scale=sigma))  # sum up pdf values
	return weight_sum


def compute_density_parallel_ray(data, sigma, cutoff, aniso_factor, logger):
	'''
	This function attemps to use the 'ray' package to get better speed/memory usage when using multiprocessors with a shared object.
	'''
	# Data comes as [delta_time, lon, lat, elev, distance]
	DE = np.array([data[:, 4] / aniso_factor, data[:, 3]]).T  # Distance and Elevation
	DE_shared = ray.put(DE)  # Store the array in shared memory

	start = time.time()
	tree = cKDTree(DE, balanced_tree=False)  # make the KDTree to use for rangesearch. Balanced_tree=False has faster query times
	logger.info('Time elapsed creating cKDTree: {}'.format(time.time() - start))
	start = time.time()
	neighbors = tree.query_ball_tree(tree, r=sigma*cutoff, p=2)  # calculate neighbors within radius
	logger.info('Time elapsed finding neighbors within sigma*cutoff: {}'.format(time.time() - start))


	start = time.time()
	density = np.array([])

	idxs = [np.array(neighbors[pt])[np.array(neighbors[pt]) != pt] for pt in range(len(DE))]
	density = ray.get([density_worker_ray.remote(pt, idx, sigma, DE_shared) for pt, idx in zip(range(len(DE)), idxs)])
	logger.info('Time elapsed calculating density for each point: {}'.format(time.time() - start))
	return density


# @ray.remote
def density_worker_ray(pt, idx, sigma, DE_shared):
	print('density worker')
	distances = np.linalg.norm(DE_shared[idx, :] - DE_shared[pt, :], axis=1)  # Euclidean distance norm along axis=1
	weight_sum = np.sum(gaussian.pdf(distances, loc=0, scale=sigma))  # sum up pdf values
	return weight_sum



def find_ridges(f, h, sigma):
	'''
	Uses a central difference formula to calculate the second derivative of a 2d histogram. 
	'''
	# Smooth histogram data using Gaussian kernel
	f = filters.gaussian(f, sigma=sigma, mode='mirror')
	# Calculate second derivative in y direction for each pixel, using 5pt central difference
	# for interior points and forward and backward differences for edges
	n, m = f.shape
	fyy = np.zeros(shape=(n, m))
	for i in range(n):
		for j in range(m):
			if i > n - 3:  # if we are at bottom edge
				fyy[i, j] = (2 * f[i, j] - 5 * f[i - 1, j] + 4 * f[i - 2, j] - f[i - 3, j]) / pow(h, 3)
			elif i < 2:  # if we are at top edge
				fyy[i, j] = (2 * f[i, j] - 5 * f[i + 1, j] + 4 * f[i + 2, j] - f[i + 3, j]) / pow(h, 3)
			else:  # if we are in the middle somewhere, do central difference
				fyy[i, j] = (-f[i - 2, j] + 16 * f[i - 1, j] - 30 * f[i, j] + 16 * f[i + 1, j] - f[i + 2, j]) / (12 * pow(h, 2))
	return fyy
"""


