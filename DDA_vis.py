from DDA_func import *
import os
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Some global variables for plot cosmetics
fig_dims = (20, 8)
pt_size = 3

# TODO: make plotting class that inherits from DDAice class

def plot_subset_granule_over_region(DDA):
	'''
	Plots the UTM projected shapefile for the proper region with the subset part of the ATLAS track on top.
	'''

	# Turn photon data into geopandas dataframe, downsample, and project into proper CRS
	downsampleRate = 1000  # Downsample photons at start and upsample indicies at end
	photonLine = gpd.GeoDataFrame(geometry=gpd.points_from_xy(DDA.photon_data[0:-1:downsampleRate, 1], DDA.photon_data[0:-1:downsampleRate, 2]))
	photonLine = photonLine.set_crs('EPSG:4326')  # Set as regular lat/lon for WGS84
	photonLineProjected = photonLine.to_crs(DDA.crs)  # Project to proper coordinate reference system (determined from location_to_polygon)

	outline = DDA.shape_polygon.boundary.plot(color='orange')
	photonLineProjected.plot(ax=outline, color='blue', markersize=3)

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Subsetted Granule Coverage over {}\n{}'.format(DDA.location, DDA.crs))
	plt.savefig(os.path.join(DDA.outdir, 'GlacierCoverage.png'), bbox_inches='tight')
	plt.close()


def plot_raw_data(DDA, segment=None, ax=None):
	'''
	Plots the raw photon point cloud data.
	ax: default is None. If ax=None, plot on a new axis (It will look like a normal plot)
		If ax is a matplotlib.pyplot axis object, plot on that axis
	'''
	if ax is None:
		indPlots = True
		titleSize = 40
		labelSize = 35
		tickSize = 35
		tinyTickSize = 20
	else:
		indPlots = False
		titleSize = 30
		labelSize = 20
		tickSize = 15
		tinyTickSize = 15
	plt.figure('Raw Data', figsize=fig_dims)

	if segment is not None:
		ps = segment
	else:
		ps = DDA.plot_segments

	photon_data_sortedInds = np.argsort(DDA.photon_data[:, 4])  # Sort the data based on along-track distance
	for i,seg in enumerate(ps):
		# start = DDA.plot_segments[seg][0]
		# end = DDA.plot_segments[seg][1]
		start = seg[0]
		end = seg[1]
		current_bin_endpoints = np.searchsorted(DDA.photon_data[:, 4], [start, end], sorter=photon_data_sortedInds)  # Use the sorted array to find the current bin's endpoints
		photon_segment = DDA.photon_data[current_bin_endpoints[0]:current_bin_endpoints[1], :]

		if indPlots is True:
			ax = plt.subplot(111)
			ax.set_position([.05, .15, .91, .8])

		ax.scatter(photon_segment[:, 4], photon_segment[:, 3], color='black', s=pt_size)
		ax.set_title('Raw Photon Data (Segment {})'.format(i), fontsize=titleSize)
		ax.set_xlabel('Along-Track Distance (m)', fontsize=labelSize)
		ax.set_ylabel('Elevation (m)', fontsize=labelSize)
		ax.set_xlim([start, end])
		ax.tick_params(labelsize=tickSize)

		tax = ax.twiny()
		if indPlots:
			tax.set_position([.05, .15, .91, .8])
		tax.set_xlim(ax.get_xlim())
		if len(photon_segment) > 0:
			time_labels = np.arange(photon_segment[0, 0], photon_segment[-1, 0], 1)
			time_xticks = np.interp(time_labels, photon_segment[:, 0], photon_segment[:, 4])

			tax.set_xticks(time_xticks)
			tax.set_xticklabels(time_labels, y=1)
			tax.tick_params(labelsize=tinyTickSize)

		if indPlots:
			plt.savefig(os.path.join(DDA.plotdir, 'Fig1_raw_data_segment{}.png'.format(i)), bbox_inches='tight')
		plt.clf()
	plt.close()


def plot_slabs(DDA_obj, segment=None, ax=None):
	'''
	Plots the signal slab in green and the noise slab in red.
	ax: default is None. If ax=None, plot on a new axis (It will look like a normal plot)
		If ax is a matplotlib.pyplot axis object, plot on that axis
	'''
	if ax is None:  # set some values for when it is an individual plot
		indPlots = True
		titleSize = 40
		labelSize = 35
		tickSize = 35
		markerSize = 10
		bbox_anchor = (0.5, -0.15)
	else:  # set some values for when it is part of a subplot
		indPlots = False
		titleSize = 30
		labelSize = 20
		tickSize = 15
		markerSize = 5
		bbox_anchor = None

	if segment is not None:
		ps = segment
	else:
		ps = DDA_obj.plot_segments

	plt.figure('Slabs', figsize=fig_dims)

	for i,seg in enumerate(ps):
		# start = DDA_obj.plot_segments[seg][0]
		# end = DDA_obj.plot_segments[seg][1]
		start = seg[0]
		end = seg[1]
		current_signal_endpoints = np.searchsorted(DDA_obj.photon_signal[:, 4], [start, end])  # Search for the start and end
		current_noise_endpoints = np.searchsorted(DDA_obj.photon_noise[:, 4], [start, end])

		signal_segment = DDA_obj.photon_signal[current_signal_endpoints[0]:current_signal_endpoints[1], :]
		noise_segment = DDA_obj.photon_noise[current_noise_endpoints[0]:current_noise_endpoints[1], :]

		if indPlots is True:
			ax = plt.subplot(111)
			ax.set_position([.05, .15, .91, .8])

		ax.scatter(signal_segment[:, 4], signal_segment[:, 3], color='green', s=pt_size)
		ax.scatter(noise_segment[:, 4], noise_segment[:, 3], color='red', s=pt_size)
		ax.set_title('Noise and Signal Slab (Segment {})'.format(i), fontsize=titleSize)
		ax.set_xlabel('Along-Track Distance (m)', fontsize=labelSize)
		ax.set_ylabel('Elevation (m)', fontsize=labelSize)
		ax.set_xlim([start, end])
		ax.tick_params(labelsize=tickSize)
		ax.legend(['Signal', 'Noise'], loc='upper center', bbox_to_anchor=bbox_anchor, ncol=2, fontsize=labelSize, markerscale=markerSize)

		# if indPlots == True: plt.savefig('output/'+run_name+'/plots/Fig2:slabs_segment'+str(seg)+'.png',bbox_inches='tight')
		if indPlots is True:
			plt.savefig(os.path.join(DDA_obj.plotdir, 'Fig2_slabs_segment{}.png'.format(i)), bbox_inches='tight')
		plt.clf()
	plt.close()


def plot_density(DDA_obj, segment=None, ax=None):
	'''
	ax: default is None. If ax=None, plot on a new axis (It will look like a normal plot)
		If ax is a list of matplotlib.pyplot axis objects, put plot on first axis, and put colorbar on second axis
	'''
	if ax is None:  # set some values for when it is an individual plot
		indPlots = True
		titleSize = 40
		labelSize = 35
		tickSize = 35
	else:  # set some values for when it is part of a subplot
		cbax = ax[1]  # colorbar axis
		ax = ax[0]  # plot axis
		indPlots = False
		titleSize = 30
		labelSize = 20
		tickSize = 15

	if segment is not None:
		ps = segment
	else:
		ps = DDA_obj.plot_segments

	min_dens = -3

	# For log plot, make sure no density is zero
	min_density = 0.0001  # minimum density

	DDA_obj.photon_signal[DDA_obj.photon_signal[:, 5] <= 0, 5] = min_density
	DDA_obj.photon_noise[DDA_obj.photon_noise[:, 5] <= 0, 5] = min_density


	max_dens = np.log(np.percentile(DDA_obj.photon_signal[:, 5], 99))

	plt.figure(figsize=fig_dims)
	for i,seg in enumerate(ps):
		# start = DDA_obj.plot_segments[seg][0]
		# end = DDA_obj.plot_segments[seg][1]
		start = seg[0]
		end = seg[1]
		signal_segment = DDA_obj.photon_signal[np.logical_and(DDA_obj.photon_signal[:, 4] >= start, DDA_obj.photon_signal[:, 4] < end), :]
		noise_segment = DDA_obj.photon_noise[np.logical_and(DDA_obj.photon_noise[:, 4] >= start, DDA_obj.photon_noise[:, 4] < end), :]

		if indPlots is True:
			ax = plt.subplot(111)
			cbax = plt.axes([.92, .15, .02, .8])

		sig = ax.scatter(signal_segment[:, 4], signal_segment[:, 3], c=np.log(signal_segment[:, 5]), cmap='jet', vmin=min_dens, vmax=max_dens, s=pt_size, edgecolor=None)
		ax.scatter(noise_segment[:, 4], noise_segment[:, 3], c=np.log(noise_segment[:, 5]), cmap='jet', vmin=min_dens, vmax=max_dens, s=pt_size, edgecolor=None)
		ax.set_title('Density Dimension Along-Track (Pass ' + str(DDA_obj.pass_num) + ', Segment ' + str(i) + ')', fontsize=titleSize)

		ax.set_xlabel('Along-Track Distance (m)', fontsize=labelSize)
		ax.set_ylabel('Elevation (m)', fontsize=labelSize)
		ax.set_xlim([start, end])


		cb = plt.colorbar(sig, cax=cbax, orientation='vertical')
		cb.ax.tick_params(labelsize=labelSize)
		cb.ax.yaxis.offsetText.set(size=labelSize)
		cb.set_label('Logarithm of Density', rotation=270, size=labelSize, labelpad=labelSize)
		ax.tick_params(labelsize=tickSize)

		# if indPlots == True: plt.savefig('output/'+run_name+'/plots/Fig3:density_segment_pass'+str(pass_num)+'_seg'+str(seg)+'.png',bbox_inches='tight')
		if indPlots is True:
			plt.savefig(os.path.join(DDA_obj.plotdir, 'Fig3_density_segment_pass{}_seg{}.png'.format(DDA_obj.pass_num, i)), bbox_inches='tight')
		plt.clf()
	plt.close()


def plot_thresholds(DDA_obj, segment=None, ax=None):
	'''
	ax: default is None. If ax=None, plot on a new axis (It will look like a normal plot)
		If ax is a matplotlib.pyplot axis object, plot on that axis
	'''
	if ax is None:  # set some values for when it is an individual plot
		indPlots = True
		titleSize = 40
		labelSize = 35
		tickSize = 35
		markerSize = 10
		bbox_anchor = (.5, -.15)

	else:  # set some values for when it is part of a subplot
		indPlots = False
		titleSize = 30
		labelSize = 20
		tickSize = 15
		markerSize = 5
		bbox_anchor = None

	if segment is not None:
		ps = segment
	else:
		ps = DDA_obj.plot_segments

	final_signal = DDA_obj.photon_signal[DDA_obj.signal_mask, :]  # points that passed thresholding AND quantile
	thresholded_signal = DDA_obj.photon_signal[DDA_obj.threshold_mask, :]  # points that passed thresholding but NOT quantile

	plt.figure('Thresholds', figsize=fig_dims)
	for i,seg in enumerate(ps):
		# start = DDA_obj.plot_segments[seg][0]
		# end = DDA_obj.plot_segments[seg][1]
		start = seg[0]
		end = seg[1]
		signal_segment = DDA_obj.photon_signal[np.logical_and(DDA_obj.photon_signal[:, 4] >= start, DDA_obj.photon_signal[:, 4] < end), :]
		noise_segment = DDA_obj.photon_noise[np.logical_and(DDA_obj.photon_noise[:, 4] >= start, DDA_obj.photon_noise[:, 4] < end), :]
		thresholded_signal_segment = thresholded_signal[np.logical_and(thresholded_signal[:, 4] >= start, thresholded_signal[:, 4] < end), :]
		final_signal_segment = final_signal[np.logical_and(final_signal[:, 4] >= start, final_signal[:, 4] < end), :]

		if len(final_signal_segment) == 0: continue  # the case for no data in a segment

		if indPlots is True:
			ax = plt.subplot(111)

		ax.scatter(signal_segment[:, 4], signal_segment[:, 5], color='green', s=pt_size)  # plot ALL signal slab
		ax.scatter(noise_segment[:, 4], noise_segment[:, 5], color='red', s=pt_size)  # plot noise slab
		ax.scatter(thresholded_signal_segment[:, 4], thresholded_signal_segment[:, 5], color='blue', s=pt_size)
		ax.scatter(final_signal_segment[:, 4], final_signal_segment[:, 5], color='lightgreen', s=pt_size)
		ax.set_title('Density Dimension of Thresholds Along-Track (Pass ' + str(DDA_obj.pass_num) + ', Segment ' + str(i) + ')', fontsize=titleSize)
		ax.set_xlabel('Along-Track Distance (m)', fontsize=labelSize)
		ax.set_ylabel('Density Dimension', fontsize=labelSize)
		ax.set_xlim([start, end])
		ax.tick_params(labelsize=tickSize)
		ax.set_ylim([0, np.max(final_signal_segment[:, 5]) + 1])
		ax.legend(['False Signal', 'Noise', 'Pre-Quantile', 'Post-Quantile'], loc='upper center', fancybox=True, framealpha=0.75, bbox_to_anchor=bbox_anchor, ncol=4, fontsize=labelSize, markerscale=markerSize)

		if indPlots is True:
			plt.savefig(os.path.join(DDA_obj.plotdir, 'Fig4_thresholds_segment_pass{}_seg_{}.png'.format(DDA_obj.pass_num, i)), bbox_inches='tight')
		plt.clf()
	plt.close()


def plot_ground_estimate(DDA, meltpondBool, topBool=False, ax=None):
	'''
	ax: default is None. If ax=None, plot on a new axis (It will look like a normal plot)
		If ax is a list of matplotlib.pyplot axis objects, put plot on first axis, and put colorbar on second axis
	'''
	if ax is None:  # set some values for when it is an individual plot
		indPlots = True
		titleSize = 40
		labelSize = 35
		tickSize = 35
		tinyTickSize = 20
		markerSize = 7
		bbox_anchor = (0.5, -0.15)
	else:  # set some values for when it is part of a subplot
		cbax = ax[1]  # colorbar axis
		ax = ax[0]  # plot axis
		indPlots = False
		titleSize = 30
		labelSize = 20
		tickSize = 15
		tinyTickSize = 10
		markerSize = 5
		bbox_anchor = None

	if not meltpondBool:
		# DDA-ice-1 ground estimate
		photon_signal_thresh = DDA.photon_signal_thresh
		ground_estimate = DDA.ground_estimate
		pass_num = DDA.pass_num
	else:
		# DDA-bifurcate ground estimate
		if topBool:
			# TOP surface
			photon_signal_thresh = DDA.photon_signal_top
			ground_estimate = DDA.ground_estimate_top
			pass_num = 0
		else:
			# BOTTOM surface
			photon_signal_thresh = DDA.photon_signal_bot
			ground_estimate = DDA.ground_estimate_bot
			pass_num = 1

	# photon stuff comes as [time, lon, lat, elev, distance, density]
	# ground_estimate comes as [lon, lat, elev, distance, time, elev_stdev, density_mean, weighted_elev_stdev]
	if len(photon_signal_thresh) == 0:
		print('No photons to plot ground with. All have been filtered out.')
		return
	min_dens = np.min(photon_signal_thresh[:, 5])
	max_dens = np.max(photon_signal_thresh[:, 5])

	plt.figure('Ground Estimate', figsize=fig_dims)
	for seg in range(len(DDA.plot_segments)):
		# if not DDA.cloud_filter_segments[seg]: continue # auto-remove cloudy segments
		start = DDA.plot_segments[seg][0]
		end = DDA.plot_segments[seg][1]
		unweighted_segment = DDA.photon_signal[np.logical_and(DDA.photon_signal[:, 4] >= start, DDA.photon_signal[:, 4] < end), :]
		weighted_segment = photon_signal_thresh[np.logical_and(photon_signal_thresh[:, 4] >= start, photon_signal_thresh[:, 4] < end), :]
		ground_segment = ground_estimate[np.logical_and(ground_estimate[:, 3] >= start, ground_estimate[:, 3] < end), :]

		if np.all(np.isnan(ground_segment)): continue

		if indPlots is True:
			ax = plt.subplot(111)
			cbax = plt.axes([.92, .15, .02, .8])

		if len(weighted_segment) == 0: continue  # the case for no data in a segment

		ax.set_title('Final Ground Estimate (Pass {}, Segment {})'.format(str(DDA.pass_num),str(seg)), fontsize=titleSize)
		ax.set_xlabel('Along-Track Distance (m)', fontsize=labelSize)
		ax.set_ylabel('Elevation (m)', fontsize=labelSize)
		ax.set_xlim([start, end])
		ax.set_ylim([np.nanmin(ground_segment[:, 2]) - 10, np.nanmax(ground_segment[:, 2]) + 10])  # set y limits based only on weighted points

		# thresholded out points
		ax.scatter(unweighted_segment[:, 4], unweighted_segment[:, 3], color='gray', s=pt_size, label='Photons Below Threshold')
		# weighted points
		w = ax.scatter(weighted_segment[:, 4], weighted_segment[:, 3], c=weighted_segment[:, 5], cmap='jet', vmin=min_dens, vmax=max_dens, s=2*pt_size, edgecolor=None, label='Signal Photons')

		# estimated ground
		if pass_num == 0:
			ax.plot(ground_segment[:, 3], ground_segment[:, 2], color='red', linewidth=1, label='Ground Estimate Pass 0')
		else:
			ax.plot(ground_segment[:, 3], ground_segment[:, 2], color='green', linewidth=1, label='Ground Estimate Pass 1')

		# +- 1.96*elev_stdev
		# ax.plot(ground_segment[:,3], ground_segment[:,2]+1.96*ground_segment[:,4], color='darkred', linewidth=1, linestyle='--', label=r'$\pm1.96\sigma$')
		# ax.plot(ground_segment[:,3], ground_segment[:,2]-1.96*ground_segment[:,4], color='darkred', linewidth=1, linestyle='--', label=r'$\pm1.96\sigma$')
		# titling and labels

		# Generate time labels and locations for plotting
		time_labels = np.arange(weighted_segment[0, 0], weighted_segment[-1, 0], 30)
		time_xticks = np.interp(time_labels, weighted_segment[:, 0], weighted_segment[:, 4])

		# Setup the axes for plotting the time (but not for MegaPlots)
		if indPlots:
			ax.set_position([.05, .15, .86, .8])
			tax = ax.twiny()
			tax.set_position([.05, .15, .86, .8])
			tax.set_xlim(ax.get_xlim())

			# Plot the track time at the top right corner (Toggle these lines to change if time labels get plotted)
			tax.set_xticks(time_xticks)
			tax.set_xticklabels(time_labels, y=1)
			tax.tick_params(labelsize=tinyTickSize)

		handles, labels = ax.get_legend_handles_labels()  # retrieve the handles and labels for all plotted lines
		# handles = [handles[3],handles[0],handles[1]]
		# labels = [labels[3],labels[0],labels[1]]
		ax.legend(handles=handles, labels=labels, loc='upper center', fancybox=True, framealpha=0.75, ncol=4, bbox_to_anchor=bbox_anchor, fontsize=labelSize, markerscale=markerSize)

		cb = plt.colorbar(w, cax=cbax, orientation='vertical')
		cb.ax.tick_params(labelsize=labelSize)
		cb.ax.yaxis.offsetText.set(size=labelSize)
		cb.set_label('Density', rotation=270, size=labelSize, labelpad=labelSize)
		ax.tick_params(labelsize=tickSize)

		if indPlots is True:
			plt.savefig(os.path.join(DDA.plotdir, 'Fig5_ground_segment_pass{}_seg{}.png'.format(pass_num, seg)), bbox_inches='tight')
		plt.clf()
	plt.close()


def plot_ground_estimate_no_interpolate(DDA, segment=None, ax=None):
	'''
	ax: default is None. If ax=None, plot on a new axis (It will look like a normal plot)
		If ax is a list of matplotlib.pyplot axis objects, put plot on first axis, and put colorbar on second axis
	''' 

	# photon_signal = DDA.photon_signal_thresh
	# photon_thresholded = DDA.photon_signal

	if ax is None:  # set some values for when it is an individual plot
		indPlots = True
		titleSize = 40
		labelSize = 35
		tickSize = 35
		tinyTickSize = 20
		markerSize = 7
		bbox_anchor = (0.5, -0.15)
	else:  # set some values for when it is part of a subplot
		cbax = ax[1]  # colorbar axis
		ax = ax[0]  # plot axis
		indPlots = False
		titleSize = 30
		labelSize = 20
		tickSize = 15
		tinyTickSize = 10
		markerSize = 5
		bbox_anchor = None

	if segment is not None:
		ps = segment
	else:
		ps = DDA.plot_segments

	# photon stuff comes as [time, lon, lat, elev, distance, density]
	# ground_estimate comes as [lon, lat, elev, distance, time, elev_stdev, density_mean, weighted_elev_stdev]
	if len(DDA.photon_signal_thresh) == 0:
		print('No photons to plot ground with. All have been filtered out.')
		return
	min_dens = np.min(DDA.photon_signal_thresh[:, 5])
	max_dens = np.max(DDA.photon_signal_thresh[:, 5])

	plt.figure('Ground Estimate', figsize=fig_dims)
	for i,seg in enumerate(ps):
		# start = DDA.plot_segments[seg][0]
		# end = DDA.plot_segments[seg][1]
		start = seg[0]
		end = seg[1]
		unweighted_segment = DDA.photon_signal[np.logical_and(DDA.photon_signal[:, 4] >= start, DDA.photon_signal[:, 4] < end), :]
		weighted_segment = DDA.photon_signal_thresh[np.logical_and(DDA.photon_signal_thresh[:, 4] >= start, DDA.photon_signal_thresh[:, 4] < end), :]
		ground_segment = DDA.ground_estimate[np.logical_and(DDA.ground_estimate[:, 3] >= start, DDA.ground_estimate[:, 3] < end), :]

		if indPlots is True:
			ax = plt.subplot(111)
			cbax = plt.axes([.92, .15, .02, .8])

		if len(weighted_segment) == 0: continue  # the case for no data in a segment

		ax.set_title('Final Signal Photons (Pass {}, Segment {})'.format(str(DDA.pass_num),str(i)), fontsize=titleSize)
		ax.set_xlabel('Along-Track Distance (m)', fontsize=labelSize)
		ax.set_ylabel('Elevation (m)', fontsize=labelSize)
		ax.set_xlim([start, end])
		if len(ground_segment[:, 2]) > 0:
			ax.set_ylim([np.min(ground_segment[:, 2]) - 10, np.max(ground_segment[:, 2]) + 10])  # set y limits based only on weighted points
		
		# thresholded out points
		ax.scatter(unweighted_segment[:, 4], unweighted_segment[:, 3], color='gray', s=pt_size, label='Photons Below Threshold')
		# weighted points
		w = ax.scatter(weighted_segment[:, 4], weighted_segment[:, 3], c=weighted_segment[:, 5], cmap='jet', vmin=min_dens, vmax=max_dens, s=2*pt_size, edgecolor=None, label='Signal Photons')


		# Generate time labels and locations for plotting
		time_labels = np.arange(weighted_segment[0, 0], weighted_segment[-1, 0], 30)
		time_xticks = np.interp(time_labels, weighted_segment[:, 0], weighted_segment[:, 4])

		# Setup the axes for plotting the time (but not for MegaPlots)
		if indPlots:
			ax.set_position([.05, .15, .86, .8])
			tax = ax.twiny()
			tax.set_position([.05, .15, .86, .8])
			tax.set_xlim(ax.get_xlim())

			# Plot the track time at the top right corner (Toggle these lines to change if time labels get plotted)
			tax.set_xticks(time_xticks)
			tax.set_xticklabels(time_labels, y=1)
			tax.tick_params(labelsize=tinyTickSize)

		handles, labels = ax.get_legend_handles_labels()  # retrieve the handles and labels for all plotted lines
		ax.legend(handles=handles, labels=labels, loc='upper center', fancybox=True, framealpha=0.75, ncol=4, bbox_to_anchor=bbox_anchor, fontsize=labelSize, markerscale=markerSize)

		cb = plt.colorbar(w, cax=cbax, orientation='vertical')
		cb.ax.tick_params(labelsize=labelSize)
		cb.ax.yaxis.offsetText.set(size=labelSize)
		cb.set_label('Density', rotation=270, size=labelSize, labelpad=labelSize)
		ax.tick_params(labelsize=tickSize)

		if indPlots is True:
			plt.savefig(os.path.join(DDA.plotdir, 'Fig5b_ground_segment_pass{}_seg{}.png'.format(DDA.pass_num, i)), bbox_inches='tight')
		plt.clf()
	plt.close()

def plot_ground_estimate_both(DDA, segment=None, ax=None):

	if ax is None:  # set some values for when it is an individual plot
		indPlots = True
		titleSize = 40
		labelSize = 35
		tickSize = 35
		tinyTickSize = 20
		markerSize = 7
		bbox_anchor = (0.5, -0.15)
	else:  # set some values for when it is part of a subplot
		cbax = ax[1]  # colorbar axis
		ax = ax[0]  # plot axis
		indPlots = False
		titleSize = 30
		labelSize = 20
		tickSize = 10
		tinyTickSize = 15
		markerSize = 5
		bbox_anchor = None

	if segment is not None:
		ps = segment
	else:
		ps = DDA.plot_segments

	# ground_estimate_pass0 = DDA.ground_estimate_top
	# ground_estimate_pass1 = DDA.ground_estimate_bot
	# photon_signal_pass0 = DDA.photon_signal_top
	# photon_signal_pass1 = DDA.photon_signal_bot
	# photon_thresholded = DDA.photon_signal

	# photon stuff comes as [time, lon, lat, elev, distance, density]
	# ground_estimate comes as [lon, lat, elev, distance, time, elev_stdev, density_mean, weighted_elev_stdev]
	if len(DDA.photon_signal_top) == 0 or len(DDA.photon_signal_bot) == 0:
		print('No photons to plot ground with. All have been filtered out.')
		return
	min_dens0 = np.min(DDA.photon_signal_top[:, 5])
	max_dens0 = np.max(DDA.photon_signal_top[:, 5])
	min_dens1 = np.min(DDA.photon_signal_bot[:, 5])
	max_dens1 = np.max(DDA.photon_signal_bot[:, 5])
	min_dens = np.min([min_dens0, min_dens1])
	max_dens = np.max([max_dens0, max_dens1])

	plt.figure('Ground Estimate', figsize=fig_dims)
	for i,seg in enumerate(ps):
		# start = DDA.plot_segments[seg][0]
		# end = DDA.plot_segments[seg][1]
		start = seg[0]
		end = seg[1]
		unweighted_segment = DDA.photon_signal[np.logical_and(DDA.photon_signal[:, 4] >= start, DDA.photon_signal[:, 4] < end), :]
		weighted_segment_pass0 = DDA.photon_signal_top[np.logical_and(DDA.photon_signal_top[:, 4] >= start, DDA.photon_signal_top[:, 4] < end), :]
		weighted_segment_pass1 = DDA.photon_signal_bot[np.logical_and(DDA.photon_signal_bot[:, 4] >= start, DDA.photon_signal_bot[:, 4] < end), :]
		weighted_segment = np.append(weighted_segment_pass0, weighted_segment_pass1, axis=0)
		weighted_segment = weighted_segment[np.argsort(weighted_segment[:, 4])]
		ground_segment_pass0 = DDA.ground_estimate_top[np.logical_and(DDA.ground_estimate_top[:, 3] >= start, DDA.ground_estimate_top[:, 3] < end), :]
		ground_segment_pass1 = DDA.ground_estimate_bot[np.logical_and(DDA.ground_estimate_bot[:, 3] >= start, DDA.ground_estimate_bot[:, 3] < end), :]

		if len(weighted_segment) == 0: continue  # the case for no data in a segment

		if indPlots is True:
			ax = plt.subplot(111)
			box = ax.get_position()
			ax.set_position([.05, .15, .86, .8])
			cbax = plt.axes([.92, .15, .02, .8])

		# tax = ax.twiny()
		# tax.set_position([.05, .15, .86, .8])

		# thresholded out points
		ax.scatter(unweighted_segment[:, 4], unweighted_segment[:, 3], color='gray', s=pt_size, label='Photons Below Threshold')
		# weighted points
		w = ax.scatter(weighted_segment[:, 4], weighted_segment[:, 3], c=weighted_segment[:, 5], cmap='jet', vmin=min_dens, vmax=max_dens, s=2*pt_size, edgecolor=None, label='Signal Photons')

		# Ground Estimate
		ax.plot(ground_segment_pass0[:, 3], ground_segment_pass0[:, 2], color='red', linewidth=2, label='Ground Estimate Pass 0')
		ax.plot(ground_segment_pass1[:, 3], ground_segment_pass1[:, 2], color='green', linewidth=2, label='Ground Estimate Pass 1')
		# +- 1.96*elev_stdev
		# ax.plot(ground_segment[:,3], ground_segment[:,2]+1.96*ground_segment[:,4], color='darkred', linewidth=1, linestyle='--', label=r'$\pm1.96\sigma$')
		# ax.plot(ground_segment[:,3], ground_segment[:,2]-1.96*ground_segment[:,4], color='darkred', linewidth=1, linestyle='--', label=r'$\pm1.96\sigma$')
		# titling and labels

		# Generate time labels and locations for plotting
					# time_labels = np.arange(weighted_segment[0, 0], weighted_segment[-1, 0], 30)
					# time_xticks = np.interp(time_labels, weighted_segment[:, 0], weighted_segment[:, 4])

		# Setup the axes for plotting the time
			# ax.set_position([.05, .15, .86, .8])
			# tax = ax.twiny()
			# tax.set_position([.05, .15, .86, .8])
			# tax.set_xlim(ax.get_xlim())

		# Plot the track time at the top right corner (Toggle these lines to change if time labels get plotted)
			# tax.set_xticks(time_xticks)
			# tax.set_xticklabels(time_labels, y=1)
			# tax.tick_params(labelsize=tinyTickSize)

		ax.set_title('Final Ground Estimate (Both Passes, Segment ' + str(i) + ')', fontsize=40/2)
		ax.set_xlabel('Along-Track Distance (m)', fontsize=35/2)
		ax.set_ylabel('Elevation (m)', fontsize=35/2)
		ax.set_xlim([start, end])
		if len(ground_segment_pass1) == 0: # No ground found in pass 1
			ax.set_ylim([np.nanmin(ground_segment_pass0[:, 2]) - 7, np.nanmax(ground_segment_pass0[:, 2]) + 8])  # set y limits based only on weighted points in pass 1
		else:
			ax.set_ylim([np.nanmin(ground_segment_pass1[:, 2]) - 7, np.nanmax(ground_segment_pass1[:, 2]) + 8])  # set y limits based only on weighted points in pass 1
		handles, labels = ax.get_legend_handles_labels()  # retrieve the handles and labels for all plotted lines
		# handles = [handles[3],handles[0],handles[1]]
		# labels = [labels[3],labels[0],labels[1]]
		ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=bbox_anchor, ncol=2, fontsize=35/2, markerscale=7)
		cb = plt.colorbar(w, cax=cbax, orientation='vertical')
		cb.ax.tick_params(labelsize=labelSize)
		cb.ax.yaxis.offsetText.set(size=labelSize)
		cb.set_label('Density', rotation=270, size=labelSize, labelpad=labelSize)
		ax.tick_params(labelsize=tickSize)


		if indPlots is True:
			plt.savefig(os.path.join(DDA.plotdir,'Fig6_ground_segment_both_seg{}.png'.format(i)), bbox_inches='tight')
		plt.clf()
	plt.close()

def plot_all(
	photon_data, plot_slabs_photon_signal, plot_slabs_photon_noise, full_signal,
	photon_noise, plot_thresholds_photon_signal, plot_thresholds_signal_mask, threshold_mask,
	photon_signal, ground_estimate, plot_segments, run_name, pass_num, plotdir):
	# These inputs are the union of the inputs for all individual plot functions
	'''
	This function calls the following plot functions:
		plot_raw_data, plot_slabs, plot_density, plot_thresholds, plot_ground_estimate
	with a specific axis passed in. It makes one single plot with the individual plotting
	functions making up the subplots
	'''
	for seg in range(len(plot_segments)):
		ps = np.array([plot_segments[seg]])
		fig, ax = plt.subplots(6, 2, sharex='col', figsize=(fig_dims[0]*1.03, fig_dims[1]*7), gridspec_kw={'width_ratios': [35, 1]})
		fig.subplots_adjust(wspace=0.05)

		# Call the plotting functions (with the proper axes) and set the title
		plot_raw_data(photon_data, ps, run_name, plotdir, ax[0, 0])
		ax[0, 0].set_title('Raw Photon Data (Segment {})'.format(seg), fontsize=30)

		plot_slabs(plot_slabs_photon_signal, plot_slabs_photon_noise, ps, run_name, plotdir, ax[1, 0])
		ax[1, 0].set_title('Noise and Signal Slab (Segment {})'.format(seg), fontsize=30)


		plot_density(full_signal, photon_noise, ps, run_name, pass_num, plotdir, [ax[2, 0], ax[2, 1]])
		ax[2, 0].set_title('Density Dimension Along-Track (Pass {}, Segment {})'.format(pass_num, seg), fontsize=30)

		plot_thresholds(plot_thresholds_photon_signal, plot_thresholds_signal_mask, threshold_mask, photon_noise, ps, run_name, pass_num, plotdir, ax[3, 0])
		ax[3, 0].set_title('Density Dimension of Thresholds Along-Track (Pass {}, Segment {})'.format(pass_num, seg), fontsize=30)

		plot_ground_estimate_no_interpolate(ground_estimate, photon_signal, full_signal, ps, run_name, pass_num, plotdir, ax=[ax[4, 0], ax[4, 1]])
		ax[4, 0].set_title('Signal Photons Colored By Density (Pass {}, Segment {})'.format(pass_num, seg), fontsize=30)

		plot_ground_estimate(ground_estimate, photon_signal, full_signal, ps, run_name, pass_num, plotdir, ax=[ax[5, 0], ax[5, 1]])
		ax[5, 0].set_title('Final Ground Estimate (Pass {}, Segment {})'.format(pass_num, seg), fontsize=30)

		# Delete the plots that are blank
		fig.delaxes(ax[0, 1])
		fig.delaxes(ax[1, 1])
		fig.delaxes(ax[3, 1])

		# Resize the x-axis tick labels
		ax[0, 0].tick_params(axis='x', reset=True, labelsize=20)
		ax[1, 0].tick_params(axis='x', reset=True, labelsize=20)
		ax[2, 0].tick_params(axis='x', reset=True, labelsize=20)
		ax[3, 0].tick_params(axis='x', reset=True, labelsize=20)
		ax[4, 0].tick_params(axis='x', reset=True, labelsize=20)
		ax[5, 0].tick_params(axis='x', reset=True, labelsize=20)

		# Save the MegaPlot figure
		fig.savefig(plotdir + '/Fig7_MegaPlot_segment_pass{}_seg{}.png'.format(pass_num, seg), bbox_inches='tight')
		plt.close()

def plot_all_both_grounds(DDA):
	# These inputs are the union of the inputs for all individual plot functions
	'''
	This function is for making the 'MegaPlots' with the ground estimate from both passes
	'''



	# photon_signal_pass0 = photon_signal[0]
	# photon_signal = photon_signal[1]

	# ground_estimate_pass0 = ground_estimate[0]
	# ground_estimate = ground_estimate[1]

	# full_signal_pass0 = full_signal[0]
	# full_signal = full_signal[1]


	for seg in range(len(DDA.plot_segments)):

		fig, ax = plt.subplots(6, 2, sharex='col', figsize=(fig_dims[0]*1.03, fig_dims[1]*7), gridspec_kw={'width_ratios': [35, 1]})
		fig.subplots_adjust(wspace=0.05)

		# Call the plotting functions (with the proper axes) and set the title
		plot_raw_data(DDA, np.array([DDA.plot_segments[seg]]), ax[0, 0])
		ax[0, 0].set_title('Raw Photon Data (Segment {})'.format(seg), fontsize=30)

		plot_slabs(DDA, np.array([DDA.plot_segments[seg]]), ax[1, 0])
		ax[1, 0].set_title('Noise and Signal Slab (Segment {})'.format(seg), fontsize=30)

		plot_density(DDA, np.array([DDA.plot_segments[seg]]), [ax[2, 0], ax[2, 1]])
		ax[2, 0].set_title('Density Dimension Along-Track (Pass {}, Segment {})'.format(0, seg), fontsize=30)

		plot_thresholds(DDA, np.array([DDA.plot_segments[seg]]), ax[3, 0])
		ax[3, 0].set_title('Density Dimension of Thresholds Along-Track (Pass {}, Segment {})'.format(0, seg), fontsize=30)

		plot_ground_estimate_no_interpolate(DDA, np.array([DDA.plot_segments[seg]]), ax=[ax[4, 0], ax[4, 1]])
		ax[4, 0].set_title('Signal Photons Colored By Density (Pass {}, Segment {})'.format(0, seg), fontsize=30)

		plot_ground_estimate_both(DDA, np.array([DDA.plot_segments[seg]]), ax=[ax[5, 0], ax[5, 1]])
		ax[5, 0].set_title('Final Ground Estimate (Both Passes, Segment {})'.format(np.array([seg])), fontsize=30)




		# Delete the plots that are blank
		fig.delaxes(ax[0, 1])
		fig.delaxes(ax[1, 1])
		fig.delaxes(ax[3, 1])

		# Resize the x-axis tick labels
		ax[0, 0].tick_params(axis='x', reset=True, labelsize=20)
		ax[1, 0].tick_params(axis='x', reset=True, labelsize=20)
		ax[2, 0].tick_params(axis='x', reset=True, labelsize=20)
		ax[3, 0].tick_params(axis='x', reset=True, labelsize=20)
		ax[4, 0].tick_params(axis='x', reset=True, labelsize=20)
		ax[5, 0].tick_params(axis='x', reset=True, labelsize=20)

		# Save the MegaPlot figure
		fig.savefig(os.path.join(DDA.plotdir, 'Fig7_MegaPlot_both_seg{}.png'.format(seg)), bbox_inches='tight')
		plt.close()
