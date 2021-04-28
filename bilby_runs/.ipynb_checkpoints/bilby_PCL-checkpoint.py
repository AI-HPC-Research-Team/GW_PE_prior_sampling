#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on GW150914

This example estimates all 15 parameters of the binary black hole system using
commonly used prior distributions. This will take several hours to run. The
data is obtained using gwpy, see [1] for information on how to access data on
the LIGO Data Grid instead.

[1] https://gwpy.github.io/docs/stable/timeseries/remote-access.html

Main modifications (for consistency with LFI analysis):
    * More precise trigger_time
    * duration -> 8 s
    * psd_duration -> 1024 s
    * Prior file GW150914.prior modified
    * reference_frequency -> 20 Hz
    * set minimum_frequency -> 20 Hz
    * set maximum_frequency -> 1024 Hz
    * set roll_off for data segment window to 0.4 s. Default is 0.2 s.
      This was causing an issue with a spike around 1000 Hz. Now it is
      consistent with the window function for the PSD estimation.
    * Changed to pycbc PSD estimation. For some reason, the gwpy methods
      were giving very slightly different results.
    * nlive -> 1500, nact -> 10
"""
# from __future__ import division, print_function
import bilby
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import pycbc.psd
from scipy.signal import tukey
import numpy as np
import h5py, json
from pathlib import Path

logger = bilby.core.utils.logger

event = 'GW150914'
alpha = '0.01'
version = 'v3'
npool = 20 # 这个根据你们机器的情况来设定，可以通过 from multiprocessing import cpu_count; cpu_count() 来查看总量

outdir = event
label = event + '_' + alpha + '_' + version
addr = 'Model{event}_PSD{event}_Noise{event}_Inj{event}_{alpha}_{version}'.format(event = event[2:],
                                                                      alpha = alpha,
                                                                      version = version) 

config_fn = 'event_info.json'
event_dir = '../../injection/{}/'.format(addr)

# We now use gwpy to obtain analysis and psd data and create the ifo_list
ifo_list = bilby.gw.detector.InterferometerList([])


p = Path(event_dir)
event_dir = p

# Load event info
with open(p / 'event_info.json', 'r') as f:
    d = json.load(f)
    event = d['event']
    f_min = d['f_min']
    f_max = d['f_max']
    f_min_psd = f_min
    duration = d['T']  # Analysis segment duration
    post_trigger_duration = d['T_buffer'] # Time between trigger time and end of segment
    trigger_time = d['t_event']
    window_factor = d['window_factor']
    detectors = d['detectors']
    roll_off = d['f_max']  # Roll off duration of tukey window in seconds, default is 0.4s
delta_f = 1.0/duration
fs = 4096

# Data set up
psd_duration = 1024


# Load strain data for event
event_strain = {}
with h5py.File(event_dir / 'strain_TD.hdf5', 'r') as f:
    event_strain['sample_times'] = f['sample_times'][:]
    event_strain['H1'] = f['H1'][:].astype(np.float64)
    event_strain['L1'] = f['L1'][:].astype(np.float64)
    if len(detectors) == 3:
        event_strain['V1'] = f['V1'][:].astype(np.float64)
        
# Load PSD data for event
event_PSD = {}
with h5py.File(event_dir / 'PSD_TD.hdf5', 'r') as f:
    event_PSD['sample_times'] = f['sample_times'][:]
    event_PSD['H1'] = f['H1'][:].astype(np.float64)
    event_PSD['L1'] = f['L1'][:].astype(np.float64)
    if len(detectors) == 3:
        event_PSD['V1'] = f['V1'][:].astype(np.float64)

for det in detectors:
    logger.info("Loading analysis data for ifo {}".format(det))
    timeseries = TimeSeries(event_strain[det], times=event_strain['sample_times'], channel=det)
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    ifo.strain_data.roll_off = 0.4  # Set this explicitly. Default is 0.2.
    ifo.strain_data.set_from_gwpy_timeseries(timeseries)

    logger.info("Loading psd data for ifo {}".format(det))
    # The PSD length should be the same as the length of FD
    # waveforms, which is determined from delta_f and f_max.
    PSDtimeseries = TimeSeries(event_PSD[det], times=event_PSD['sample_times'], channel=det)
    
    psd_alpha = 2 * roll_off / duration

    # Use pycbc psd routine
    sampling_rate = len(PSDtimeseries)/psd_duration
    psd_data_pycbc = PSDtimeseries.to_pycbc()
    w = tukey(int(duration * sampling_rate), psd_alpha)
    psd = pycbc.psd.estimate.welch(psd_data_pycbc,
                                   seg_len=int(duration * sampling_rate),
                                   seg_stride=int(duration * sampling_rate),
                                   window=w,
                                   avg_method='median')
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=np.array(psd.sample_frequencies),
        psd_array=np.array(psd))

    ifo_list.append(ifo)
    

logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)

# Save strain data
ifo_list.save_data(outdir, label)

# We now define the prior.
# We have defined our prior distribution in a local file, GW150914.prior
# The prior is printed to the terminal at run-time.
# You can overwrite this using the syntax below in the file,
# or choose a fixed value by just providing a float value as the prior.

# Modified this file as well.
priors = bilby.gw.prior.BBHPriorDict(filename='GW150914.prior')
priors['geocent_time'].minimum = trigger_time - 0.1
priors['geocent_time'].maximum = trigger_time + 0.1

# In this step we define a `waveform_generator`. This is the object which
# creates the frequency-domain strain. In this instance, we are using the
# `lal_binary_black_hole model` source model. We also pass other parameters:
# the waveform approximant and reference frequency and a parameter conversion
# which allows us to sample in chirp mass and ratio rather than component mass
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments={'waveform_approximant': 'IMRPhenomPv2',
                        'reference_frequency': f_min,
                        'minimum_frequency': f_min,
                        'maximum_frequency': f_max})

# In this step, we define the likelihood. Here we use the standard likelihood
# function, passing it the data and the waveform generator.
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list, waveform_generator, priors=priors, time_marginalization=False,
    phase_marginalization=False, distance_marginalization=False)

# Finally, we run the sampler. This function takes the likelihood and prior
# along with some options for how to do the sampling and how to save the data
result = bilby.run_sampler(
    likelihood, priors, sampler='dynesty', outdir=outdir, label=label,
    nlive=1000, nact=10, walks=100, n_check_point=10000, check_point_plot=True,
    npool=npool,resume=False,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    plot=False)
# result.plot_corner()