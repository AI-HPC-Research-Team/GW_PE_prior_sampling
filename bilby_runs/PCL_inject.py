# This is based on the example script provided in the bilby documentation.
import numpy as np
import pandas as pd
import bilby

# 这个根据你们机器的情况来设定，
# 可以通过 from multiprocessing import cpu_count; cpu_count() 来查看总量
npool = eval(input('Input the number of CPUs for this runing: [int]'))

############ event ####################
event_list = ['GW150914','GW151012','GW151226', # O1
              'GW170104','GW170608','GW170729',
              'GW170809','GW170814','GW170818',
              'GW170823',]
print('(Hint: ', end='')
[print(event, end=' ') for event in event_list]
print(')')
event = input('Input an event name: ')
assert event in event_list

############# param ####################
param_names = ['mass_1', 'mass_2', 'phase', 'geocent_time','luminosity_distance',
               'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
               'theta_jn', 'psi', 'ra', 'dec']

print('(Hint: ', end='')
[print(param, end=' ') for param in param_names]
print(')')
param = input('Input a param name: ')
assert param in param_names

############# value #################### 
# Refine the distance fiducial_params for each event
fiducial_params = dict(
    GW150914=1000,
    GW151012=2000,
    GW151226=1000,
    GW170104=2000,
    GW170608=1000,
    GW170729=5000,
    GW170809=2000,
    GW170814=1000,
    GW170817=500,
    GW170818=2000,
    GW170823=4000,
)
value_list = dict( # TODO
    luminosity_distance=np.linspace(100, fiducial_params[event], 10, dtype=np.int),
)

print('(Hint: ', end='')
[print(value, end=' ') for value in value_list[param]]
print(')')
value = eval(input('Input a param name: '))
assert value in value_list[param]

#################################

# Specify the output directory and the name of the simulation.
label = param + '_' + str(value)
outdir = './{}/{}'.format(event, label)

bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)

event_gps_dict = {
     'GW150914': 1126259462.4,
     'GW151012': 1128678900.4,
     'GW151226': 1135136350.6,
     'GW170104': 1167559936.6,
     'GW170608': 1180922494.5,
     'GW170729': 1185389807.3,
     'GW170809': 1186302519.8,
     'GW170814': 1186741861.5,
     'GW170818': 1187058327.1,
     'GW170823': 1187529256.5
}
event_detectors_dict = {
     'GW150914': ['H1', 'L1'],
     'GW151012': ['H1', 'L1'],
     'GW151226': ['H1', 'L1'],
     'GW170104': ['H1', 'L1'],
     'GW170608': ['H1', 'L1'],
     'GW170729': ['H1', 'L1', 'V1'],
     'GW170809': ['H1', 'L1', 'V1'],
     'GW170814': ['H1', 'L1', 'V1'],
     'GW170818': ['H1', 'L1', 'V1'],
     'GW170823': ['H1', 'L1']
}

# Refine the distance fiducial_params for each event
mass_dist = dict(
    GW150914=[10, 80],
    GW151012=[5, 80],
    GW151226=[1, 50],
    GW170104=[10, 80],
    GW170608=[1, 50],
    GW170729=[10, 80],
    GW170809=[10, 80],
    GW170814=[10, 80],
    GW170817=[0.1, 5],
    GW170818=[10, 80],
    GW170823=[10, 80],
)

# Load bilby samples
df = pd.read_csv('../downsampled_posterior_samples_v1.0.0/{}_downsampled_posterior_samples.dat'.format(event), sep=' ')
bilby_samples = df.dropna()[param_names].values.astype('float64')

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 8.
sampling_frequency = 4096.
event_buffer = 2.

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
injection_parameters = {
    key:value for key, value in zip(param_names, 
                                    np.median(bilby_samples, axis=0))
}

# Change to fiducial params and save them
injection_parameters[param] = value
np.save(outdir+'/'+label+'_injectparams', injection_parameters)

# Fixed arguments passed into the source model
waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=20., minimum_frequency=20.)

# Create the waveform_generator using a LAL BinaryBlackHole source function
# the generator will convert all the parameters
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)

# Set up interferometers. These default to their design sensitivity
ifos = bilby.gw.detector.InterferometerList(event_detectors_dict[event])

# Insert noise
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - duration + event_buffer)


# Save strain data
ifos.save_data(outdir, label)

# Start with standard BBH priors. Modify certain priors.
# priors = bilby.gw.prior.BBHPriorDict()
priors = bilby.gw.prior.BBHPriorDict(filename='default.prior')
priors['mass_1'] = bilby.prior.Uniform(
    minimum=mass_dist[event][0], maximum=mass_dist[event][1], 
    name='mass_1', latex_label='$m_1$',
    unit='$M_{\\odot}$', boundary=None)
priors['mass_2'] = bilby.prior.Uniform(
    minimum=mass_dist[event][0], maximum=mass_dist[event][1], 
    name='mass_2', latex_label='$m_2$',
    unit='$M_{\\odot}$', boundary=None)
priors['a_1'] = bilby.prior.Uniform(
    minimum=0, maximum=0.99, name='a_1', latex_label='$a_1$',
    unit=None, boundary='reflective')
priors['a_2'] = bilby.prior.Uniform(
    minimum=0, maximum=0.99, name='a_2', latex_label='$a_2$',
    unit=None, boundary='reflective')
priors['luminosity_distance'].maximum = fiducial_params[event]
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 0.1,
    maximum=injection_parameters['geocent_time'] + 0.1,
    name='geocent_time', latex_label='$t_c$', unit='$s$')

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveoform generator, as well the priors.
# The explicit time, distance, and phase marginalizations are turned on to
# improve convergence, and the parameters are recovered by the conversion
# function.
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator, priors=priors,
    distance_marginalization=False, phase_marginalization=False,
    time_marginalization=False)

# Run sampler. In this case we're going to use the `cpnest` sampler Note that
# the maxmcmc parameter is increased so that between each iteration of the
# nested sampler approach, the walkers will move further using an mcmc
# approach, searching the full parameter space. The conversion function will
# determine the distance, phase and coalescence time posteriors in post
# processing.
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty',
    injection_parameters=injection_parameters, outdir=outdir,
    label=label, nlive=1000, nact=10, walks=100, n_check_point=10000,
    npool=npool,resume=False,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    plot=False)