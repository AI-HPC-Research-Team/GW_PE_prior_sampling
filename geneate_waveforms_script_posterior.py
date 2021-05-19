import lfigw.waveform_generator_extra as wfg

# O1
# event = 'GW150914'
# event = 'GW151012'
# event = 'GW151226'
# O2
# event = 'GW170104'
# event = 'GW170608'
event = 'GW170729'
# event = 'GW170809'
# event = 'GW170814'
# event = 'GW170818'
# event = 'GW170823'


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

wfd = wfg.WaveformDataset_extra(spins_aligned=False, domain='RB',
                          extrinsic_at_train=True)

wfd.Nrb = 600
wfd.approximant = 'IMRPhenomPv2'

wfd.load_event('data/events/{}/'.format(event))

wfd.importance_sampling = 'uniform_distance'
wfd.fiducial_params['distance'] = fiducial_params[event]

print('Dataset properties')
print('Event', wfd.event)
print(wfd.prior)
print('f_min', wfd.f_min)
print('f_min_psd', wfd.f_min_psd)
print('f_max', wfd.f_max)
print('T', wfd.time_duration)
print('reference time', wfd.ref_time)

wfd._load_posterior(wfd.event) # loading bilby posterior as training dist.

wfd.generate_reduced_basis(50000)

wfd.generate_dataset(1000000)

wfd.generate_noisy_test_data(5000)

wfd.save('waveforms/{}_posterior'.format(event))
wfd.save_train('waveforms/{}_posterior'.format(event))
wfd.save_noisy_test_data('waveforms/{}_posterior'.format(event))

print('Program complete. Waveform dataset has been saved.')
