#! /usr/local/bin/python
# Copyright (c) 2020 Stephen Green
# Copyright (c) 2021 Peng Cheng Laboratory.
# Licensed under the MIT license.

import inference.waveform as wfg

def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import sys
    import os
    import datetime
 
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
 
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
 
        def flush(self):
            pass
 

    fileName = datetime.datetime.now().strftime('prepreparing4RBasis_'+'%Y_%m_%d')
    sys.stdout = Logger('data/' + fileName + '.logfile', path=path)
 
    #############################################################
    # 这里输出之后的所有的输出的 print 内容即将写入日志
    #############################################################
    print(datetime.datetime.now().strftime(' prepreparing4RBasis_'+'_%Y_%m_%d_%H_%M_%S ').center(60,'*'))

if __name__ == '__main__':
    make_print_to_file(path='./')

    # O1
    # event = 'GW150914'
    # event = 'GW151012'
    #event = 'GW151226'
    # O2
    # event = 'GW170104'
    # event = 'GW170608'
    # event = 'GW170729'
    #event = 'GW170809'
    event = 'GW170817'
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
        GW170817=100,
        GW170818=2000,
        GW170823=4000,
    )

    wfd = wfg.WaveformDataset(spins_aligned=False, domain='RB',
                            extrinsic_at_train=True)

    wfd.Nrb = 1200#600
    wfd.approximant = 'IMRPhenomPv2'

    wfd.load_event('data/events/{}/'.format(event))
    #wfd.load_event('data/events/{}_30s/'.format(event))

    wfd.importance_sampling = 'uniform_distance'
    wfd.fiducial_params['distance'] = fiducial_params[event]

    if event in ['GW151226', 'GW170608']:
        wfd.prior['mass_1'][0] = 5.0
        wfd.prior['mass_2'][0] = 5.0
    if event in ['GW170817']:
        wfd.prior['mass_1'][0] = 1.0
        wfd.prior['mass_2'][0] = 1.0
        wfd.prior['mass_1'][1] = 5.0
        wfd.prior['mass_2'][1] = 5.0       
        wfd.prior['distance'][0] = 10
        wfd.prior['distance'][1] = 100
    print('Dataset properties')
    print('Event', wfd.event)
    print(wfd.prior)
    print('f_min', wfd.f_min)
    print('f_min_psd', wfd.f_min_psd)
    print('f_max', wfd.f_max)
    print('T', wfd.time_duration)
    print('reference time', wfd.ref_time)
    
    wfd.sample_extrinsic_only = False
    wfd._load_posterior(wfd.event)  # loading bilby posterior as training dist.

    div = 1#200 # For demo

    print()
    wfd.train_reduced_basis(n_train=500000//div, prior_fun=wfd._sample_prior)
    #wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior,
    #                        fiducial_distance=1000, truncate=None)
    wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior,
                            fiducial_distance=100, truncate=100)
    #wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior_posterior,
    #                        fiducial_distance=1000, truncate=None)
    wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior_posterior,
                            fiducial_distance=100, truncate=100)
    addr = 'data/{}{}_basis/'.format(event, wfd._sample_prior.__name__)
    #addr = 'data/{}_30s_{}_basis/'.format(event, wfd._sample_prior.__name__)
    wfd.basis.save(addr)
    wfd.save_setting(data_dir=addr)

    print()
    #wfd.train_reduced_basis(n_train=50000//div, prior_fun=wfd._sample_prior_posterior)
    #wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior_posterior,
    #                        fiducial_distance=1000, truncate=None)
    #wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior_posterior,
    #                        fiducial_distance=450, truncate=100)
    #wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior,
    #                        fiducial_distance=1000, truncate=None)
    #wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior,
    #                        fiducial_distance=450, truncate=100)
    #addr = 'data/{}{}_basis/'.format(event, wfd._sample_prior_posterior.__name__)
    #addr = 'data/{}_30s_{}_basis/'.format(event, wfd._sample_prior_posterior.__name__)
    #wfd.basis.save(addr)
    #wfd.save_setting(data_dir=addr)
