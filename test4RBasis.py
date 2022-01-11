#! /usr/local/bin/python
# Copyright (c) 2020 Stephen Green
# Copyright (c) 2021 Peng Cheng Laboratory.
# Licensed under the MIT license.

import inference.waveform as wfg
from inference.reduced_basis import SVDBasis

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
 

    fileName = datetime.datetime.now().strftime('test4RBasis_'+'%Y_%m_%d')
    sys.stdout = Logger('data/' + fileName + '.logfile', path=path)
 
    #############################################################
    # 这里输出之后的所有的输出的 print 内容即将写入日志
    #############################################################
    print(datetime.datetime.now().strftime(' test4RBasis_'+'_%Y_%m_%d_%H_%M_%S ').center(60,'*'))

if __name__ == '__main__':
    make_print_to_file(path='./')

    # O1
    #event = 'GW150914'
    # event = 'GW151012'
    # event = 'GW151226'
    # O2
    # event = 'GW170104'
    #event = 'GW170608'
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

    wfd = wfg.WaveformDataset(spins_aligned=False, domain='RB',
                            extrinsic_at_train=True, sampling_from='uniform')
    addr = 'data/{}{}_basis/'.format(event, wfd._sample_prior.__name__)    
    wfd.load_setting(addr, sample_extrinsic_only = False)
    wfd._load_posterior(wfd.event)  # loading bilby posterior as training dist.

    div = 1#200 # For demo

    print()  

    ## 1
    addr = 'data/{}{}_basis/'.format(event, wfd._sample_prior.__name__)    
    print('Loading reduced basis:', addr)
    if wfd.domain == 'RB':
        wfd.basis = SVDBasis()
        wfd.basis.load(addr)
        wfd.Nrb = wfd.basis.n
    wfd.load_setting(addr, sample_extrinsic_only = False)

    print('Dataset properties')
    print('Event', wfd.event)
    print(wfd.prior)
    print('f_min', wfd.f_min)
    print('f_min_psd', wfd.f_min_psd)
    print('f_max', wfd.f_max)
    print('T', wfd.time_duration)
    print('reference time', wfd.ref_time)

    # wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior,
    #                         fiducial_distance=1000, truncate=None)
    # wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior,
    #                         fiducial_distance=450, truncate=100)
    #wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior,
    #                        fiducial_distance=450, truncate=None)    
    #wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior,
    #                        fiducial_distance=1000, truncate=100)    
    #wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior_posterior,
    #                        fiducial_distance=450, truncate=None)
    #wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior_posterior,
    #                        fiducial_distance=1000, truncate=100)    
    wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior,
                             truncate=None,save_dir='./data/{}_uniform_prior'.format(event))
    wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior,
                             truncate=100,save_dir='./data/{}_uniform_prior'.format(event))
    wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior_posterior,
                             truncate=None,save_dir='./data/{}_uniform_prior'.format(event))
    wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior_posterior,
                            truncate=100,save_dir='./data/{}_uniform_prior'.format(event))
    #wfd.test_reduced_basis(n_test=150000//div, 
    #                        prior_fun=wfd._sample_prior,
    #                        fiducial_distance=None,
    #                        truncate=None,
    #                        save_dir='./data/_uniform_prior')    
    #wfd.test_reduced_basis(n_test=150000//div, 
    #                        prior_fun=wfd._sample_prior_posterior,
    #                        fiducial_distance=None,
    #                        truncate=None,
    #                        save_dir='./data/_uniform_prior')

    #wfd.test_reduced_basis(n_test=100000//div,
    #                        prior_fun=wfd._sample_prior,
    #                        fiducial_distance=None,
    #                        truncate=50,
    #                        save_dir='./data/_uniform_prior')
    #wfd.test_reduced_basis(n_test=100000//div,
    #                        prior_fun=wfd._sample_prior_posterior,
    #                        fiducial_distance=None,
    #                        truncate=50,
    #                        save_dir='./data/_uniform_prior')
    #wfd.test_reduced_basis(n_test=100000//div, 
    #                        prior_fun=wfd._sample_prior,
    #                        fiducial_distance=None,
    #                        truncate=300,
    #                        save_dir='./data/_uniform_prior')    
    #wfd.test_reduced_basis(n_test=100000//div, 
    #                        prior_fun=wfd._sample_prior_posterior,
    #                        fiducial_distance=None,
    #                        truncate=300,
    #                        save_dir='./data/_uniform_prior')

    #input('continue?')
    ## 2
    addr = 'data/{}{}_basis/'.format(event, wfd._sample_prior_posterior.__name__)    
    print('Loading reduced basis:', addr)
    if wfd.domain == 'RB':
        wfd.basis = SVDBasis()
        wfd.basis.load(addr)
        wfd.Nrb = wfd.basis.n                            
    wfd.load_setting(addr, sample_extrinsic_only = False)

    print('Dataset properties')
    print('Event', wfd.event)
    print(wfd.prior)
    print('f_min', wfd.f_min)
    print('f_min_psd', wfd.f_min_psd)
    print('f_max', wfd.f_max)
    print('T', wfd.time_duration)
    print('reference time', wfd.ref_time)

    # wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior_posterior,
    #                         fiducial_distance=1000, truncate=None)
    # wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior_posterior,
    #                         fiducial_distance=450, truncate=100)
    # wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior,
    #                         fiducial_distance=450, truncate=None)    
    # wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior,
    #                         fiducial_distance=1000, truncate=100)        
    # wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior_posterior,
    #                         fiducial_distance=450, truncate=None)
    # wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior_posterior,
    #                         fiducial_distance=1000, truncate=100)    
    # wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior,
    #                         truncate=None)
    # wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior,
    #                         truncate=100)
    wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior_posterior,
                             truncate=None,save_dir='./data/{}_posterior_prior'.format(event))
    wfd.test_reduced_basis(n_test=10000//div, prior_fun=wfd._sample_prior_posterior,
            truncate=100,save_dir='./data/{}_posterior_prior'.format(event))
    #wfd.test_reduced_basis(n_test=150000//div, 
    #                        prior_fun=wfd._sample_prior,
    #                        fiducial_distance=None,
    #                        truncate=None,
    #                        save_dir='./data/_posterior_prior')    
    #wfd.test_reduced_basis(n_test=150000//div, 
    #                        prior_fun=wfd._sample_prior_posterior,
    #                        fiducial_distance=None,
    #                        truncate=None,
    #                        save_dir='./data/_posterior_prior')
