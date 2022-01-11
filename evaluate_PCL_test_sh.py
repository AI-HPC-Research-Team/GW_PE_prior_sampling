# Copyright (c) 2020 Stephen Green
# Copyright (c) 2021 Peng Cheng Laboratory.
# Licensed under the MIT license.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import h5py
import corner
import time
import os

from inference.gwpe_main import PosteriorModel
import inference.waveform as wfg

from inference.nde_flows import obtain_samples
from inference.reduced_basis import SVDBasis
from gwpy.frequencyseries import FrequencySeries


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
def kl_divergence(samples, kde=stats.gaussian_kde, decimal=5, base=2.0):
    try:
         kernel = [kde(i,bw_method='scott') for i in samples]
    except np.linalg.LinAlgError:
         return float("nan")
        
    x = np.linspace(
        np.min([np.min(i) for i in samples]),
        np.max([np.max(i) for i in samples]),
        100
    )
    factor = 1.0e-5

    a, b = [k(x) for k in kernel]
    
    for index in range(len(a)):
        if a[index] < max(a) * factor:
            a[index] = max(a) * factor
            
    for index in range(len(b)):
        if b[index] < max(b) * factor:
            b[index] = max(b) * factor

    a = np.asarray(a)
    b = np.asarray(b)
    kl_forward = stats.entropy(a, qk=b, base=base)
    return kl_forward

def js_divergence(samples, kde=stats.gaussian_kde, decimal=5, base=2.0):
    try:
         kernel = [kde(i) for i in samples]
    except np.linalg.LinAlgError:
         return float("nan")
        
    x = np.linspace(
        np.min([np.min(i) for i in samples]),
        np.max([np.max(i) for i in samples]),
        100
    )
    
    a, b = [k(x) for k in kernel]
    a = np.asarray(a)
    b = np.asarray(b)
    
    m = 1. / 2 * (a + b)
    kl_forward = stats.entropy(a, qk=m, base=base)
    kl_backward = stats.entropy(b, qk=m, base=base)
    return np.round(kl_forward / 2. + kl_backward / 2., decimal)

labels = ['$m_1$', '$m_2$', '$\\phi_c$', '$t_c$', '$d_L$', '$a_1$',
       '$a_2$', '$t_1$', '$t_2$', '$\\phi_{12}$',
       '$\\phi_{jl}$', '$\\theta_{JN}$', '$\\psi$', '$\\alpha$',
       '$\\delta$']
######

model_path = '../../models/'
all_models = os.listdir(model_path)
for event in event_gps_dict.keys():
#     print (event)
    data_dir = 'data/' + event + '_sample_prior_basis/'
#     print (data_dir)
    for model_name in all_models:
        try:
            model_dir = os.path.join(model_path, model_name)
    #         print (model_dir)

            save_model_name = [f for f in os.listdir(model_dir) if ('_model.pt' in f) and ('.e' not in f) ][0]
            save_aux_filename = [f for f in os.listdir(model_dir) if ('_waveforms_supplementary.hdf5' in f) and ('.e' not in f) ][0]
            assert save_model_name[0] == 'e'
            assert save_aux_filename[0] == 'e'

            pm = PosteriorModel(model_dir=model_dir, 
                                data_dir=data_dir,
                                basis_dir=data_dir, 
                                sample_extrinsic_only=False, 
                                save_aux_filename=save_aux_filename,
                                save_model_name=save_model_name,
                                use_cuda=True)

            pm.load_model(pm.save_model_name)
            pm.wfd = wfg.WaveformDataset()
            pm.wfd.basis = SVDBasis()
            pm.wfd.basis.load(directory=pm.basis_dir)
            pm.wfd.Nrb = pm.wfd.basis.n

            pm.wfd.load_setting(pm.basis_dir, sample_extrinsic_only = False)
            pm.init_waveform_supp(pm.save_aux_filename)

            t_event = event_gps_dict[event]  # GPS time of coalescence
            T = 8.0  # number of seconds to analyze in a segment
            T_buffer = 2.0  # buffer time after the event to include

            # Load strain data for event
            test_data = 'data/events/{}/strain_FD_whitened.hdf5'.format(event)
            event_strain = {}
            with h5py.File(test_data, 'r') as f:
                event_strain = {det:f[det][:].astype(np.complex64) for det in pm.detectors}

            # Project onto reduced basis

            d_RB = {}
            for ifo, di in event_strain.items():
                h_RB = pm.wfd.basis.fseries_to_basis_coefficients(di)
                d_RB[ifo] = h_RB
            _, pm.event_y = pm.wfd.x_y_from_p_h(np.zeros(pm.wfd.nparams), d_RB, add_noise=False)

#             print('obtain samples...')
            nsamples_target_event = 50000
            x_samples = obtain_samples(pm.model, pm.event_y, nsamples_target_event, pm.device)

            x_samples = x_samples.cpu()
            # Rescale parameters. The neural network preferred mean zero and variance one. This undoes that scaling.
            test_samples = pm.wfd.post_process_parameters(x_samples.numpy())

            ##################### Save ###############
#             print('saving...')
            save_dir = '../all_test/event_' + event + '_model_' + model_name
            np.save(save_dir, test_samples)
            print('saving at', save_dir+'.npy')
            
        except Exception as e:
            print('-----------------------------------------------------')
            print('error:',e)
            print('event name:', event)
            print('model_dir:', model_dir)
            print('data_dir:', data_dir)
            print('save_model_name:', save_model_name)
            print('save_aux_filename:', save_aux_filename)
            print('Loading reduced basis:', pm.basis_dir)
            print('pm.wfd.basis.standardization_dict.keys():', pm.wfd.basis.standardization_dict.keys())
            print('-----------------------------------------------------')
            continue

print('done...')
