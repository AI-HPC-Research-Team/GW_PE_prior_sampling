import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm
import h5py
import corner
import time

from lfigw.gwpe_extra import PosteriorModel
import lfigw.waveform_generator_extra as wfg

import os
path = '/userhome/O2/injection'

event = 'GW150914'
event_name = 'Model150914_PSD150914_Noise150914_Inj150914'
os.mkdir('/userhome/O2/output/' + event_name)
version = 'v3'

#alpha = ['0.0002', '0.0005', '0.001', '0.0011', '0.0012', '0.0013', '0.0014', '0.0016', '0.0017', '0.002', '0.003', '0.004', '0.005', '0.006', '0.009', '0.01']
l = os.listdir(path)
testd = {}
ind = 1
for x in l:
    if x.startswith(event_name) and x.endswith(version):
        testd[ind] = x
        ind += 1
        
# for tmp in alpha:
#     print("#######", tmp)
#     tmp_dir = event_name + '_' + str(tmp) + '_' + version
#     testd[ind] = tmp_dir
#     ind += 1

#pm = PosteriorModel(model_dir='/userhome/O2/gwpe_151012/models/{}_posterior/'.format(event), data_dir='/userhome/O2/gwpe_151012/waveforms/{}_posterior/'.format(event))
pm = PosteriorModel(model_dir='/userhome/O2/models/202103/{}/'.format(event), data_dir='/userhome/O2/waveforms/{}/'.format(event))
pm.load_model()
pm.wfd = wfg.WaveformDataset()
pm.wfd.load_noisy_test_data(pm.data_dir)
pm.init_waveform_supp()

from lfigw.nde_flows import obtain_samples

for i in range(len(testd)):
    event_strain = {}
    # with h5py.File('../data/events/{}/strain_FD_whitened.hdf5'.format(event), 'r') as f:
    with h5py.File('/userhome/O2/injection/{}/strain_FD_whitened.hdf5'.format(testd[i+1]), 'r') as f:
        event_strain['H1'] = f['H1'][:].astype(np.complex64)
        event_strain['L1'] = f['L1'][:].astype(np.complex64)
        #event_strain['V1'] = f['V1'][:].astype(np.complex64)

    d_RB = {}
    for ifo, di in event_strain.items():
        h_RB = pm.wfd.basis.fseries_to_basis_coefficients(di)
        d_RB[ifo] = h_RB
    _, y = pm.wfd.x_y_from_p_h(pm.wfd.noisy_waveforms_parameters[0], d_RB, add_noise=False)

#     for det, h in d_RB.items():
#         plt.plot(h)
#     plt.savefig('/userhome/O2/output/figure1_{}_{}_{}.jpg'.format(event, testd[i+1], version))

    nsamples = 50000

    start = time.time()
    x_samples = obtain_samples(pm.model, y, nsamples, pm.device)
    end = time.time()
    print(start-end)

    x_samples = x_samples.cpu()

    # Rescale parameters. The neural network preferred mean zero and variance one. This undoes that scaling.
    params_samples = pm.wfd.post_process_parameters(x_samples.numpy())

    np.savez_compressed('/userhome/O2/output/' + event_name + '/posterior_{}.npz'.format(testd[i+1]), samples=params_samples, parameter_labels=pm.wfd.parameter_labels,)

#     corner.corner(params_samples, labels=pm.wfd.parameter_labels, color='black',
#                   levels=[0.5, 0.9],
#                   scale_hist=True, plot_datapoints=False#, range=dom
#                  )
#     # plt.suptitle('Red: bilby dynesty, Black: NN')
#     plt.savefig('/userhome/O2/output/figure2_{}_{}_{}.jpg'.format(event, testd[i+1], version))

#     # mpl.rcParams['font.size'] = 14


#     def make_pp(percentiles, parameter_labels, ks=True):
#         percentiles = percentiles / 100.
#         nparams = percentiles.shape[-1]
#         nposteriors = percentiles.shape[0]

#         ordered = np.sort(percentiles, axis=0)
#         ordered = np.concatenate((np.zeros((1, nparams)), ordered, np.ones((1, nparams))))
#         y = np.linspace(0, 1, nposteriors + 2)

#         fig = plt.figure(figsize=(10, 10))

#         for n in range(nparams):
#             if ks:
#                 pvalue = stats.kstest(percentiles[:, n], 'uniform')[1]
#                 plt.step(ordered[:, n], y, where='post', label=parameter_labels[n] + r' ({:.3g})'.format(pvalue))
#             else:
#                 plt.step(ordered[:, n], y, where='post', label=parameter_labels[n])
#         plt.plot(y, y, 'k--')
#         plt.legend()
#         plt.ylabel(r'$CDF(p)$')
#         plt.xlim((0, 1))
#         plt.ylim((0, 1))

#         plt.xlabel(r'$p$')

#         ax = fig.gca()
#         ax.set_aspect('equal', anchor='SW')

#         # plt.show()
#         plt.savefig('/userhome/O2/output/figure3_{}_{}_{}.jpg'.format(event, testd[i+1], version))

#     neval = 100    # number of injections
#     nparams = pm.wfd.nparams

#     percentiles = np.empty((neval, nparams))
#     for idx in tqdm(range(neval)):
#         samples = pm.evaluate(idx=idx, nsamples=10000, plot=False)
#         parameters_true =  pm.wfd.noisy_waveforms_parameters[idx]
#         for n in range(nparams):
#             percentiles[idx, n] = stats.percentileofscore(samples[:,n], parameters_true[n])

#     parameter_labels = pm.wfd.parameter_labels

#     make_pp(percentiles, parameter_labels)

#     np.savez_compressed('/userhome/O2/output/' + event_name + '/pp_{}_{}_{}.npz'.format(event, testd[i+1], version), parameter_labels=parameter_labels, percentiles=percentiles)
