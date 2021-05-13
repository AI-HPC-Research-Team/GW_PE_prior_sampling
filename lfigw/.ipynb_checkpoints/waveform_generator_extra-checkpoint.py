from .reduced_basis import SVDBasis
from .waveform_generator import WaveformDataset
from imblearn.combine import SMOTETomek # pip install imblearn
import pandas as pd
import numpy as np
from tqdm import tqdm

def oversampling(x: np.array, threshold=512, random_state=0):
    num = len(x)//4
    cache = []
    while True:
        Input = pd.DataFrame(x).sample(2*num).values
        fitfor = Input, [1,]*(num)+[0,]*(num)
        smote_tomek = SMOTETomek(random_state=random_state)
        Output_tomek, _ = smote_tomek.fit_resample(*fitfor)            
        cache.append(pd.concat([pd.DataFrame(Input), pd.DataFrame(Output_tomek)]).drop_duplicates(keep=False).values)
        if len(np.concatenate(cache)) > threshold:
            break
    return np.concatenate(cache)

class WaveformDataset_extra(WaveformDataset):

    def _load_posterior(self, event, sample_extrinsic_only=True):
        print('sample_extrinsic_only:', sample_extrinsic_only)
        df = pd.read_csv('./bilby_runs/downsampled_posterior_samples_v1.0.0/{}_downsampled_posterior_samples.dat'.format(event), sep=' ')
        self.bilby_samples = df.dropna()[['mass_1', 'mass_2', 'phase', 'geocent_time', 'luminosity_distance',
                              'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                              'theta_jn', 'psi', 'ra', 'dec']].values.astype('float64')
        self.bilby_samples[:,3] = self.bilby_samples[:,3] - self.ref_time
        self.sample_extrinsic_only = sample_extrinsic_only
        self.bilby_samples_extrisinc = self.bilby_samples[:,[3,4,12,13,14]]

    def _check_prior(self):
        assert self.nparams == 15
        assert len(self.extrinsic_params) == 5

    def _sample_prior(self, n):
        """Obtain samples from the target posterior distribution.

        Arguments:
            n {int} -- number of samples

        Returns:
            array -- samples
        """
        self._check_prior()
        return oversampling(self.bilby_samples, threshold=n)[:n]

    def _cache_oversampled_parameters(self, nsample):
        self._check_prior()
        self.ncache_parameters = nsample
        self.cache_parameters = oversampling(self.bilby_samples, threshold=len(self.parameters))
        self.cache_parameters_extrinsic = oversampling(self.bilby_samples_extrisinc, threshold=nsample)[:nsample]
    
    def sample_prior_extrinsic(self, n):
        """Draw samples of extrinsic parameters from the posterior prior.

        Arguments:
            n {int} -- number of prior samples

        Returns:
            array -- n x m array of samples, where m is number of extrinsic
                     parameters
        """        
        assert n == 1
        return self.cache_parameters_extrinsic[np.random.randint(self.ncache_parameters)][np.newaxis,...].astype(np.float32)

    def _compute_parameter_statistics(self):
        """Compute mean and standard deviation for physical parameters, in
        order to standardize later.
        (Do not use analytic expressions )
        """
        #parameters_train = self.parameters[self.train_selection]
        self.parameters_mean = np.mean(self.bilby_samples, axis=0).astype(np.float32)
        self.parameters_std = np.std(self.bilby_samples, axis=0).astype(np.float32)

    def _resample_distance(self, p, h_det):
        """(Do not) Resample the luminosity distance for a waveform based on
        new distance prior and / or an SNR threshold.

        Arguments:
            p {array} -- initial parameters
            h_det {dict} -- initial detector waveforms

        Returns:
            array -- ori parameters, with distance resampled
            dict -- ori detector waveforms
            float -- constant 1
        """        
        self._check_prior()
        return p, h_det, 1.0 # weight

    def generate_reduced_basis(self, n_train=10000, n_test=10000):
        """Generate the reduced basis elements.

        This draws parameters from the prior, generates detector waveforms,
        and trains the SVD basis based on these.

        It then evaluates performance on the training waveforms, and a set
        of validation waveforms.

        Keyword Arguments:
            n_train {int} -- number of training waveforms (default: {10000})
            n_test {int} -- number of test waveforms (default: {10000})
        """

        print('Generating {} detector FD waveforms for training reduced basis.'
              .format(n_train))

        h_detector = {}
        for ifo in self.detectors.keys():
            h_detector[ifo] = np.empty((n_train, self.Nf), dtype=np.complex64)
        
        pall = self._sample_prior(n_train)
        for i in tqdm(range(n_train)):
            p =  pall[i] # shape: (nparams, )
            # To generate reduced basis, fix all waveforms to same fiducial
            # distance.
            p[self.param_idx['distance']] = self.fiducial_params['distance']
            h_d = self._generate_whitened_waveform(p, intrinsic_only=False)
            for ifo, h in h_d.items():
                h_detector[ifo][i] = h

        print('Generating reduced basis for training detector waveforms')

        training_array = np.vstack(list(h_detector.values()))
        self.basis = SVDBasis()
        self.basis.generate_basis(training_array, n=self.Nrb)

        # print('Calculating standard deviations for training standardization.')
        # for ifo, h_array_FD in h_detector.items():

        #     # Project training data for given ifo onto reduced basis
        #     h_array_RB = np.empty((n_train, self.Nrb), dtype=np.complex64)
        #     for i, h in enumerate(h_array_FD):
        #         h_array_RB[i] = self.basis.fseries_to_basis_coefficients(h)

        #     # Compute standardization for given ifo
        #     self.basis.init_standardization(ifo, h_array_RB, self._noise_std)

        print('Evaluating performance on training set waveforms.')
        matches = []
        for h_FD in tqdm(training_array):
            h_RB = self.basis.fseries_to_basis_coefficients(h_FD)
            h_reconstructed = self.basis.basis_coefficients_to_fseries(h_RB)

            norm1 = np.mean(np.abs(h_FD)**2)
            norm2 = np.mean(np.abs(h_reconstructed)**2)
            inner = np.mean(h_FD.conj()*h_reconstructed).real

            matches.append(inner / np.sqrt(norm1 * norm2))
        mismatches = 1 - np.array(matches)
        print('  Mean mismatch = {}'.format(np.mean(mismatches)))
        print('  Standard deviation = {}'.format(np.std(mismatches)))
        print('  Max mismatch = {}'.format(np.max(mismatches)))
        print('  Median mismatch = {}'.format(np.median(mismatches)))
        print('  Percentiles:')
        print('    99    -> {}'.format(np.percentile(mismatches, 99)))
        print('    99.9  -> {}'.format(np.percentile(mismatches, 99.9)))
        print('    99.99 -> {}'.format(np.percentile(mismatches, 99.99)))

        # Evaluation on test waveforms

        print('Generating {} detector FD waveforms for testing reduced basis.'
              .format(n_test))

        h_detector = {}
        for ifo in self.detectors.keys():
            h_detector[ifo] = np.empty((n_test, self.Nf), dtype=np.complex64)

        pall = self._sample_prior(n_test)
        for i in tqdm(range(n_test)):
            p =  pall[i] # shape: (nparams, )     
            # To generate reduced basis, fix all waveforms to same fiducial
            # distance.
            p[self.param_idx['distance']] = self.fiducial_params['distance']
            h_d = self._generate_whitened_waveform(p, intrinsic_only=False)
            for ifo, h in h_d.items():
                h_detector[ifo][i] = h

        print('Evaluating performance on test set waveforms.')
        test_array = np.vstack(list(h_detector.values()))
        matches = []
        for h_FD in tqdm(test_array):
            h_RB = self.basis.fseries_to_basis_coefficients(h_FD)
            h_reconstructed = self.basis.basis_coefficients_to_fseries(h_RB)

            norm1 = np.mean(np.abs(h_FD)**2)
            norm2 = np.mean(np.abs(h_reconstructed)**2)
            inner = np.mean(h_FD.conj()*h_reconstructed).real

            matches.append(inner / np.sqrt(norm1 * norm2))
        mismatches = 1 - np.array(matches)
        print('  Mean mismatch = {}'.format(np.mean(mismatches)))
        print('  Standard deviation = {}'.format(np.std(mismatches)))
        print('  Max mismatch = {}'.format(np.max(mismatches)))
        print('  Median mismatch = {}'.format(np.median(mismatches)))
        print('  Percentiles:')
        print('    99    -> {}'.format(np.percentile(mismatches, 99)))
        print('    99.9  -> {}'.format(np.percentile(mismatches, 99.9)))
        print('    99.99 -> {}'.format(np.percentile(mismatches, 99.99)))
    
    
    
    
    
def method_is_overided_in_subclass(method_name: str, sub_class, base_class) -> bool:
    """Define whether subclass override specified method
    Args:
        method_name: to be defined method name
        sub_class: to be defined  sub class
        base_class: to be defined base class
    Returns:
        True or False
    Example:
        method_is_overided_in_subclass(method_name="sample_prior_extrinsic", 
                                       sub_class=WaveformDataset_extra,
                                       base_class=WaveformDataset)
    """
    assert issubclass(sub_class, base_class), "class %s is not subclass of class %s" % (
        sub_class,
        base_class,
    )
    this_method = getattr(sub_class, method_name)
    base_method = getattr(base_class, method_name)
    return this_method is not base_method
