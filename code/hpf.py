"""
Bayesian Poisson tensor factorization with variational inference.
"""
import sys
import time
import numpy as np
import numpy.random as rn
import scipy.special as sp
import sktensor as skt
from sklearn.base import BaseEstimator, TransformerMixin

from argparse import ArgumentParser
from utils import *


class BPTF(BaseEstimator, TransformerMixin):
    def __init__(self, n_modes, n_components,  max_iter, tol,
                 smoothness, verbose, alpha):
        self.n_modes = n_modes-1
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.verbose = verbose

        self.alpha = alpha                                      # shape hyperparameter
        self.beta_M = np.ones(self.n_modes, dtype=float)        # rate hyperparameter (inferred)

        self.gamma_DK_M = np.empty(self.n_modes, dtype=object)  # variational shapes
        self.delta_DK_M = np.empty(self.n_modes, dtype=object)  # variational rates

        self.E_DK_M = np.empty(self.n_modes, dtype=object)      # arithmetic expectations
        self.G_DK_M = np.empty(self.n_modes, dtype=object)      # geometric expectations

        # Inference cache
        self.sumE_MK = np.empty((self.n_modes, self.n_components), dtype=float)
        
    def _reconstruct_nz(self, subs_I_M, G_DK_M):
        """Computes the reconstruction for only non-zero entries."""
        nz_recon_IK = G_DK_M[0][subs_I_M[0], :] * G_DK_M[0][subs_I_M[1], :]
        return nz_recon_IK.sum(axis=1)

    def _init_all_components(self, mode_dims):
        self.mode_dims = mode_dims[0]
        self._init_component(0, mode_dims[0])

    def _init_component(self, m, dim):
        K = self.n_components
        s = self.smoothness
        gamma_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
        delta_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
        self.gamma_DK_M[m] = gamma_DK
        self.delta_DK_M[m] = delta_DK
        self.E_DK_M[m] = gamma_DK / delta_DK
        self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.G_DK_M[m] = np.exp(sp.psi(gamma_DK) - np.log(delta_DK))
        self.beta_M[m] = 1. / self.E_DK_M[m].mean()

    def _check_component(self, m):
        assert np.isfinite(self.E_DK_M[m]).all()
        assert np.isfinite(self.G_DK_M[m]).all()
        assert np.isfinite(self.gamma_DK_M[m]).all()
        assert np.isfinite(self.delta_DK_M[m]).all()

    def _update_gamma(self, m, data):
        U = np.empty(2,dtype=object)
        U[0] = np.copy(self.G_DK_M[0])
        U[1] = np.copy(self.G_DK_M[0])
        tmp = data.vals / self._reconstruct_nz(data.subs,self.G_DK_M)
        uttkrp_DK = sp_uttkrp(tmp, data.subs, m, U)
        del U
        self.gamma_DK_M[m][:, :] = self.alpha + self.G_DK_M[m] * uttkrp_DK

    def _update_delta(self, m):
        self.delta_DK_M[m][:, :] = self.alpha * self.beta_M[m] + self.sumE_MK[m,:]

    def _update_cache(self, m):
        gamma_DK = self.gamma_DK_M[m]
        delta_DK = self.delta_DK_M[m]
        self.E_DK_M[m] = gamma_DK / delta_DK
        self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.G_DK_M[m] = np.exp(sp.psi(gamma_DK)- np.log(delta_DK))

    def _update_beta(self, m):
        self.beta_M[m] = 1. / self.E_DK_M[m].mean()

    def _update(self, data):

        curr_elbo = -np.inf
        for itn in xrange(self.max_iter):
            s = time.time()
            self._update_gamma(0, data)
            self._update_delta(0)
            self._update_cache(0)
            self._update_beta(0)  # must come after cache update!
            self._check_component(0)

            bound = self.mae_nz(data)
            delta = (curr_elbo - bound) if itn > 0 else np.nan
            e = time.time() - s
            if self.verbose:
                print 'ITERATION %d:    Time: %f   Objective: %.2f    Change: %.5f'% (itn, e, bound, delta)

            curr_elbo = bound
            if delta < self.tol:
                break

    def fit(self, data):
        self._init_all_components(data.shape)
        self._update(data)
        return self

    def mae_nz(self,data):

        nz_recon_I = self._reconstruct_nz(data.subs,self.G_DK_M)
        return ((np.absolute(data.vals-nz_recon_I)).sum())/data.vals.size

def main():
    p = ArgumentParser()
    p.add_argument('-d', '--data', type=path, required=True)
    p.add_argument('-o', '--out', type=path, required=True)
    p.add_argument('-k', '--n_components', type=int, required=True)
    p.add_argument('-n', '--max_iter', type=int, default=200)
    p.add_argument('-t', '--tol', type=float, default=1e-3)
    p.add_argument('-s', '--smoothness', type=int, default=100)
    p.add_argument('-a', '--alpha', type=float, default=0.1)
    p.add_argument('-v', '--verbose', action="store_true", default=False)
    p.add_argument('-i', '--info',required=True)

    args = p.parse_args()

    data_dict = np.load(args.data)         # contain ['indices','vals','size'] as keys
    ind_tup = (data_dict['indices'][0].tolist(),data_dict['indices'][1].tolist())
    val_tup = (data_dict['vals'].tolist())
    data = skt.sptensor(ind_tup,val_tup,shape= data_dict['size'].tolist(), dtype=np.int32)
    print("Number of non-zero entries = %d"%(len(data.vals)))
    del ind_tup
    del val_tup
    del data_dict

    s = time.time()
    bptf = BPTF(n_modes=data.ndim,
                   n_components=args.n_components,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    smoothness=args.smoothness,
                    verbose=args.verbose,
                    alpha=args.alpha)

    bptf.fit(data)
    e = time.time()
    print "Training time = %d"%(e-s)
    serialize_bptf(bptf, args.out, desc=args.info)


if __name__ == '__main__':
    main()