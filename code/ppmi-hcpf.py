# %load bptf.py
"""
Bayesian Poisson tensor factorization with variational inference with gamma distribution below the poisson distribution.
"""
import sys
import time
import numpy as np
import numpy.random as rn
import scipy.special as sp
import sktensor as skt
from sklearn.base import BaseEstimator, TransformerMixin

#from path import path
from argparse import ArgumentParser
from utils import *


class BPTF(BaseEstimator, TransformerMixin):
    def __init__(self, n_modes, n_components,  max_iter, tol,smoothness, verbose, n_trunc, alpha_prime=0.1, beta_prime=0.1, alpha=0.1):
        self.n_modes = n_modes
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.verbose = verbose
        self.n_trunc = n_trunc
        self.log_fac = np.empty(shape=n_trunc,dtype=np.float64)
        self.log_fac_m = np.empty(shape=n_trunc,dtype=np.float64)
        self.log_fac[0] = 0.0
        self.log_fac_m[0] = self.log_fac_m[1] = 0.0
        for i in range(1,n_trunc):
            self.log_fac[i] = self.log_fac[i-1] + np.log(i+1)
        for i in range(2,n_trunc):
            self.log_fac_m[i] = self.log_fac[i-1]

        self.alpha = alpha                                      # shape hyperparameter
        self.alpha_prime = alpha_prime
        self.beta_prime = beta_prime

        self.gamma_DK_M = np.empty(self.n_modes, dtype=object)  # variational shapes
        self.delta_DK_M = np.empty(self.n_modes, dtype=object)  # variational rates

        self.E_DK_M = np.empty(self.n_modes, dtype=object)      # arithmetic expectations
        self.G_DK_M = np.empty(self.n_modes, dtype=object)      # geometric expectations

        self.kappa_shp = np.empty(self.n_modes,dtype=object)
        self.kappa_rte = np.empty(self.n_modes,dtype=object)        
        # Inference cache
        self.sumE_MK = np.empty((self.n_modes, self.n_components), dtype=float)

    def _reconstruct_nz(self, subs_I_M, G_DK_M):
        """Computes the reconstruction for only non-zero entries."""
        I = len(subs_I_M[0])
        K = self.n_components
        nz_recon_IK = np.ones((I, K))
        for m in xrange(self.n_modes):
            nz_recon_IK *= G_DK_M[m][subs_I_M[m], :]
        return nz_recon_IK.sum(axis=1)

    def _init_all_components(self, data):
        mode_dims = data.shape
        self.log_data = np.log(data.vals)[np.newaxis,:]
        self.E_N = np.ones_like(data.vals)
        self.G_B = np.ones_like(data.vals)
        assert len(mode_dims) == self.n_modes
        self.mode_dims = mode_dims
        for m, D in enumerate(mode_dims):
            self._init_component(m, D)

    def _init_component(self, m, dim):
        assert self.mode_dims[m] == dim
        K = self.n_components
        s = self.smoothness
        gamma_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
        delta_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
        self.gamma_DK_M[m] = gamma_DK
        self.delta_DK_M[m] = delta_DK
        self.E_DK_M[m] = gamma_DK / delta_DK
        self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.G_DK_M[m] = np.exp(sp.psi(gamma_DK) - np.log(delta_DK))
        self.kappa_shp[m] = np.ones(dim,dtype=np.float64) * (self.alpha_prime + K*self.alpha)
        self.kappa_rte[m] = np.ones(dim,dtype=np.float64)

    def _check_component(self, m):
        assert np.isfinite(self.E_DK_M[m]).all()
        assert np.isfinite(self.G_DK_M[m]).all()
        assert np.isfinite(self.gamma_DK_M[m]).all()
        assert np.isfinite(self.delta_DK_M[m]).all()

    def _update_gamma(self, m, data):
 
        tmp = self.E_N / self._reconstruct_nz(data.subs,self.G_DK_M)
        uttkrp_DK = sp_uttkrp(tmp, data.subs, m, self.G_DK_M)
        self.gamma_DK_M[m][:, :] = self.alpha + self.G_DK_M[m] * uttkrp_DK

    def _update_delta(self, m, mask=None):

        self.sumE_MK[m, :] = 1.
        uttrkp_DK = self.sumE_MK.prod(axis=0)
        if uttrkp_DK.shape == (self.n_components,):
            uttrkp_DK = uttrkp_DK.reshape((1,-1))
            uttrkp_DK = uttrkp_DK.repeat(self.mode_dims[m],axis=0)
        self.delta_DK_M[m][:, :] = (self.kappa_shp[m]/self.kappa_rte[m])[:,np.newaxis] + uttrkp_DK

    def _update_kappa(self,m):
        
        self.kappa_rte[m] = self.beta_prime + self.E_DK_M[m].sum(axis=1)

    def _update_N(self,data):

        n_trunc_range = np.arange(self.n_trunc) + 1.0
        temp = self.G_B + np.log(self._reconstruct_nz(data.subs,self.G_DK_M))
        #temp = np.log(2.0) + np.log(self._reconstruct_nz(data.subs,self.E_DK_M))
        prob = (np.ones((self.n_trunc,data.vals.shape[0]))*temp)*n_trunc_range[:,np.newaxis]
        temp = n_trunc_range - 1.0
        prob += np.multiply(temp[:,np.newaxis],self.log_data) - self.log_fac[:,np.newaxis] - self.log_fac_m[:,np.newaxis]
        m = np.max(prob,0)
        log_sum = m + np.log(np.exp(prob - m).sum(axis=0))
        prob = np.exp(prob - log_sum)
        self.E_N = (prob*n_trunc_range[:,np.newaxis]).sum(axis=0)

    def _update_B(self,data):

        self.E_B = (self.E_N + 2.0) / (data.vals + 2.0)
        self.G_B = sp.psi(self.E_N + 2.0) - np.log(data.vals + 2.0)     # RHS should be equal to log(self.G_B)

    def _update_cache(self, m):
        gamma_DK = self.gamma_DK_M[m]
        delta_DK = self.delta_DK_M[m]
        self.E_DK_M[m] = gamma_DK / delta_DK
        self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.G_DK_M[m] = np.exp(sp.psi(gamma_DK)) / delta_DK

    def _update(self, data, mask=None, modes=None):
        if modes is not None:
            modes = list(set(modes))
        else:
            modes = range(self.n_modes)
        assert all(m in range(self.n_modes) for m in modes)

        curr_elbo = -np.inf
        for itn in xrange(self.max_iter):
            s = time.time()
            self._update_N(data)
            self._update_B(data)
            for m in modes:
                self._update_gamma(m, data)
                self._update_delta(m, mask)
                self._update_cache(m)
                self._update_kappa(m)
                self._check_component(m)

            bound = np.absolute((self.E_N/self.E_B) - data.vals).sum()/data.vals.size  #self.mae_nz(data)
            # bound = (np.absolute((self.E_N/2.0) - data.vals)).sum()/data.vals.size  #self.mae_nz(data)
            delta = (curr_elbo - bound) if itn > 0 else np.nan
            e = time.time() - s
            if self.verbose:
                print 'ITERATION %d:    Time: %f   Objective: %.2f    Change: %.5f'% (itn, e, bound, delta)

            curr_elbo = bound
            # if delta < self.tol:
                # break

    def fit(self, data):
        self._init_all_components(data)
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
    p.add_argument('-t', '--tol', type=float, default=1e-4)
    p.add_argument('-s', '--smoothness', type=int, default=100)
    p.add_argument('-a', '--alpha', type=float, default=0.1)
    p.add_argument('-v', '--verbose', action="store_true", default=False)
    p.add_argument('-i', '--info',required=True)
    p.add_argument('-x', '--trunc', type=int, default=0)
    p.add_argument('-r', '--n_trunc',type=int, required=True)

    args = p.parse_args()

    data_dict = np.load(args.data)         # contain ['indices','vals','size'] as keys
    ind_tup = (data_dict['indices'][0].tolist(),data_dict['indices'][1].tolist())
    val_tup = data_dict['vals']
    
    if args.trunc != 0:
        val_tup = np.where(val_tup>args.trunc, args.trunc, val_tup)
    #val_tup = np.log(val_tup + 1.0)
    val_tup = val_tup.tolist()

    ind_to_sum = {}

    for i in range(0,len(val_tup)):
        if ind_tup[0][i] in ind_to_sum:
            ind_to_sum[ind_tup[0][i]] += val_tup[i]
        else:
            ind_to_sum[ind_tup[0][i]] = val_tup[i]

    total_sum = 0
    for ind,val in ind_to_sum.iteritems():
        total_sum += val

    assert total_sum == sum(val_tup)

    new_ind_tup = ([],[])
    new_val_tup = []

    for i in range(0,len(val_tup)):

        temp = np.log(val_tup[i]) + np.log(total_sum) - np.log(ind_to_sum[ind_tup[0][i]]) - np.log(ind_to_sum[ind_tup[1][i]])
        if temp > 0:
            new_val_tup.append(temp)
            new_ind_tup[0].append(ind_tup[0][i])
            new_ind_tup[1].append(ind_tup[1][i])

    ind_tup = new_ind_tup
    val_tup = new_val_tup

    data = skt.sptensor(ind_tup,val_tup,shape= data_dict['size'].tolist(), dtype=np.float64)
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
                    alpha=args.alpha,
                    n_trunc=args.n_trunc)

    bptf.fit(data)
    e = time.time()
    print "Training time = %d"%(e-s)
    serialize_bptf(bptf, args.out, desc=args.info, algo_name="ppmi-hcpf")
    # np.savez("test_stuff/hcpf.npz", vals=data.vals, E_N = bptf.E_N, E_B = bptf.E_B)


if __name__ == '__main__':
    main()

