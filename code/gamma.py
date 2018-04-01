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
from sklearn.decomposition import NMF
from scipy.sparse import coo_matrix

from argparse import ArgumentParser
from utils import *

class BPTF(BaseEstimator, TransformerMixin):
    def __init__(self, n_modes, n_components,  max_iter, tol,
                 smoothness, verbose, a_pri=10.0, c_pri=2.0, d_pri=2.0):
        self.n_modes = n_modes
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.verbose = verbose

        self.a_pri = a_pri                                    # shape hyperparameter
        self.c_pri = c_pri
        self.d_pri = d_pri

        self.gamma_DK_M = np.empty(self.n_modes, dtype=object)  # variational shapes
        self.delta_DK_M = np.empty(self.n_modes, dtype=object)  # variational rates

        self.E_DK_M = np.empty(self.n_modes, dtype=object)      # arithmetic expectations
        self.G_DK_M = np.empty(self.n_modes, dtype=object)      # geometric expectations

    def _nndsvd(self,data):
        
        X = coo_matrix((data.vals, data.subs), shape=data.shape)
        model = NMF(verbose=True,n_components=self.n_components, init='nndsvda', random_state=0,max_iter=1,beta_loss='kullback-leibler',solver='mu')
        W = model.fit_transform(X)
        self.G_DK_M[0] = W.copy()
        self.E_DK_M[0] = W.copy()
        self.G_DK_M[1] = model.components_.T.copy()
        self.E_DK_M[1] = model.components_.T.copy()
        del X
        del W
        del model 

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
        assert len(mode_dims) == self.n_modes
        self.mode_dims = mode_dims
        for m, D in enumerate(mode_dims):
            self._init_component(m, D)

    def _init_component(self, m, dim):

        assert self.mode_dims[m] == dim
        K = self.n_components
        self.gamma_DK_M[m] = np.empty((dim,K))
        self.delta_DK_M[m] = np.empty((dim,K))

    def _check_component(self, m):
        assert np.isfinite(self.E_DK_M[m]).all()
        assert np.isfinite(self.G_DK_M[m]).all()
        assert np.isfinite(self.gamma_DK_M[m]).all()
        assert np.isfinite(self.delta_DK_M[m]).all()

    def _update_gamma(self, m, data):

        tmp = 1.0 / self._reconstruct_nz(data.subs,self.G_DK_M)
        uttkrp_DK = sp_uttkrp(tmp, data.subs, m, self.G_DK_M)
        self.gamma_DK_M[m][:, :] = self.a_pri * self.G_DK_M[m] * uttkrp_DK + self.c_pri

    def _update_delta(self, m, data):
        
        self.delta_DK_M[m][:, :] = self.d_pri + sp_uttkrp(data.vals, data.subs, m, self.E_DK_M)

    def _update_cache(self, m):

        self.E_DK_M[m] = self.gamma_DK_M[m] / self.delta_DK_M[m]
        self.G_DK_M[m] = np.exp(sp.psi(self.gamma_DK_M[m]))/self.delta_DK_M[m]

    def _update(self, data,orig_data=None ,modes=None):
        if modes is not None:
            modes = list(set(modes))
        else:
            modes = range(self.n_modes)
        assert all(m in range(self.n_modes) for m in modes)

        curr_elbo = -np.inf
        for itn in xrange(self.max_iter):
            s = time.time()
            for m in modes:
                self._update_gamma(m, data)
                self._update_delta(m, data)
                self._update_cache(m)
                self._check_component(m)

            bound = self.mae_nz(data)
            delta = (curr_elbo - bound) if itn > 0 else np.nan
            e = time.time() - s
            if self.verbose:
                print 'ITERATION %d:    Time: %f   Objective: %.2f    Change: %.5f'% (itn, e, bound, delta)

            curr_elbo = bound
            #if delta < self.tol:
            #    break

    def fit(self, data):

        self._nndsvd(data)
        self._init_all_components(data)
        self._update(data)
        return self

    def mae_nz(self,data):

        nz_recon_I = self._reconstruct_nz(data.subs,self.G_DK_M)
        nz_recon_I = self.a_pri/nz_recon_I
        return ((np.absolute(data.vals-nz_recon_I)).sum())/data.vals.size


def main():
    p = ArgumentParser()
    p.add_argument('-d', '--data', type=path, required=True)
    p.add_argument('-o', '--out', type=path, required=True)
    p.add_argument('-k', '--n_components', type=int, required=True)
    p.add_argument('-n', '--max_iter', type=int, default=200)
    p.add_argument('-t', '--tol', type=float, default=1e-3)
    p.add_argument('-s', '--smoothness', type=int, default=100)
    p.add_argument('-v', '--verbose', action="store_true", default=False)
    p.add_argument('-i', '--info',required=True)
    p.add_argument('-x', '--trunc', type=int, default=0)


    args = p.parse_args()

    data_dict = np.load(args.data)         # contain ['indices','vals','size'] as keys
    ind_tup = (data_dict['indices'][0].tolist(),data_dict['indices'][1].tolist())
    val_tup = data_dict['vals']
    
    if args.trunc != 0:
        val_tup = np.where(val_tup>args.trunc, args.trunc, val_tup)

    val_tup = np.log(val_tup + 1.0)
    val_tup = val_tup.tolist()    

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
                    verbose=args.verbose)

    bptf.fit(data)
    e = time.time()
    print "Training time = %d"%(e-s)
    serialize_bptf(bptf, args.out, desc=args.info,algo_name="gamma")


if __name__ == '__main__':
    main()
