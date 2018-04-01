import numpy as np
import nimfa
from scipy.sparse import coo_matrix

k=50
data_dict = np.load('data/train/text8/text8_top_10000_w_2_data.npz')
ind_tup = (data_dict['indices'][0].tolist(),data_dict['indices'][1].tolist())
val_tup = np.log(1.0+data_dict['vals'])


X = coo_matrix((val_tup, ind_tup), shape=data_dict['size'])
mat = nimfa.(X,rank=k)
print type(mat)

