import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse import coo_matrix


data_dict = np.load('data/train/text8/text8_top_10000_w_2_data.npz')

ind_tup = (data_dict['indices'][0].tolist(),data_dict['indices'][1].tolist())
val_tup = np.log(1.0+data_dict['vals'])

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

X = coo_matrix((val_tup, ind_tup), shape=data_dict['size'])
model = NMF(verbose=True,n_components=50, init='nndsvda', random_state=0,max_iter=300,beta_loss='kullback-leibler',solver='mu')
W = model.fit_transform(X)
save_string = 'word_vec/nmf_text8_top_10000_w_2_k_50_x_0.npz'
print save_string
np.savez(save_string,E_DK_M=W)