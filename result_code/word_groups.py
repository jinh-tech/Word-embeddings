import numpy as np

vec_prefix = "../word_vec/"
vec_file = "text8_top_10000_w_2_k_50.npz"

dict_prefix = "../data/train/text8/"
dict_file = "text 8_top_10000_w_2_dict.txt"

top_words = 10

W = np.load(vec_prefix+vec_file)['G_DK_M'][0]
vocab_size = W.shape[0]
vector_dim = W.shape[1]
d = (np.sum(W ** 2, 1) ** (0.5))
W = (W.T / d).T

ind_to_word = {}
with open(dict_prefix + dict_file, 'r') as f:
        for x in f.readlines():
            x = x.split(' ')
            ind_to_word[int(x[1])] = x[0]

topic_words = []            

for k in xrange(0,vector_dim):
	
	top_ind = W[:,k].argsort()[vocab_size-top_words : vocab_size][::-1]
	topic_words.append([ind_to_word[i] for i in top_ind])

for k in xrange(0,vector_dim):
	print topic_words[k]
	print "----------------------------------------------"