import numpy as np
import struct

path = 'data/train/text8/glove/'
file = 'cooccurrence.bin'

temp_f = open(path+file,'rb')
f = temp_f.read()
temp_f.close()
data_size = len(f)
s_format = 'I I d'
s = struct.Struct(s_format)

data = [[],[],[]]

for i in range(0,data_size,s.size):
	word_tuple = s.unpack(f[i:i+s.size])
	data[0].append(word_tuple[0]-1)
	data[1].append(word_tuple[1]-1)
	data[2].append(word_tuple[2])

print max(data[0]), max(data[1]), max(data[2])
print min(data[0]), min(data[1]), min(data[2])
print len(data[0])


temp_f = path + 'glovetext8_top_10000_w_2_data.npz'
np.savez(temp_f, indices=(data[0],data[1]), vals=data[2], size=(10000,10000))
