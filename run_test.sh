from os import listdir
from os.path import isfile, join
from subprocess import call

prefix_vec = "word_vec/"
prefix_res = "results/"
prefix_data = "data/train/"

all_files = [f for f in listdir(prefix_vec) if isfile(join(mypath, f)) and f.endswith(".npz")]
print all_files

with f as open(prefix_res+"tested.csv",'r'):
	tested_files = [s.rstrip('\n') for s in f.readlines()]

for i in list(set(all_files)-set(tested_files)):
	call(["python", "eval/evaluate.py", "--vocab_file=", "vectors_file="+i])

#echo python eval/evaluate.py --vocab_file=data/train/text8/text8_top_10000_w_2_dict.txt --vectors_file=word_vec/text8_top_10000_w_2_k_50_x_10.npz
#python eval/evaluate.py --vocab_file=data/train/text8/text8_top_10000_w_2_dict.txt --vectors_file=word_vec/text8_top_10000_w_2_k_50_x_10.npz
