from os import listdir
from os.path import isfile, join
from subprocess import call
import re

prefix_vec = "word_vec/"
prefix_data = "data/train/"

pattern_dict = r'^[^_]*_(.*)_k_'
pattern_data = r'^[^_]*_(.*)_top_'

all_files = [f for f in listdir(prefix_vec) if isfile(join(prefix_vec, f)) and f.endswith(".npz")]

with open(prefix_vec+"tested.txt",'r') as f:
	tested_files = [s.rstrip('\n') for s in f.readlines()]

for i in list(set(all_files)-set(tested_files)):
	print "Testing vector %s"%i
	dict_vocab = re.search(pattern_dict,i).group(1) + "_dict.txt"
	dataset = re.search(pattern_data,i).group(1) + "/"
	call(["python", "eval/evaluate.py", "--vocab_file="+prefix_data + dataset + dict_vocab, "--vectors_file="+prefix_vec+i, "--write_out=True"])

