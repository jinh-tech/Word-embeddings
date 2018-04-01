
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


dataset = "text8"
path = "../data/train/text8/"
f = open(path+dataset,'r')
line = f.readline().split()
# line = f.readline()
f.close()
# path = '.'
# dataset = 'a'
# line = "hello my name is nitish and i think this will work correctly lolol".split()


# In[3]:


print len(line)
# for i in range(0,1000):
#     print line[i],


# In[4]:


word_to_ind = {}
frq_word = []
c = 0
for i in line:
    if not (i in word_to_ind):
        word_to_ind[i] = c
        frq_word.append(1)
        c += 1
    else:
        frq_word[word_to_ind[i]] += 1
print("Vocabulary size of the dataset %d" %(c))
# print frq_word
# print word_to_ind
frq_word = np.array(frq_word,dtype=np.int32)
ind_frq_word = np.argsort(frq_word)
ind_to_word = {v: k for k, v in word_to_ind.iteritems()}


# In[5]:


windows = (5,)   #window sizes
no_words = [10000,]     #number of most frequent words

for c_window in windows:
    
    print "Window size = %d"%c_window
    indices = {}
    
    for j in range(1,c_window+1):
        for i in xrange(0,len(line)-j):
       
            temp_ind = (word_to_ind[line[i]],word_to_ind[line[i+j]])
            if temp_ind in indices:
                indices[temp_ind] += 1
            else:
                indices[temp_ind] = 1

            temp_ind = (word_to_ind[line[i+j]],word_to_ind[line[i]])
            if temp_ind in indices:
                indices[temp_ind] += 1
            else:
                indices[temp_ind] = 1
               
    print "Number of non-zero entries, max index, min index"            
    print len(indices.keys()) , max(indices.values()),min(indices.values())
    for top_words in no_words:

        rand_ind = np.arange(0,top_words)
        np.random.shuffle(rand_ind)
        c = 0
        temp_word_to_ind = {}

        for ind in xrange(len(ind_frq_word)-1,len(ind_frq_word)-top_words-1,-1):
            temp_word_to_ind[ind_to_word[ind_frq_word[ind]]] = rand_ind[c]
            c += 1
        subs = [[],[]]
        vals = []
        for ind,val in indices.iteritems():
            if(ind_to_word[ind[0]] in temp_word_to_ind and ind_to_word[ind[1]] in temp_word_to_ind):
                subs[0].append(temp_word_to_ind[ind_to_word[ind[0]]])
                subs[1].append(temp_word_to_ind[ind_to_word[ind[1]]])
                vals.append(val)

        with open(path+dataset+"_top_"+str(top_words)+"_w_"+str(c_window)+"_data.npz", 'w+') as f:
            np.savez(f, indices=subs, vals=vals, size=(top_words,top_words))
        f.close()
        with open(path+dataset+"_top_"+str(top_words)+"_w_"+str(c_window)+"_dict.txt", 'w+') as f:
            for ind,val in temp_word_to_ind.iteritems():
                f.write(str(ind)+" "+str(val)+"\n")
        f.close()
        print("Finished for no_words = %d"%top_words)

