import argparse
import numpy as np
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', required=True, type=str)
    parser.add_argument('--vectors_file', required=True, type=str)
    parser.add_argument('--write_out',type=str,default='False')
    args = parser.parse_args()


    with open(args.vocab_file, 'r') as f:
        words= []
        vocab = {}
        ivocab = {}
        for x in f.readlines():
            x = x.split(' ')
            words.append(x[0])
            vocab[x[0]] = int(x[1])
            ivocab[int(x[1])] = x[0]

    W = np.load(args.vectors_file)['G_DK_M'][0]
    vocab_size = W.shape[0]
    vector_dim = W.shape[1]

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    tot_ques, sem_acc, syn_acc, tot_acc = evaluate_vectors(W_norm, vocab, ivocab)
    
    if args.write_out != 'False':
        
        dataset = re.search(r'^[^/]*/[^_]*_(.*)_top_', args.vectors_file).group(1)
        vocab = re.search(r'^.*_top_([0-9]*)_w', args.vectors_file).group(1)
        algo = re.search(r'^[^/]*/([^_]*)_', args.vectors_file).group(1)
        window = re.search(r'.*_w_([0-9]*)_', args.vectors_file).group(1)
        v_dim = re.search(r'.*_k_([0-9]*)_', args.vectors_file).group(1)
        t_val = re.search(r'.*_x_([0-9]*)\.', args.vectors_file).group(1)
        file_tested = re.search(r'^[^/]*/(.*\.npz)', args.vectors_file).group(1)

        result_file = open('results/word_analogy_results.csv','a')
        tested_file = open('word_vec/tested.txt','a')
        
        result_file.write(algo+', '+dataset+', '+vocab+', '+window+', '+v_dim+', '+t_val+', '+str(tot_ques)+', '+str(sem_acc)+', '+str(syn_acc)+', '+str(tot_acc)+'\n')
        tested_file.write(file_tested+'\n')
        result_file.close()
        tested_file.close()



def evaluate_vectors(W, vocab, ivocab):
    """Evaluate the trained word vectors on a variety of tasks"""

    filenames = [
        'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
        'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
        'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
        'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
        'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
        ]
    prefix = 'data/test/word_analogy/'

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0; # count correct semantic questions
    correct_syn = 0; # count correct syntactic questions
    correct_tot = 0 # count correct questions
    count_sem = 0; # count all semantic questions
    count_syn = 0; # count all syntactic questions
    count_tot = 0 # count all questions
    full_count = 0 # count all questions, including those with unknown words

    for i in range(len(filenames)):
        with open('%s/%s' % (prefix, filenames[i]), 'r') as f:
            full_data = [line.rstrip().split(' ') for line in f]
            full_count += len(full_data)
            data = [x for x in full_data if all(word in vocab for word in x)]

        indices = np.array([[vocab[word] for word in row] for row in data])
        ind1, ind2, ind3, ind4 = indices.T

        predictions = np.zeros((len(indices),))
        num_iter = int(np.ceil(len(indices) / float(split_size)))
        for j in range(num_iter):
            subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))

            pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
                +  W[ind3[subset], :])
            #cosine similarity if input W has been normalized
            dist = np.dot(W, pred_vec.T)

            for k in range(len(subset)):
                dist[ind1[subset[k]], k] = -np.Inf
                dist[ind2[subset[k]], k] = -np.Inf
                dist[ind3[subset[k]], k] = -np.Inf

            # predicted word index
            predictions[subset] = np.argmax(dist, 0).flatten()

        val = (ind4 == predictions) # correct predictions
        count_tot = count_tot + len(ind1)
        correct_tot = correct_tot + sum(val)
        if i < 5:
            count_sem = count_sem + len(ind1)
            correct_sem = correct_sem + sum(val)
        else:
            count_syn = count_syn + len(ind1)
            correct_syn = correct_syn + sum(val)

        print("%s:" % filenames[i])
        print('ACCURACY TOP1: %.2f%% (%d/%d)' %
            (np.mean(val) * 100, np.sum(val), len(val)))

    print('Questions seen/total: %.2f%% (%d/%d)' %
        (100 * count_tot / float(full_count), count_tot, full_count))
    print('Semantic accuracy: %.2f%%  (%i/%i)' %
        (100 * correct_sem / float(count_sem), correct_sem, count_sem))
    print('Syntactic accuracy: %.2f%%  (%i/%i)' %
        (100 * correct_syn / float(count_syn), correct_syn, count_syn))
    print('Total accuracy: %.2f%%  (%i/%i)' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))
    return count_tot, 100 * correct_sem / float(count_sem), 100 * correct_syn / float(count_syn), 100 * correct_tot / float(count_tot)

if __name__ == "__main__":
    main()
