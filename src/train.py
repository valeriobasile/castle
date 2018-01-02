#!/usr/bin/env python
import logging as log
import sys
import numpy as np
from random import seed, random, shuffle
from scipy.spatial.distance import cosine
from tqdm import tqdm
from itertools import izip, product
import matplotlib.pyplot as plt
import numpy.random as rnd
from math import ceil

LANG = 'en'
with open('dependency_list.conll2009.semantic.short') as f:
    dependencies = map(lambda x: x.strip(), f.readlines())
#dependencies = ['nsubja', 'nsubjv', 'nsubjr', 'nsubjn', 'nsubjpass', 'dobj', 'iobj']

# configuration
np.seterr(divide='ignore', invalid='ignore')
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S')

# avoid np.inner
def dotp(x,y):
    return sum(x[:]*y[:])


class VectorPairReader(object):
    def __init__(self, datafile, cutoff=None):
        self.datafile = datafile
        self.cutoff = cutoff

    def __iter__(self):
        with open(self.datafile)as f:
            i = 0
            for line in f:
                fields = line.rstrip().split(' ')
                vector1 = map(lambda x: np.longdouble(eval(x)), fields[:len(fields)/2])
                vector2 = map(lambda x: np.longdouble(eval(x)), fields[len(fields)/2:])
                i += 1
                if self.cutoff and i > self.cutoff:
                    break
                yield vector1, vector2

    def __len__(self):
        i = 0
        with open(self.datafile)as f:
            for line in f:
                i += 1
        return i

# read arguments
data_dir = sys.argv[1]
target_dependency = sys.argv[2]
dimensions = eval(sys.argv[3])
learningrate = eval(sys.argv[4])
adversarial = eval(sys.argv[5])

# initialize random matrix
M = rnd.rand(dimensions, dimensions)

# read training data
training_pairs = dict()
for dependency in dependencies:
    training_pairs[dependency] = []
    if dependency == target_dependency or adversarial > 0.0:
        input_file = "{0}/{1}.{2}.txt".format(data_dir, dependency, LANG)
        log.info ("reading input file {0}".format(input_file))
        dr = VectorPairReader(input_file)
        for v1, v2 in dr:
            training_pairs[dependency].append((v1, v2))

# balance datasets
if adversarial > 0.0:
    log.info ("balancing training data")
    max_len = max([len(vector_pairs) for dependency, vector_pairs in training_pairs.iteritems()])
    training_data = []
    for dependency, vector_pairs in training_pairs.iteritems():
        if dependency == target_dependency:
            data = [(dependency, v1, v2) for v1, v2 in vector_pairs]
        else:
            if len(vector_pairs)>0:
                repeat = max_len/len(vector_pairs)
            else:
                repeat = max_len
            data = [(dependency, v1, v2) for v1, v2 in (vector_pairs * repeat)][:len(training_pairs[target_dependency])/(len(dependencies)-1)]
        log.info ("adding {0} pairs ({1})".format(len(data), dependency))
        training_data.extend(data)

else:
    training_data = [(target_dependency, v1, v2) for v1, v2 in training_pairs[target_dependency]]

log.info ("shuffling training data")
shuffle(training_data)

#training
error_plot = []
epoch = 0
past_error = 0.0
log.info ("starting training")
while True:
    iteration = 0
    error = 0.0
    for dependency, h, t in training_data:
        # compute gradient analytically
        x = np.dot(M,t)
        a = (dotp(h,x) * np.outer(x,t)) / (dotp(x,x)**1.5)
        b = np.outer(h, t) / (dotp(x,x)**0.5)

        # gradient update
        G = a-b

        # update the tensor
        if dependency == target_dependency:
            M = np.subtract(M, (learningrate * G))
        else:
            M = np.add(M, (learningrate * adversarial * G))

        iteration += 1

        # plot error
        error += cosine(x, h)

    avg_error = error/float(iteration)
    error_plot.append(avg_error)
    error_delta = avg_error - past_error
    past_error = avg_error

    # log progress
    log.info('epoch {0}, {1} iterations, error: {2:.3f} ({3:.4f})'.format(epoch+1, iteration, avg_error, error_delta))

    epoch += 1

    #if epoch > 100 or (error_delta > -0.0001 and epoch > 1):
    #if error_delta > -0.0001 and epoch > 1:
    if epoch > 100:
        break
    


# plot error rate
'''
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(list(range(len(error_plot))), error_plot, alpha=0.1, color='blue')

ax.set_xlabel("i")
ax.set_ylabel("error")

ax.set_xlim([0,len(error_plot)])
ax.set_ylim([0,1])

ax.set_title("Error plot")
fig_name = "error_{0}.png".format(learningrate)
fig.savefig(fig_name)
'''
# write tensor output
for j in M:
    print ' '.join(map(str, j))
