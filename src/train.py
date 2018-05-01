#!/usr/bin/env python3
import logging as log
import sys
import numpy as np
from random import seed, random, shuffle, sample
from scipy.spatial.distance import cosine
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
import numpy.random as rnd
from math import ceil
from optparse import OptionParser
from os.path import abspath, dirname, isfile
from vector import VectorPairReader

parser = OptionParser()
parser.add_option('-i',
                  '--input',
                  dest="input_file",
                  help="read training vector pairs from FILE",
                  metavar="FILE")
parser.add_option('-d',
                  '--input-dir',
                  dest="input_dir",
                  help="read training vector pairs for contrastive learing from DIRECTORY",
                  metavar="DIRECTORY")
parser.add_option('-o',
                  '--output',
                  dest="output_file",
                  default='tensor.txt',
                  help="write tensor to FILE",
                  metavar="FILE")
parser.add_option('-l',
                  '--learning-rate',
                  dest="learningrate",
                  default="0.05",
                  help="learning rate for the gradient descent (default: 0.05)")
parser.add_option('-c',
                  '--contrastive',
                  default="0.0",
                  dest="contrastive",
                  help="parameter for contrastive learning (default: 0)")
parser.add_option('-e',
                  '--epochs',
                  dest="epochs",
                  default="100",
                  help="maximum number of training epochs (default: 100)")
parser.add_option('-t',
                  '--threshold',
                  dest="threshold",
                  default="0.0",
                  help="stop training when the error is below this threshold (default: 0)")
#TODO option for online learning without reading all the vectors into memory first

(options, args) = parser.parse_args()

# check option values
if options.input_file and isfile(options.input_file):
    input_file = options.input_file
else:
    log.error("invalid input file: {0}, exiting".format(options.input_file))
    parser.print_usage()
    sys.exit(1)

if not options.input_dir:
    input_dir = dirname(abspath(options.input_file))
else:
    input_dir = options.input_dir

if eval(options.learningrate) > 0.0:
    learningrate = eval(options.learningrate)
else:
    log.error("invalid value for learning rate: {0}, exiting".format(options.learningrate))
    parser.print_usage()
    sys.exit(1)

if eval(options.contrastive) >= 0.0 and eval(options.contrastive) <= 1.0:
    contrastive = eval(options.contrastive)
else:
    log.error("invalid value for contrastive factor: {0}, exiting".format(options.contrastive))
    parser.print_usage()
    sys.exit(1)

if eval(options.epochs) > 1:
    epochs = eval(options.epochs)
else:
    log.error("invalid value for epochs: {0}, exiting".format(options.epochs))
    parser.print_usage()
    sys.exit(1)

if eval(options.threshold) >= 0.0:
    threshold = eval(options.threshold)
else:
    log.error("invalid value for threshold: {0}, exiting".format(options.threshold))
    parser.print_usage()
    sys.exit(1)


# static configuration
np.seterr(divide='ignore', invalid='ignore')
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S')

# workaround to avoid np.inner for numerical precision reasons
def dotp(x,y):
    return sum(x[:]*y[:])

# read training data
training_pairs = []
log.info ("reading input file {0}".format(input_file))
dr = VectorPairReader(input_file)
for v1, v2 in dr:
    training_pairs.append((v1, v2))
dimensions = len(training_pairs[0][0])

contrastive_training_pairs = []
if contrastive > 0.0:
    for root, dirs, files in os.walk(input_dir, topdown=False):
        for contrastive_input_file in files:
            log.info ("reading input file {0} for constrastive learning".format(contrastive_input_file))
            dr = VectorPairReader(contrastive_input_file)
            for v1, v2 in dr:
                contrastive_training_pairs.append((v1, v2))

# initialize random matrix
M = rnd.rand(dimensions, dimensions)

# randomize order and balance datasets
if contrastive > 0.0:
    #print len(contrastive_training_pairs), len(training_pairs)
    log.info ("balancing training data")
    contrastive_training_pairs = sample(contrastive_training_pairs, len(training_pairs))
    training_data = [(1, v1, v2) for v1, v2 in training_pairs] + [(0, v1, v2) for v1, v2 in contrastive_training_pairs]
else:
    training_data = [(1, v1, v2) for v1, v2 in training_pairs]

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
    for sign, h, t in training_data:
        # compute gradient analytically
        x = np.dot(M,t)
        a = (dotp(h,x) * np.outer(x,t)) / (dotp(x,x)**1.5)
        b = np.outer(h, t) / (dotp(x,x)**0.5)

        # gradient update
        G = a-b

        # update the tensor
        if sign == 1:
            M = np.subtract(M, (learningrate * G))
        else:
            M = np.add(M, (learningrate * contrastive * G))

        error += cosine(x, h)
        iteration += 1

    avg_error = error/float(iteration)
    error_plot.append(avg_error)
    error_delta = avg_error - past_error
    past_error = avg_error

    # log progress
    log.info('epoch {0}, {1} iterations, error: {2:.3f} ({3:.4f})'.format(epoch+1, iteration, avg_error, error_delta))
    epoch += 1

    if epoch > epochs or (epoch >1 and error_delta > threshold):
        break

# write tensor output
with open(options.output_file, 'w') as fo:
    for j in M:
        fo.write("{0}\n".format(' '.join(map(str, j))))
