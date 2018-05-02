#!/usr/bin/env python
import logging as log
from optparse import OptionParser
import numpy as np
from vector import VectorPairReader
from scipy.spatial.distance import cosine
from os.path import isfile

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S')

# command line argument partsing
parser = OptionParser()
parser.add_option('-i',
                  '--input',
                  dest="input_file",
                  help="read training vector pairs from FILE",
                  metavar="FILE")
parser.add_option('-l',
                  '--labels',
                  dest="label_file",
                  help="read label pairs from FILE",
                  metavar="FILE")
parser.add_option('-t',
                  '--tensor',
                  dest="tensor_file",
                  default='tensor.txt',
                  help="read tensor from FILE",
                  metavar="FILE")
(options, args) = parser.parse_args()

if options.input_file and isfile(options.input_file):
    input_file = options.input_file
else:
    parser.usage()
    sys.exit(1)
if options.label_file and isfile(options.label_file):
    label_file = options.label_file
else:
    parser.usage()
    sys.exit(1)
if options.tensor_file:
    tensor_file = options.tensor_file
else:
    parser.usage()
    sys.exit(1)

# read training data
testing_pairs = []
log.info ("reading input file {0}".format(input_file))
dr = VectorPairReader(input_file)
for v1, v2 in dr:
    testing_pairs.append((v1, v2))
dimensions = len(testing_pairs[0][0])
testing_data = [(1, v1, v2) for v1, v2 in testing_pairs]

testing_labels = []
log.info ("reading label file {0}".format(label_file))
with open(label_file) as f:
    for line in f:
      testing_labels.append(line.strip().split('\t'))

tensor = []
log.info ("reading tensor file {0}".format(tensor_file))
with open(tensor_file) as f:
    for line in f:
        tensor.append(list(map(eval, line.rstrip().split(' '))))

log.info ("testing")
avg_dist = 0.0
for i, (sign, h, t) in enumerate(testing_data):
    x = np.dot(tensor,h)
    dist = cosine(t, x)
    print ("{0:.3f} {1} {2}".format(dist, testing_labels[i][0], testing_labels[i][1]))
    avg_dist += dist

avg_dist = avg_dist/len(testing_data)
#print ("mean cosine distance: {0}".format(avg_dist))

