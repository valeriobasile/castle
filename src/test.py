#!/usr/bin/env python
import logging as log
from optparse import OptionParser
import numpy as np
from vector import VectorPairReader
from scipy.spatial.distance import cosine

# command line argument partsing
parser = OptionParser()
parser.add_option('-i',
                  '--input',
                  dest="input_file",
                  help="read training vector pairs from FILE",
                  metavar="FILE")
parser.add_option('-t',
                  '--tensor',
                  dest="tensor_file",
                  default='tensor.txt',
                  help="read tensor from FILE",
                  metavar="FILE")
(options, args) = parser.parse_args()

if options.input_file:
    input_file = options.input_file
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

tensor = []
log.info ("reading tensor file {0}".format(tensor_file))
with open(tensor_file) as f:
    for line in f:
        tensor.append(list(map(eval, line.rstrip().split(' '))))

log.info ("testing")
avg_dist = 0.0
for sign, h, t in testing_data:
    x = np.dot(tensor,h)
    avg_dist += cosine(t, x)

print ("mean cosine distance: {0}".format(avg_dist/len(testing_data)))


