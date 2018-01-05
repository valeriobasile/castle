import numpy as np

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
