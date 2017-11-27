import _init_paths
import caffe
import argparse
import pprint
import numpy as np
import sys

caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('/home/zf/caffe/multible_channel_identification/solver.prototxt')
solver.solve()


