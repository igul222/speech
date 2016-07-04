import lib
import numpy
import theano
import theano.tensor as T
import lasagne
# from theano.compile.nanguardmode import NanGuardMode

import math
import time
import locale

import numpy

locale.setlocale(locale.LC_ALL, '')

def print_params_info(cost, params):
    """Print information about the parameters in the given param set."""

    params = sorted(params, key=lambda p: p.name)
    values = [p.get_value(borrow=True) for p in params]
    shapes = [p.shape for p in values]
    print "Params for cost:"
    for param, value, shape in zip(params, values, shapes):
        print "\t{0} ({1})".format(
            param.name,
            ",".join([str(x) for x in shape])
        )

    total_param_count = 0
    for shape in shapes:
        param_count = 1
        for dim in shape:
            param_count *= dim
        total_param_count += param_count
    print "Total parameter count: {0}".format(
        locale.format("%d", total_param_count, grouping=True)
    )