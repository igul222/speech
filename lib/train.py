import lib
import numpy
import theano
import theano.tensor as T
import lasagne
# from theano.compile.nanguardmode import NanGuardMode

import math
import time
import locale

locale.setlocale(locale.LC_ALL, '')

def print_params_info(cost, params):
    """Print information about the parameters in the given param set."""

    params = sorted(params, key=lambda p: p.name)
    shapes = [p.get_value(borrow=True).shape for p in params]
    print "Params:"
    for param, shape in zip(params, shapes):
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