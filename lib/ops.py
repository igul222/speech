import lib
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams(seed=234)

def Linear(
        name, 
        input_dims, 
        output_dim, 
        inputs,
        biases=True,
        initialization=None,
        weightnorm=True
        ):
    """
    Compute a linear transform of one or more inputs, optionally with a bias.

    input_dims: list of ints, or int (if single input); the dimensionality of
                the input(s).
    output_dim: the dimensionality of the output.
    biases:     whether or not to include a bias term.
    inputs:     a theano variable, or list of variables (if multiple inputs);
                the inputs to which to apply the transform.
    initialization: one of None, `lecun`, `he`, `orthogonal`
    """

    if not isinstance(input_dims, list):
        input_dims = [input_dims]
        inputs = [inputs]

    terms = []

    def uniform(stdev, size):
        """uniform distribution with the given stdev and size"""
        return numpy.random.uniform(
            low=-stdev * numpy.sqrt(3),
            high=stdev * numpy.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    for i, (inp, inp_dim) in enumerate(zip(inputs, input_dims)):
        if initialization == 'lecun' or (initialization == None and inp_dim != output_dim):
            weight_values = uniform(numpy.sqrt(1. / inp_dim), (inp_dim, output_dim))
        elif initialization == 'he':
            weight_values = uniform(numpy.sqrt(2. / inp_dim), (inp_dim, output_dim))
        elif initialization == 'orthogonal' or (initialization == None and inp_dim == output_dim):
            # From lasagne
            def sample(shape):
                if len(shape) < 2:
                    raise RuntimeError("Only shapes of length 2 or more are "
                                       "supported.")
                flat_shape = (shape[0], numpy.prod(shape[1:]))
                # TODO: why normal and not uniform?
                a = numpy.random.normal(0.0, 1.0, flat_shape)
                u, _, v = numpy.linalg.svd(a, full_matrices=False)
                # pick the one with the correct shape
                q = u if u.shape == flat_shape else v
                q = q.reshape(shape)
                return q.astype(theano.config.floatX)
            weight_values = sample((inp_dim, output_dim))
        else:
            raise Exception("Invalid initialization!")

        weight = lib.param(
            name + '.W'+str(i),
            weight_values
        )

        if weightnorm:
            norm_values = numpy.linalg.norm(weight_values, axis=0)
            norms = lib.param(
                name + '.g'+str(i),
                norm_values
            )

            normed_weight = weight * (norms / weight.norm(2, axis=0)).dimshuffle('x', 0)
            terms.append(T.dot(inp, normed_weight))
        else:        
            terms.append(T.dot(inp, weight))

    if biases:
        terms.append(lib.param(
            name + '.b',
            numpy.zeros((output_dim,), dtype=theano.config.floatX)
        ))

    out = reduce(lambda a,b: a+b, terms)
    out.name = name + '.output'
    return out


def Embedding(name, n_symbols, output_dim, indices):
    vectors = lib.param(
        name,
        numpy.random.randn(
            n_symbols, 
            output_dim
        ).astype(theano.config.floatX)
    )

    output_shape = [
        indices.shape[i]
        for i in xrange(indices.ndim)
    ] + [output_dim]

    return vectors[indices.flatten()].reshape(output_shape)

def softmax_and_sample(logits):
    old_shape = logits.shape
    flattened_logits = logits.reshape((-1, logits.shape[logits.ndim-1]))
    samples = T.cast(
        srng.multinomial(pvals=T.nnet.softmax(flattened_logits)),
        theano.config.floatX
    ).reshape(old_shape)
    return T.argmax(samples, axis=samples.ndim-1)

def Recurrent(name, hidden_dims, step_fn, inputs, non_sequences=[], h0s=None):
    if not isinstance(inputs, list):
        inputs = [inputs]

    if not isinstance(hidden_dims, list):
        hidden_dims = [hidden_dims]

    if h0s is None:
        h0s = [None]*len(hidden_dims)

    for i in xrange(len(hidden_dims)):
        if h0s[i] is None:
            h0_unbatched = lib.param(
                name + '.h0_' + str(i),
                numpy.zeros((hidden_dims[i],), dtype=theano.config.floatX)
            )
            num_batches = inputs[0].shape[1]
            h0s[i] = T.alloc(h0_unbatched, num_batches, hidden_dims[i])

        h0s[i] = T.patternbroadcast(h0s[i], [False] * h0s[i].ndim)

    outputs, _ = theano.scan(
        step_fn,
        sequences=inputs,
        outputs_info=h0s,
        non_sequences=non_sequences
    )

    return outputs

def GRUStep(name, input_dim, hidden_dim, current_input, last_hidden):
    processed_input = lib.ops.Linear(
        name+'.Input',
        input_dim,
        3 * hidden_dim,
        current_input
    )

    gates = T.nnet.sigmoid(
        lib.ops.Linear(
            name+'.Recurrent_Gates',
            hidden_dim,
            2 * hidden_dim,
            last_hidden,
            biases=False
        ) + processed_input[:, :2*hidden_dim]
    )

    update = gates[:, :hidden_dim]
    reset  = gates[:, hidden_dim:]

    scaled_hidden = reset * last_hidden

    candidate = T.tanh(
        lib.ops.Linear(
            name+'.Recurrent_Candidate', 
            hidden_dim, 
            hidden_dim, 
            scaled_hidden,
            biases=False,
            initialization='orthogonal'
        ) + processed_input[:, 2*hidden_dim:]
    )

    one = lib.floatX(1.0)
    return (update * candidate) + ((one - update) * last_hidden)

def LowMemGRU(name, input_dim, hidden_dim, inputs, h0=None):
    inputs = inputs.dimshuffle(1,0,2)

    def step(current_input, last_hidden):
        return GRUStep(
            name+'.Step', 
            input_dim, 
            hidden_dim, 
            current_input, 
            last_hidden
        )

    if h0 is None:
        h0s = None
    else:
        h0s = [h0]

    out = Recurrent(
        name+'.Recurrent',
        hidden_dim,
        step,
        inputs,
        h0s=h0s
    )

    out = out.dimshuffle(1,0,2)
    out.name = name+'.output'
    return out