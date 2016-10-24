import numpy
import time
import sys
import subprocess
import os
import random
import cPickle
import copy

import theano
from theano import tensor as T
from collections import OrderedDict, defaultdict
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import theano.tensor.shared_randomstreams

#########################SOME UTILITIES########################


def randomMatrix(r, c, scale=0.2):
    #W_bound = numpy.sqrt(6. / (r + c))
    W_bound = 1.
    return scale * numpy.random.uniform(low=-W_bound, high=W_bound,\
                   size=(r, c)).astype(theano.config.floatX)

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)

def _slice(_x, n, dim):
    return _x[:,n*dim:(n+1)*dim]

###############################################################

##########################Optimization function################

def adadelta(ips,cost,names,parameters,gradients,lr,norm_lim,rho=0.95,eps=1e-6):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in zip(names, parameters)]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2'%k) for k, p in zip(names, parameters)]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in zip(names, parameters)]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, gradients)]
    rg2up = [(rg2, rho * rg2 + (1. - rho) * (g ** 2)) for rg2, g in zip(running_grads2, gradients)] 
    f_grad_shared = theano.function(ips, cost, updates=zgup+rg2up, on_unused_input='ignore')

    updir = [-T.sqrt(ru2 + eps) / T.sqrt(rg2 + eps) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, rho * ru2 + (1. - rho) * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(parameters, updir)]
    
    if norm_lim > 0:
        param_up = clipGradient(param_up, norm_lim, names)

    f_param_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore')

    return f_grad_shared, f_param_update

def sgd(ips,cost,names,parameters,gradients,lr,norm_lim):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in zip(names, parameters)]
    gsup = [(gs, g) for gs, g in zip(gshared, gradients)]

    f_grad_shared = theano.function(ips, cost, updates=gsup, on_unused_input='ignore')

    pup = [(p, p - lr * g) for p, g in zip(parameters, gshared)]
    
    if norm_lim > 0:
        pup = clipGradient(pup, norm_lim, names)
    
    f_param_update = theano.function([lr], [], updates=pup, on_unused_input='ignore')

    return f_grad_shared, f_param_update

def clipGradient(updates, norm, names):
    id = -1
    res = []
    for p, g in updates:
        id += 1
        if not names[id].startswith('word') and 'multi' not in names[id] and p.get_value(borrow=True).ndim == 2:
            col_norms = T.sqrt(T.sum(T.sqr(g), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm))
            scale = desired_norms / (1e-7 + col_norms)
            g = g * scale
            
        res += [(p, g)]
    return res          

###############################################################

def _dropout_from_layer(rng, layers, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    res = []
    for layer in layers:
        mask = srng.binomial(n=1, p=1-p, size=layer.shape)
        # The cast is important because
        # int * float32 = float64 which pulls things off the gpu
        output = layer * T.cast(mask, theano.config.floatX)
        res += [output]
    return res

###############################Models###############################

def getOuter(var1, var2):
    def recurrence(var1_i, var2_i):
        def rec(var1_i_j, var2_i_j):
            varij = T.cast(T.outer(var1_i_j, var2_i_j).flatten(), dtype=theano.config.floatX)
            return varij
        rep, _ = theano.scan(fn=rec, sequences=[var1_i, var2_i], outputs_info=[None], n_steps=var1_i.shape[0])
        return rep

    out, _ = theano.scan(fn=recurrence, sequences=[var1, var2], outputs_info=[None], n_steps=var1.shape[0])
    
    return out
    
def getConcatenation(embDict, vars, features, features_dim, tranpose=False, outer=False):
    repMod = '_getConcatenation'
    if outer: repMod += 'Outer'
    return eval(repMod)(embDict, vars, features, features_dim, tranpose)

def _getConcatenation(embDict, vars, features, features_dim, tranpose=False):
    xs = []

    for ed in features:
        if features[ed] == 0:
            var = vars[ed] if not tranpose else vars[ed].T
            xs += [embDict[ed][T.cast(var.flatten(), dtype='int32')].reshape((var.shape[0], var.shape[1], features_dim[ed]))]
        elif features[ed] == 1:
            if not tranpose:
                xs += [vars[ed]]
            else:
                xs += [vars[ed].dimshuffle(1,0,2)]

    if len(xs) == 1:
        basex = xs[0]
    else:
        basex = T.cast(T.concatenate(xs, axis=2), dtype=theano.config.floatX)

    return basex
    
def _getConcatenationOuter(embDict, vars, features, features_dim, tranpose=False):
    
    var = vars['word'] if not tranpose else vars['word'].T
    wb = embDict['word'][T.cast(var.flatten(), dtype='int32')].reshape((var.shape[0], var.shape[1], features_dim['word']))

    xs = []
    for ed in features:
        if ed == 'word': continue
        if features[ed] == 0:
            var = vars[ed] if not tranpose else vars[ed].T
            xs += [embDict[ed][T.cast(var.flatten(), dtype='int32')].reshape((var.shape[0], var.shape[1], features_dim[ed]))]
        elif features[ed] == 1:
            if not tranpose:
                xs += [vars[ed]]
            else:
                xs += [vars[ed].dimshuffle(1,0,2)]

    if len(xs) == 1:
        fb = xs[0]
    else:
        fb = T.cast(T.concatenate(xs, axis=2), dtype=theano.config.floatX)

    return getOuter(wb, fb)
    
def getInverseConcatenation(embDict, vars, features, features_dim, outer=False):
    repMod = '_getInverseConcatenation'
    if outer: repMod += 'Outer'
    return eval(repMod)(embDict, vars, features, features_dim)

def _getInverseConcatenation(embDict, vars, features, features_dim):
        
    ixs = []

    for ed in features:
        if features[ed] == 0:
            var = vars[ed].T[::-1]
            ixs += [embDict[ed][T.cast(var.flatten(), dtype='int32')].reshape((var.shape[0], var.shape[1], features_dim[ed]))]
        elif features[ed] == 1:
            ixs += [vars[ed].dimshuffle(1,0,2)[::-1]]                

    if len(ixs) == 1:
        ibasex = ixs[0]
    else:
        ibasex = T.cast(T.concatenate(ixs, axis=2), dtype=theano.config.floatX)
    
    return ibasex
    
def _getInverseConcatenationOuter(embDict, vars, features, features_dim):

    var = vars['word'].T[::-1]
    iwb += [embDict['word'][T.cast(var.flatten(), dtype='int32')].reshape((var.shape[0], var.shape[1], features_dim['word']))]
        
    ixs = []

    for ed in features:
        if ed == 'word': continue
        if features[ed] == 0:
            var = vars[ed].T[::-1]
            ixs += [embDict[ed][T.cast(var.flatten(), dtype='int32')].reshape((var.shape[0], var.shape[1], features_dim[ed]))]
        elif features[ed] == 1:
            ixs += [vars[ed].dimshuffle(1,0,2)[::-1]]                

    if len(ixs) == 1:
        ifb = ixs[0]
    else:
        ifb = T.cast(T.concatenate(ixs, axis=2), dtype=theano.config.floatX)

    return getOuter(iwb, ifb)

def basicRep(embDict, vars, features, features_dim, encoding):
    rep = getConcatenation(embDict, vars, features, features_dim, encoding)
    return rep

def rnn_ff(inps, dim, hidden, batSize, prefix, params, names):
    Wx  = theano.shared(randomMatrix(dim, hidden))
    Wh  = theano.shared(randomMatrix(hidden, hidden))
    bh  = theano.shared(numpy.zeros(hidden, dtype=theano.config.floatX))
    #model.container['bi_h0']  = theano.shared(numpy.zeros(model.container['nh'], dtype=theano.config.floatX))

    # bundle
    params += [ Wx, Wh, bh ] #, model.container['bi_h0']
    names += [ prefix + '_Wx', prefix + '_Wh', prefix + '_bh' ] #, 'bi_h0'

    def recurrence(x_t, h_tm1):
        h_t = T.nnet.sigmoid(T.dot(x_t, Wx) + T.dot(h_tm1, Wh) + bh)
        return h_t

    h, _  = theano.scan(fn=recurrence, \
            sequences=inps, outputs_info=[T.alloc(0., batSize, hidden)], n_steps=inps.shape[0])
    
    return h
    
def rnn_gru(inps, dim, hidden, batSize, prefix, params, names):
    Wc = theano.shared(numpy.concatenate([randomMatrix(dim, hidden), randomMatrix(dim, hidden)], axis=1))

    bc = theano.shared(numpy.zeros(2 * hidden, dtype=theano.config.floatX))

    U = theano.shared(numpy.concatenate([ortho_weight(hidden), ortho_weight(hidden)], axis=1))
    Wx = theano.shared(randomMatrix(dim, hidden))

    Ux = theano.shared(ortho_weight(hidden))

    bx = theano.shared(numpy.zeros(hidden, dtype=theano.config.floatX))

    #model.container['bi_h0'] = theano.shared(numpy.zeros(model.container['nh'], dtype=theano.config.floatX))

    # bundle
    params += [ Wc, bc, U, Wx, Ux, bx ] #, model.container['bi_h0']
    names += [ prefix + '_Wc', prefix + '_bc', prefix + '_U', prefix + '_Wx', prefix + '_Ux', prefix + '_bx' ] #, 'bi_h0'
    
    def recurrence(x_t, h_tm1):
        preact = T.dot(h_tm1, U)
        preact += T.dot(x_t, Wc) + bc

        r_t = T.nnet.sigmoid(_slice(preact, 0, hidden))
        u_t = T.nnet.sigmoid(_slice(preact, 1, hidden))

        preactx = T.dot(h_tm1, Ux)
        preactx = preactx * r_t
        preactx = preactx + T.dot(x_t, Wx) + bx

        h_t = T.tanh(preactx)

        h_t = u_t * h_tm1 + (1. - u_t) * h_t

        return h_t

    h, _  = theano.scan(fn=recurrence, \
            sequences=inps, outputs_info=[T.alloc(0., batSize, hidden)], n_steps=inps.shape[0])
    
    return h
    
def ffBidirectCore(inps, iinps, dim, hidden, batSize, prefix, iprefix, params, names):

    bi_h = rnn_ff(inps, dim, hidden, batSize, prefix, params, names)
    
    ibi_h = rnn_ff(iinps, dim, hidden, batSize, iprefix, params, names)

    _ibi_h = ibi_h[::-1]
    
    bi_rep = T.cast(T.concatenate([ bi_h, _ibi_h ], axis=2).dimshuffle(1,0,2), dtype=theano.config.floatX)

    return bi_rep
    
def gruBidirectCore(inps, iinps, dim, hidden, batSize, prefix, iprefix, params, names):

    bi_h = rnn_gru(inps, dim, hidden, batSize, prefix, params, names)
    
    ibi_h = rnn_gru(iinps, dim, hidden, batSize, iprefix, params, names)

    _ibi_h = ibi_h[::-1]

    bi_rep = T.cast(T.concatenate([ bi_h, _ibi_h ], axis=2).dimshuffle(1,0,2), dtype=theano.config.floatX)

    return bi_rep
    
def gruBidirectOuterCore(inps, iinps, dim, hidden, batSize, prefix, iprefix, params, names):

    bi_h = rnn_gru(inps, dim, hidden, batSize, prefix, params, names)

    ibi_h = rnn_gru(iinps, dim, hidden, batSize, iprefix, params, names)

    _ibi_h = ibi_h[::-1]

    bi_rep = getOuter(bi_h, _ibi_h)

    bi_rep = T.cast(bi_rep.dimshuffle(1,0,2), dtype=theano.config.floatX)

    return bi_rep

def ffForward(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names, outer=False):
    ix = getConcatenation(embDict, vars, features, features_dim, tranpose=True, outer=outer)
    
    i_h = rnn_ff(ix, dimIn, hidden, batch, prefix, params, names)
    
    rep = T.cast(i_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def ffBackward(embDict, vars, features, features_dim, dimIn, hidden, batch, iprefix, params, names, outer=False):
    iix = getInverseConcatenation(embDict, vars, features, features_dim, outer=outer)
    
    ii_h = rnn_ff(iix, dimIn, hidden, batch, iprefix, params, names)
    
    _ii_h = ii_h[::-1]
    
    rep = T.cast(_ii_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def ffBiDirect(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names, outer=False):
    bix = getConcatenation(embDict, vars, features, features_dim, tranpose=True, outer=outer)
    ibix = getInverseConcatenation(embDict, vars, features, features_dim, outer=outer)
    
    return ffBidirectCore(bix, ibix, dimIn, hidden, batch, prefix + '_ffbi', prefix + '_ffibi', params, names)
    
def gruForward(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names, outer=False):
    ix = getConcatenation(embDict, vars, features, features_dim, tranpose=True, outer=outer)
    
    i_h = rnn_gru(ix, dimIn, hidden, batch, prefix, params, names)
    
    rep = T.cast(i_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def gruBackward(embDict, vars, features, features_dim, dimIn, hidden, batch, iprefix, params, names, outer=False):
    iix = getInverseConcatenation(embDict, vars, features, features_dim, outer=outer)
    
    ii_h = rnn_gru(iix, dimIn, hidden, batch, iprefix, params, names)
    
    _ii_h = ii_h[::-1]
    
    rep = T.cast(_ii_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def gruBiDirect(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names, outer=False):
    bix = getConcatenation(embDict, vars, features, features_dim, tranpose=True, outer=outer)
    ibix = getInverseConcatenation(embDict, vars, features, features_dim, outer=outer)
    
    return gruBidirectCore(bix, ibix, dimIn, hidden, batch, prefix + '_grubi', prefix + '_gruibi', params, names)
    
def gruBiDirectOuter(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names, outer=False):
    bix = getConcatenation(embDict, vars, features, features_dim, tranpose=True, outer=outer)
    ibix = getInverseConcatenation(embDict, vars, features, features_dim, outer=outer)

    return gruBidirectOuterCore(bix, ibix, dimIn, hidden, batch, prefix + '_gruOuterbi', prefix + '_gruOuteribi', params, names)

def ffOnePass(model):
    bix = getConcatenation(model)

    bi_h = rnn_ff(bix, model.container['bi_dimIn'], model.container['nh'], model.container['batch'], 'ffOne', model.container['params'], model.container['names'])
    
    bi_h = T.cast(bi_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    model.container['dimIn'] = model.container['nh']
    
    return bi_h

def gruOnePass(model):
    bix = getConcatenation(model)
    
    bi_h = rnn_gru(bix, model.container['bi_dimIn'], model.container['nh'], model.container['batch'], 'gruOne', model.container['params'], model.container['names'])
    
    bi_h = T.cast(bi_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    model.container['dimIn'] = model.container['nh']
    
    return bi_h
    
###############################CONVOLUTIONAL CONTEXT####################################

def convolutionalLayer(inpu, feature_map, batch, length, window, dim, prefix, params, names):
    down = window / 2
    up = window - down - 1
    zodown = T.zeros((batch, 1, down, dim), dtype=theano.config.floatX)
    zoup = T.zeros((batch, 1, up, dim), dtype=theano.config.floatX)
    
    inps = T.cast(T.concatenate([zoup, inpu, zodown], axis=2), dtype=theano.config.floatX)
    
    fan_in = window * dim
    fan_out = feature_map * window * dim / length #(length - window + 1)

    filter_shape = (feature_map, 1, window, dim)
    image_shape = (batch, 1, length + down + up, dim)

    #if non_linear=="none" or non_linear=="relu":
    #    conv_W = theano.shared(0.2 * numpy.random.uniform(low=-1.0,high=1.0,\
    #                            size=filter_shape).astype(theano.config.floatX))
        
    #else:
    #    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    #    conv_W = theano.shared(numpy.random.uniform(low=-W_bound,high=W_bound,\
    #                            size=filter_shape).astype(theano.config.floatX))

    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    conv_W = theano.shared(numpy.random.uniform(low=-W_bound,high=W_bound,\
                            size=filter_shape).astype(theano.config.floatX))

    conv_b = theano.shared(numpy.zeros(filter_shape[0], dtype=theano.config.floatX))

    # bundle
    params += [ conv_W, conv_b ]
    names += [ prefix + '_convL_W_' + str(window), prefix + '_convL_b_' + str(window) ]

    conv_out = conv.conv2d(input=inps, filters=conv_W, filter_shape=filter_shape, image_shape=image_shape)

    conv_out = T.tanh(conv_out + conv_b.dimshuffle('x', 0, 'x', 'x'))

    return conv_out.dimshuffle(0,2,1,3).flatten(3)
    
def convContextLs(inps, feature_map, convWins, batch, length, dim, prefix, params, names):
    cx = T.cast(inps.reshape((inps.shape[0], 1, inps.shape[1], inps.shape[2])), dtype=theano.config.floatX)

    fts = []
    for i, convWin in enumerate(convWins):
        fti = convolutionalLayer(cx, feature_map, batch, length, convWin, dim, prefix + '_winL' + str(i), params, names)
        fts += [fti]

    convRep = T.cast(T.concatenate(fts, axis=2), dtype=theano.config.floatX)

    return convRep
    
def LeNetConvPoolLayer(inps, feature_map, batch, length, window, dim, prefix, params, names):
    fan_in = window * dim
    fan_out = feature_map * window * dim / (length - window + 1)

    filter_shape = (feature_map, 1, window, dim)
    image_shape = (batch, 1, length, dim)
    pool_size = (length - window + 1, 1)

    #if non_linear=="none" or non_linear=="relu":
    #    conv_W = theano.shared(0.2 * numpy.random.uniform(low=-1.0,high=1.0,\
    #                            size=filter_shape).astype(theano.config.floatX))
        
    #else:
    #    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    #    conv_W = theano.shared(numpy.random.uniform(low=-W_bound,high=W_bound,\
    #                            size=filter_shape).astype(theano.config.floatX))

    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    conv_W = theano.shared(numpy.random.uniform(low=-W_bound,high=W_bound,\
                            size=filter_shape).astype(theano.config.floatX))

    conv_b = theano.shared(numpy.zeros(filter_shape[0], dtype=theano.config.floatX))

    # bundle
    params += [ conv_W, conv_b ]
    names += [ prefix + '_conv_W_' + str(window), prefix + '_conv_b_' + str(window) ]

    conv_out = conv.conv2d(input=inps, filters=conv_W, filter_shape=filter_shape, image_shape=image_shape)

        
    conv_out_act = T.tanh(conv_out + conv_b.dimshuffle('x', 0, 'x', 'x'))
    conv_output = downsample.max_pool_2d(input=conv_out_act, ds=pool_size, ignore_border=True)

    return conv_output.flatten(2)

def convContext(inps, feature_map, convWins, batch, length, dim, prefix, params, names):

    cx = T.cast(inps.reshape((inps.shape[0], 1, inps.shape[1], inps.shape[2])), dtype=theano.config.floatX)

    fts = []
    for i, convWin in enumerate(convWins):
        fti = LeNetConvPoolLayer(cx, feature_map, batch, length, convWin, dim, prefix + '_win' + str(i), params, names)
        fts += [fti]

    convRep = T.cast(T.concatenate(fts, axis=1), dtype=theano.config.floatX)

    return convRep
    
#############################Multilayer NNs################################

def HiddenLayer(inputs, nin, nout, params, names, prefix):
    W_bound = numpy.sqrt(6. / (nin + nout))
    multi_W = theano.shared(numpy.random.uniform(low=-W_bound,high=W_bound,\
                            size=(nin, nout)).astype(theano.config.floatX))

    multi_b = theano.shared(numpy.zeros(nout, dtype=theano.config.floatX))
    res = []
    for input in inputs:
        out = T.nnet.sigmoid(T.dot(input, multi_W) + multi_b)
        res += [out]
    
    params += [multi_W, multi_b]
    names += [prefix + '_multi_W', prefix + '_multi_b']
    
    return res

def MultiHiddenLayers(inputs, hids, params, names, prefix):
    
    hiddenVector = inputs
    id = 0
    for nin, nout in zip(hids, hids[1:]):
        id += 1
        hiddenVector = HiddenLayer(hiddenVector, nin, nout, params, names, prefix + '_layer' + str(id))
    return hiddenVector

#########################################################################################

class BaseModel(object):

    def __init__(self, args):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        #de :: dimension of the word embeddings
        cs :: word window context size
        '''
        self.container = {}
        
        self.args = args
        self.args['rng'] = numpy.random.RandomState(3435)
        self.args['dropout'] = args['dropout'] if args['dropout'] > 0. else 0.
        
        # parameters of the model
        
        self.container['params'], self.container['names'] = [], []
        
        self.container['embDict1'] = OrderedDict()
        self.container['vars1'] = OrderedDict()
        self.container['dimIn1'] = 0
        
        ##-1
        self.container['embDict-1'] = OrderedDict()
        self.container['vars-1'] = OrderedDict()
        self.container['dimIn-1'] = 0
        
        self.args['nh-1'] = self.args['nh1']
        self.args['features-1'] = self.args['features1']
        self.args['features_dim-1'] = self.args['features_dim1']
        self.args['conv_feature_map-1'] = self.args['conv_feature_map1']
        self.args['conv_win_feature_map-1'] = self.args['conv_win_feature_map1']
        self.args['conv_winre-1'] = self.args['conv_winre1']
        ##1
        
        self.container['embDict2'] = OrderedDict()
        self.container['vars2'] = OrderedDict()
        self.container['dimIn2'] = 0
        print '******************FEATURES 1******************'
        for ed in self.args['features1']:
            if self.args['features1'][ed] == 0:
                self.container['embDict1'][ed] = theano.shared(self.args['embs1'][ed].astype(theano.config.floatX))
                
                ##-1
                if self.args['sharedEmbs'][ed] == 1:
                    self.container['embDict-1'][ed] = self.container['embDict1'][ed]
                else:
                    self.container['embDict-1'][ed] = theano.shared(self.args['embs2'][ed].astype(theano.config.floatX))
                ##
                
                if self.args['updateEmbs']:
                    print '@@@@@@@ Will update embedding tables'
                    self.container['params'] += [self.container['embDict1'][ed]]
                    self.container['names'] += [ed + '_1']
                    
                    ##-1
                    if self.args['sharedEmbs'][ed] != 1 and '_' in self.args['model']:
                        self.container['params'] += [self.container['embDict-1'][ed]]
                        self.container['names'] += [ed + '_-1']
                    ##

            if self.args['features1'][ed] == 0:
                self.container['vars1'][ed] = T.imatrix()
                dimAdding = self.args['embs1'][ed].shape[1]
                self.container['dimIn1'] += dimAdding
                
                ##-1
                self.container['vars-1'][ed] = self.container['vars1'][ed]
                self.container['dimIn-1'] += dimAdding
                ##             
            elif self.args['features1'][ed] == 1:
                self.container['vars1'][ed] = T.tensor3()
                dimAdding = self.args['features_dim1'][ed]
                self.container['dimIn1'] += dimAdding
                
                ##-1
                self.container['vars-1'][ed] = self.container['vars1'][ed]
                self.container['dimIn-1'] += dimAdding
                ##

            if self.args['features1'][ed] >= 0:
                print 'represetation 1 - ', ed, ' : ', dimAdding 
                                
        if self.args['outer']:
            print '------- Using outer product to obtain representation'
            self.container['dimIn1'] = (self.container['dimIn1'] - self.args['embs1']['word'].shape[1]) * self.args['embs1']['word'].shape[1]
        print 'REPRESENTATION DIMENSION 1 = ', self.container['dimIn1']
        
        ##-1
        if self.args['outer']:
            self.container['dimIn-1'] = (self.container['dimIn-1'] - self.args['embs1']['word'].shape[1]) * self.args['embs1']['word'].shape[1]
        print 'REPRESENTATION DIMENSION -1 = ', self.container['dimIn-1']
        ##
        
        print '******************FEATURES 2******************'
        for ed in self.args['features2']:
            if self.args['features2'][ed] == 0:
                if self.args['sharedEmbs'][ed] == 1 and ed in self.container['embDict1']:
                    self.container['embDict2'][ed] = self.container['embDict1'][ed]
                else:
                    self.container['embDict2'][ed] = theano.shared(self.args['embs2'][ed].astype(theano.config.floatX))
                
                    if self.args['updateEmbs']:
                        print '@@@@@@@ Will update embedding tables'
                        self.container['params'] += [self.container['embDict2'][ed]]
                        self.container['names'] += [ed + '_2']

            if self.args['features2'][ed] == 0:
                self.container['vars2'][ed] = T.imatrix()
                dimAdding = self.args['embs2'][ed].shape[1]
                self.container['dimIn2'] += dimAdding
            elif self.args['features2'][ed] == 1:
                self.container['vars2'][ed] = T.tensor3()
                dimAdding = self.args['features_dim2'][ed]
                self.container['dimIn2'] += dimAdding

            if self.args['features2'][ed] >= 0:
                print 'represetation 2 - ', ed, ' : ', dimAdding 
                                
        if self.args['outer']:
            print '------- Using outer product to obtain representation'
            self.container['dimIn2'] = (self.container['dimIn2'] - self.args['embs2']['word'].shape[1]) * self.args['embs2']['word'].shape[1]
        if self.container['dimIn2'] > 0: print 'REPRESENTATION DIMENSION 2 = ', self.container['dimIn2']
        print '*******************************************'
        
        #if self.container['encoding'] != 'basicRep':
        #    self.container['bi_dimIn'] = self.container['dimIn']
        #    self.container['dimIn'] = 0

        self.container['y'] = T.ivector('y') # label
        self.container['lr'] = T.scalar('lr')
        self.container['pos11'] = T.ivector('entityPosition11')
        self.container['pos21'] = T.ivector('entityPosition21')
        self.container['pos12'] = T.ivector('entityPosition12')
        self.container['pos22'] = T.ivector('entityPosition22')
        self.container['binaryFeatures'] = T.imatrix('binaryFeatures')
        self.container['kernelScore'] = T.matrix('kernelScore')
        self.container['iidep1'] = T.matrix('iidep1')
        self.container['iidep2'] = T.matrix('iidep2')
        self.container['zeroVector'] = T.vector('zeroVector')
        
        ##-1
        self.container['pos1-1'] = self.container['pos11']
        self.container['pos2-1'] = self.container['pos21']
        self.container['iidep-1'] = self.container['iidep1']
        ##

    #def generateInput(self):
    
    #    basex = eval(self.container['encoding'])(self)
        
    #    self.container['x'] = basex

    def buildFunctions(self, p_y_given_x, p_y_given_x_dropout):
    
        if self.args['dropout'] == 0.:        
            nll = -T.mean(T.log(p_y_given_x)[T.arange(self.container['y'].shape[0]), self.container['y']])
        else:
            nll = -T.mean(T.log(p_y_given_x_dropout)[T.arange(self.container['y'].shape[0]), self.container['y']])
        
        if self.args['regularizer'] > 0.:
            for pp, nn in zip(self.container['params'], self.container['names']):
                if 'multi' in nn:
                    nll += self.args['regularizer'] * (pp ** 2).sum()
        
        y_pred = T.argmax(p_y_given_x, axis=1)
        
        gradients = T.grad( nll, self.container['params'] )

        classifyInput = [ self.container['vars1'][ed] for ed in self.args['features1'] if self.args['features1'][ed] >= 0 ]
        classifyInput += [ self.container['pos11'], self.container['pos21'] ] 
        
        if len(self.args['model'].split('-')) == 2:
            classifyInput += [ self.container['vars2'][ed] for ed in self.args['features2'] if self.args['features2'][ed] >= 0 ]
            classifyInput += [ self.container['pos12'], self.container['pos22'] ]
        
        if self.args['binaryCutoff'] >= 0:
            classifyInput += [ self.container['binaryFeatures'] ]
            
        if self.args['kernelFets']['kernelScore'] > 0:
            print '+++ Using kernelScore'
            classifyInput += [ self.container['kernelScore'] ]
        
        for prei, mmodel in enumerate(self.args['model'].split('-')):
            classifyInput += [ self.container['iidep' + str(prei+1)] ]
        
        # theano functions
        self.classify = theano.function(inputs=classifyInput, outputs=[y_pred, p_y_given_x], on_unused_input='ignore')

        trainInput = classifyInput + [self.container['y']]

        self.f_grad_shared, self.f_update_param = eval(self.args['optimizer'])(trainInput,nll,self.container['names'],self.container['params'],gradients,self.container['lr'],self.args['norm_lim'])
        
        self.container['setZero'] = OrderedDict()
        self.container['zeroVecs'] = OrderedDict()
        for ed in self.container['embDict1']:
            self.container['zeroVecs'][ed + '1'] = numpy.zeros(self.args['embs1'][ed].shape[1],dtype='float32')
            self.container['setZero'][ed + '1'] = theano.function([self.container['zeroVector']], updates=[(self.container['embDict1'][ed], T.set_subtensor(self.container['embDict1'][ed][0,:], self.container['zeroVector']))])
        
        for ed in self.container['embDict2']:
            self.container['zeroVecs'][ed + '2'] = numpy.zeros(self.args['embs2'][ed].shape[1],dtype='float32')
            self.container['setZero'][ed + '2'] = theano.function([self.container['zeroVector']], updates=[(self.container['embDict2'][ed], T.set_subtensor(self.container['embDict2'][ed][0,:], self.container['zeroVector']))])

    def save(self, folder):   
        for param, name in zip(self.container['params'], self.container['names']):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

class mainModel(BaseModel):
    def __init__(self, args):

        BaseModel.__init__(self, args)
        
        models = self.args['model'].split('-')
        fetre = []
        dim_inter = 0
        
        for i, model in enumerate(models):
            r, d = eval(model)(self, i+1)
            fetre += [r]
            dim_inter += d        
        
        fetre_dropout = _dropout_from_layer(self.args['rng'], fetre, self.args['dropout'])
        
        if len(fetre) == 1:
            fetre = fetre[0]
            fetre_dropout = fetre_dropout[0]
        else:
            fetre = T.cast(T.concatenate(fetre, axis=1), dtype=theano.config.floatX)
            fetre_dropout = T.cast(T.concatenate(fetre_dropout, axis=1), dtype=theano.config.floatX)
            
        hids = [dim_inter] + self.args['multilayerNN1']
        
        mul = MultiHiddenLayers([fetre, fetre_dropout], hids, self.container['params'], self.container['names'], 'multiMainModel')
        
        fetre, fetre_dropout = mul[0], mul[1]
        
        dim_inter = hids[len(hids)-1]
        
        fW = theano.shared(randomMatrix(dim_inter, self.args['nc']))
        fb = theano.shared(numpy.zeros(self.args['nc'], dtype=theano.config.floatX))
        
        self.container['params'] += [fW, fb]
        self.container['names'] += ['convolution_W', 'convolution_b']
        
        p_y_given_x_dropout = T.nnet.softmax(T.dot(fetre_dropout, fW) + fb)
        
        p_y_given_x = T.nnet.softmax(T.dot(fetre , (1.0 - self.args['dropout']) * fW) + fb)
        
        self.buildFunctions(p_y_given_x, p_y_given_x_dropout)

class MultiNN(BaseModel):
    
    def __init__(self, args):

        BaseModel.__init__(self, args)
        
        hids = [self.args['binaryFeatureDim']] + self.args['multilayerNN1'] + [self.args['nc']]
        
        layer0_multi_W = theano.shared(randomMatrix(self.args['binaryFeatureDim'], hids[1]))
        layer0_multi_b = theano.shared(numpy.zeros(hids[1], dtype=theano.config.floatX))
        
        self.container['params'] += [layer0_multi_W, layer0_multi_b]
        self.container['names'] += ['layer0_multi_W', 'layer0_multi_b']
        
        def recurrence(bfi, Wmat, bvec):
            idx = bfi[1:(bfi[0]+1)]
            weights = T.sum(Wmat[idx], axis=0) + bvec
            return weights
        
        firstMapped, _ = theano.scan(fn=recurrence, sequences=self.container['binaryFeatures'], outputs_info=[None], non_sequences=[layer0_multi_W, layer0_multi_b], n_steps=self.container['binaryFeatures'].shape[0])
        
        if self.args['useHeadEmbedding']:
            wordEmb_dim = self.args['embs1']['word'].shape[1]
            layer0_multi_headW = theano.shared(randomMatrix(2 * wordEmb_dim, hids[1]))
            layer0_multi_headb = theano.shared(numpy.zeros(hids[1], dtype=theano.config.floatX))
            self.container['params'] += [layer0_multi_headW, layer0_multi_headb]
            self.container['names'] += ['layer0_multi_headW', 'layer0_multi_headb']
            
            def recurrenceHead(sent, pos1, pos2, embRecur, Whmat, bhvec):
                eb1 = embRecur[sent[pos1]]
                eb2 = embRecur[sent[pos2]]
                ebcon = T.cast(T.concatenate([eb1, eb2]), dtype=theano.config.floatX)
                return T.dot(ebcon, Whmat) + bhvec
            
            headEmbs, _ = theano.scan(fn=recurrenceHead, sequences=[self.container['vars1']['word'], self.container['pos11'], self.container['pos21']], outputs_info=[None], non_sequences=[self.container['embDict1']['word'], layer0_multi_headW, layer0_multi_headb])
            firstMapped = firstMapped + headEmbs
        
        if len(hids) == 2:
            fetre = firstMapped
            fetre_dropout = firstMapped
        else:
            firstMapped = T.nnet.sigmoid(firstMapped)
            hids = hids[1:(len(hids)-1)]
            fetreArr = MultiHiddenLayers([firstMapped], hids, self.container['params'], self.container['names'], 'multiModel')
            fetre = fetreArr[0]
            dim_inter = hids[len(hids)-1]
        
            fW = theano.shared(randomMatrix(dim_inter, self.args['nc']))
            fb = theano.shared(numpy.zeros(self.args['nc'], dtype=theano.config.floatX))
        
            self.container['params'] += [fW, fb]
            self.container['names'] += ['W', 'b']
        
            fetre = T.dot(fetre , fW) + fb
            fetre_dropout = fetre
        
        
        p_y_given_x_dropout = T.nnet.softmax(fetre_dropout)
        
        p_y_given_x = T.nnet.softmax(fetre)
        
        self.buildFunctions(p_y_given_x, p_y_given_x_dropout)
        
class rnnHeadMultiExpNN(BaseModel):

    def __init__(self, args):

        BaseModel.__init__(self, args)
        
        fRnn, dim_rnn = rnnHead(self, 1)
        
        rnn_fW = theano.shared(randomMatrix(dim_rnn, self.args['nc']))
        rnn_fb = theano.shared(numpy.zeros(self.args['nc'], dtype=theano.config.floatX))
        
        self.container['params'] += [rnn_fW, rnn_fb]
        self.container['names'] += ['rnn_fW', 'rnn_fb']
        
        fRnn_dropout = _dropout_from_layer(self.args['rng'], [fRnn], self.args['dropout'])
        fRnn_dropout = fRnn_dropout[0]
        
        fRnn_dropout = T.exp(T.dot(fRnn_dropout, rnn_fW) + rnn_fb)
        fRnn = T.exp(T.dot(fRnn , (1.0 - self.args['dropout']) * rnn_fW) + rnn_fb)
        
        #-----convolute
        
        fConv, dim_conv = convolute(self, 2)
        
        convC_fW = theano.shared(randomMatrix(dim_conv, self.args['nc']))
        convC_fb = theano.shared(numpy.zeros(self.args['nc'], dtype=theano.config.floatX))
        
        self.container['params'] += [convC_fW, convC_fb]
        self.container['names'] += ['convC_fW', 'convC_fb']
        
        fConv_dropout = _dropout_from_layer(self.args['rng'], [fConv], self.args['dropout'])
        fConv_dropout = fConv_dropout[0]
        
        fConv_dropout = T.exp(T.dot(fConv_dropout, convC_fW) + convC_fb)
        fConv = T.exp(T.dot(fConv , (1.0 - self.args['dropout']) * convC_fW) + convC_fb)
        
        #-----multilayer nn
        
        hids = [self.args['binaryFeatureDim']] + self.args['multilayerNN1'] + [self.args['nc']]
        
        layer0_multi_W = theano.shared(randomMatrix(self.args['binaryFeatureDim'], hids[1]))
        layer0_multi_b = theano.shared(numpy.zeros(hids[1], dtype=theano.config.floatX))
        
        self.container['params'] += [layer0_multi_W, layer0_multi_b]
        self.container['names'] += ['layer0_multi_fW', 'layer0_multi_fb']
        
        def recurrence(bfi, Wmat, bvec):
            idx = bfi[1:(bfi[0]+1)]
            weights = T.sum(Wmat[idx], axis=0) + bvec
            return weights
        
        firstMapped, _ = theano.scan(fn=recurrence, sequences=self.container['binaryFeatures'], outputs_info=[None], non_sequences=[layer0_multi_W, layer0_multi_b], n_steps=self.container['binaryFeatures'].shape[0])
        
        if self.args['useHeadEmbedding']:
            wordEmb_dim = self.args['embs1']['word'].shape[1]
            layer0_multi_headW = theano.shared(randomMatrix(2 * wordEmb_dim, hids[1]))
            layer0_multi_headb = theano.shared(numpy.zeros(hids[1], dtype=theano.config.floatX))
            self.container['params'] += [layer0_multi_headW, layer0_multi_headb]
            self.container['names'] += ['layer0_multi_headW', 'layer0_multi_headb']
            
            def recurrenceHead(sent, pos1, pos2, embRecur, Whmat, bhvec):
                eb1 = embRecur[sent[pos1]]
                eb2 = embRecur[sent[pos2]]
                ebcon = T.cast(T.concatenate([eb1, eb2]), dtype=theano.config.floatX)
                return T.dot(ebcon, Whmat) + bhvec
            
            headEmbs, _ = theano.scan(fn=recurrenceHead, sequences=[self.container['vars1']['word'], self.container['pos11'], self.container['pos21']], outputs_info=[None], non_sequences=[self.container['embDict1']['word'], layer0_multi_headW, layer0_multi_headb])
            firstMapped = firstMapped + headEmbs
        
        if len(hids) == 2:
            fMulti = firstMapped
            fMulti_dropout = firstMapped
        else:
            firstMapped = T.nnet.sigmoid(firstMapped)
            hids = hids[1:(len(hids)-1)]
            fetreArr = MultiHiddenLayers([firstMapped], hids, self.container['params'], self.container['names'], 'rnnMultiNN')
            fMulti = fetreArr[0]
            dim_multi = hids[len(hids)-1]
        
            fW = theano.shared(randomMatrix(dim_multi, self.args['nc']))
            fb = theano.shared(numpy.zeros(self.args['nc'], dtype=theano.config.floatX))
        
            self.container['params'] += [fW, fb]
            self.container['names'] += ['fW', 'fb']
        
            fMulti = T.dot(fMulti , fW) + fb
            fMulti_dropout = fMulti
            
        fMulti = T.exp(fMulti)
        fMulti_dropout = fMulti
        
        fetre = fRnn * fConv * fMulti
        fetre_dropout = fRnn_dropout * fConv_dropout * fMulti_dropout       
        
        su = T.cast(fetre.sum(1).dimshuffle(0,'x'), dtype=theano.config.floatX)
        su_dropout = T.cast(fetre_dropout.sum(1).dimshuffle(0,'x'), dtype=theano.config.floatX)
        
        p_y_given_x_dropout = fetre_dropout / su_dropout

        p_y_given_x = fetre / su
        
        self.buildFunctions(p_y_given_x, p_y_given_x_dropout)
        
class ensembleModel(BaseModel):

    def __init__(self, args):

        BaseModel.__init__(self, args)
        
        models = self.args['model'].split('_')
        
        #------first model
        
        f1, dim_first = eval(models[0])(self, 1)
        
        first_fW = theano.shared(randomMatrix(dim_first, self.args['nc']))
        first_fb = theano.shared(numpy.zeros(self.args['nc'], dtype=theano.config.floatX))
        
        self.container['params'] += [first_fW, first_fb]
        self.container['names'] += ['first_fW', 'first_fb']
        
        f1_dropout = _dropout_from_layer(self.args['rng'], [f1], self.args['dropout'])
        f1_dropout = f1_dropout[0]
        
        f1_dropout = T.exp(T.dot(f1_dropout, first_fW) + first_fb)
        f1 = T.exp(T.dot(f1 , (1.0 - self.args['dropout']) * first_fW) + first_fb)
        
        #-----convolute
        
        f2, dim_second = eval(models[1])(self, -1)
        
        second_fW = theano.shared(randomMatrix(dim_second, self.args['nc']))
        second_fb = theano.shared(numpy.zeros(self.args['nc'], dtype=theano.config.floatX))
        
        self.container['params'] += [second_fW, second_fb]
        self.container['names'] += ['second_fW', 'second_fb']
        
        f2_dropout = _dropout_from_layer(self.args['rng'], [f2], self.args['dropout'])
        f2_dropout = f2_dropout[0]
        
        f2_dropout = T.exp(T.dot(f2_dropout, second_fW) + second_fb)
        f2 = T.exp(T.dot(f2 , (1.0 - self.args['dropout']) * second_fW) + second_fb)
        
        fetre = f1 * f2
        fetre_dropout = f1_dropout * f2_dropout    
        
        #-----multilayer nn
        
        if self.args['binaryCutoff'] >= 0:
            hids = [self.args['binaryFeatureDim']] + self.args['multilayerNN1'] + [self.args['nc']]
        
            layer0_multi_W = theano.shared(randomMatrix(self.args['binaryFeatureDim'], hids[1]))
            layer0_multi_b = theano.shared(numpy.zeros(hids[1], dtype=theano.config.floatX))
        
            self.container['params'] += [layer0_multi_W, layer0_multi_b]
            self.container['names'] += ['layer0_multi_fW', 'layer0_multi_fb']
        
            def recurrence(bfi, Wmat, bvec):
                idx = bfi[1:(bfi[0]+1)]
                weights = T.sum(Wmat[idx], axis=0) + bvec
                return weights
        
            firstMapped, _ = theano.scan(fn=recurrence, sequences=self.container['binaryFeatures'], outputs_info=[None], non_sequences=[layer0_multi_W, layer0_multi_b], n_steps=self.container['binaryFeatures'].shape[0])
        
            if self.args['useHeadEmbedding']:
                wordEmb_dim = self.args['embs1']['word'].shape[1]
                layer0_multi_headW = theano.shared(randomMatrix(2 * wordEmb_dim, hids[1]))
                layer0_multi_headb = theano.shared(numpy.zeros(hids[1], dtype=theano.config.floatX))
                self.container['params'] += [layer0_multi_headW, layer0_multi_headb]
                self.container['names'] += ['layer0_multi_headW', 'layer0_multi_headb']
            
                def recurrenceHead(sent, pos1, pos2, embRecur, Whmat, bhvec):
                    eb1 = embRecur[sent[pos1]]
                    eb2 = embRecur[sent[pos2]]
                    ebcon = T.cast(T.concatenate([eb1, eb2]), dtype=theano.config.floatX)
                    return T.dot(ebcon, Whmat) + bhvec
            
                headEmbs, _ = theano.scan(fn=recurrenceHead, sequences=[self.container['vars1']['word'], self.container['pos11'], self.container['pos21']], outputs_info=[None], non_sequences=[self.container['embDict1']['word'], layer0_multi_headW, layer0_multi_headb])
                firstMapped = firstMapped + headEmbs
        
            if len(hids) == 2:
                fMulti = firstMapped
                fMulti_dropout = firstMapped
            else:
                firstMapped = T.nnet.sigmoid(firstMapped)
                hids = hids[1:(len(hids)-1)]
                fetreArr = MultiHiddenLayers([firstMapped], hids, self.container['params'], self.container['names'], 'ensembleModel')
                fMulti = fetreArr[0]
                dim_multi = hids[len(hids)-1]
        
                fW = theano.shared(randomMatrix(dim_multi, self.args['nc']))
                fb = theano.shared(numpy.zeros(self.args['nc'], dtype=theano.config.floatX))
        
                self.container['params'] += [fW, fb]
                self.container['names'] += ['fW', 'fb']
        
                fMulti = T.dot(fMulti , fW) + fb
                fMulti_dropout = fMulti
            
            fMulti = T.exp(fMulti)
            fMulti_dropout = fMulti
        
            fetre = fetre * fMulti
            fetre_dropout = fetre_dropout * fMulti_dropout       
        
        su = T.cast(fetre.sum(1).dimshuffle(0,'x'), dtype=theano.config.floatX)
        su_dropout = T.cast(fetre_dropout.sum(1).dimshuffle(0,'x'), dtype=theano.config.floatX)
        
        p_y_given_x_dropout = fetre_dropout / su_dropout

        p_y_given_x = fetre / su
        
        self.buildFunctions(p_y_given_x, p_y_given_x_dropout)

class hybridModel(BaseModel):

    def __init__(self, args):

        BaseModel.__init__(self, args)
        
        fModel, dim_model = eval(self.args['model'])(self, 1)
        
        fModel_dropout = _dropout_from_layer(self.args['rng'], [fModel], self.args['dropout'])
        fModel_dropout = fModel_dropout[0]
        
        nnhids = [dim_model] + self.args['multilayerNN2']
        
        nnmul = MultiHiddenLayers([fModel, fModel_dropout], nnhids, self.container['params'], self.container['names'], 'multiHybridModelMultiNN')
        
        fModel, fModel_dropout = nnmul[0], nnmul[1]
        
        dim_model = nnhids[len(nnhids)-1]
        
        model_fW = theano.shared(randomMatrix(dim_model, self.args['nc']))
        model_fb = theano.shared(numpy.zeros(self.args['nc'], dtype=theano.config.floatX))
        
        self.container['params'] += [model_fW, model_fb]
        self.container['names'] += ['model_fW', 'model_fb']
        
        fModel_dropout = T.exp(T.dot(fModel_dropout, model_fW) + model_fb)
        fModel = T.exp(T.dot(fModel , (1.0 - self.args['dropout']) * model_fW) + model_fb)
        
        #-----multilayer nn
        
        hids = [self.args['binaryFeatureDim']] + self.args['multilayerNN1'] + [self.args['nc']]
        
        layer0_multi_W = theano.shared(randomMatrix(self.args['binaryFeatureDim'], hids[1]))
        layer0_multi_b = theano.shared(numpy.zeros(hids[1], dtype=theano.config.floatX))
        
        self.container['params'] += [layer0_multi_W, layer0_multi_b]
        self.container['names'] += ['layer0_hybrid_multi_fW', 'layer0_hybrid_multi_fb']
        
        def recurrence(bfi, Wmat, bvec):
            idx = bfi[1:(bfi[0]+1)]
            weights = T.sum(Wmat[idx], axis=0) + bvec
            return weights
        
        firstMapped, _ = theano.scan(fn=recurrence, sequences=self.container['binaryFeatures'], outputs_info=[None], non_sequences=[layer0_multi_W, layer0_multi_b], n_steps=self.container['binaryFeatures'].shape[0])
        
        if self.args['useHeadEmbedding']:
            wordEmb_dim = self.args['embs1']['word'].shape[1]
            layer0_multi_headW = theano.shared(randomMatrix(2 * wordEmb_dim, hids[1]))
            layer0_multi_headb = theano.shared(numpy.zeros(hids[1], dtype=theano.config.floatX))
            self.container['params'] += [layer0_multi_headW, layer0_multi_headb]
            self.container['names'] += ['layer0_hybrid_multi_headW', 'layer0_hybrid_multi_headb']
            
            def recurrenceHead(sent, pos1, pos2, embRecur, Whmat, bhvec):
                eb1 = embRecur[sent[pos1]]
                eb2 = embRecur[sent[pos2]]
                ebcon = T.cast(T.concatenate([eb1, eb2]), dtype=theano.config.floatX)
                return T.dot(ebcon, Whmat) + bhvec
            
            headEmbs, _ = theano.scan(fn=recurrenceHead, sequences=[self.container['vars1']['word'], self.container['pos11'], self.container['pos21']], outputs_info=[None], non_sequences=[self.container['embDict1']['word'], layer0_multi_headW, layer0_multi_headb])
            firstMapped = firstMapped + headEmbs
        
        if len(hids) == 2:
            fMulti = firstMapped
            fMulti_dropout = firstMapped
        else:
            firstMapped = T.nnet.sigmoid(firstMapped)
            hids = hids[1:(len(hids)-1)]
            fetreArr = MultiHiddenLayers([firstMapped], hids, self.container['params'], self.container['names'], 'modelMultiNN')
            fMulti = fetreArr[0]
            dim_multi = hids[len(hids)-1]
        
            fW = theano.shared(randomMatrix(dim_multi, self.args['nc']))
            fb = theano.shared(numpy.zeros(self.args['nc'], dtype=theano.config.floatX))
        
            self.container['params'] += [fW, fb]
            self.container['names'] += ['multi_fW', 'multi_fb']
        
            fMulti = T.dot(fMulti , fW) + fb
            fMulti_dropout = fMulti
            
        fMulti = T.exp(fMulti)
        fMulti_dropout = fMulti
        
        fetre = fModel * fMulti
        fetre_dropout = fModel_dropout * fMulti_dropout
        
        if self.args['kernelFets']['kernelScore'] > 0:
            fetre = fetre * T.exp(self.container['kernelScore'])
            fetre_dropout = fetre_dropout * T.exp(self.container['kernelScore'])
        
        su = T.cast(fetre.sum(1).dimshuffle(0,'x'), dtype=theano.config.floatX)
        su_dropout = T.cast(fetre_dropout.sum(1).dimshuffle(0,'x'), dtype=theano.config.floatX)
        
        p_y_given_x_dropout = fetre_dropout / su_dropout

        p_y_given_x = fetre / su
        
        self.buildFunctions(p_y_given_x, p_y_given_x_dropout)

def alternateHead(model, i):

    dimIn = model.container['dimIn' + str(i)]
    _x = getConcatenation(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], tranpose=False, outer=model.args['outer'])
    
    _x = convContextLs(_x, model.args['conv_feature_map' + str(i)], model.args['conv_win_feature_map' + str(i)], model.args['batch'], model.args['conv_winre' + str(i)], dimIn, 'convAlternate', model.container['params'], model.container['names'])
    
    _x = _x.dimshuffle(1,0,2)
    _ix = _x[::-1]
    
    dimIn = model.args['conv_feature_map' + str(i)] * len(model.args['conv_win_feature_map' + str(i)])
    
    _x = gruBidirectCore(_x, _ix, dimIn, model.args['nh' + str(i)], model.args['batch'], 'rnnAlternate', 'irnnAlternate', model.container['params'], model.container['names'])
    
    return rnnHeadIn(model, _x, i, 4)
    
def alternateHeadForward(model, i):

    dimIn = model.container['dimIn' + str(i)]
    _x = getConcatenation(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], tranpose=False, outer=model.args['outer'])
    
    _x = convContextLs(_x, model.args['conv_feature_map' + str(i)], model.args['conv_win_feature_map' + str(i)], model.args['batch'], model.args['conv_winre' + str(i)], dimIn, 'convAlternate', model.container['params'], model.container['names'])
    
    _x = _x.dimshuffle(1,0,2)
    
    dimIn = model.args['conv_feature_map' + str(i)] * len(model.args['conv_win_feature_map' + str(i)])
    
    _x = rnn_gru(_x, dimIn, model.args['nh' + str(i)], model.args['batch'], 'rnnAlternateForward', model.container['params'], model.container['names'])
    
    _x = T.cast(_x.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rnnHeadIn(model, _x, i, 2)
    
def alternateHeadBackward(model, i):

    dimIn = model.container['dimIn' + str(i)]
    _x = getConcatenation(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], tranpose=False, outer=model.args['outer'])
    
    _x = convContextLs(_x, model.args['conv_feature_map' + str(i)], model.args['conv_win_feature_map' + str(i)], model.args['batch'], model.args['conv_winre' + str(i)], dimIn, 'convAlternate', model.container['params'], model.container['names'])
    
    _x = _x.dimshuffle(1,0,2)[::-1]
    
    dimIn = model.args['conv_feature_map' + str(i)] * len(model.args['conv_win_feature_map' + str(i)])
    
    _x = rnn_gru(_x, dimIn, model.args['nh' + str(i)], model.args['batch'], 'rnnAlternateBackward', model.container['params'], model.container['names'])
    
    _x = T.cast(_x[::-1].dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rnnHeadIn(model, _x, i, 2)
    
def alternateHeadDeep(model, i):
    
    depth = 2
    
    dimIn = model.container['dimIn' + str(i)]
    _x = getConcatenation(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], tranpose=False, outer=model.args['outer'])
    
    for deid in range(depth):
        _x = convContextLs(_x, model.args['conv_feature_map' + str(i)], model.args['conv_win_feature_map' + str(i)], model.args['batch'], model.args['conv_winre' + str(i)], dimIn, 'convAlternate' + str(deid), model.container['params'], model.container['names'])
    
        _x = _x.dimshuffle(1,0,2)
        _ix = _x[::-1]
    
        dimIn = model.args['conv_feature_map' + str(i)] * len(model.args['conv_win_feature_map' + str(i)])
    
        _x = gruBidirectCore(_x, _ix, dimIn, model.args['nh' + str(i)], model.args['batch'], 'rnnAlternate' + str(deid), 'irnnAlternate' + str(deid), model.container['params'], model.container['names'])
    
    return rnnHeadIn(model, _x, i, 4)

def alternateMax(model, i):

    dimIn = model.container['dimIn' + str(i)]
    _x = getConcatenation(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], tranpose=False, outer=model.args['outer'])
    
    _x = convContextLs(_x, model.args['conv_feature_map' + str(i)], model.args['conv_win_feature_map' + str(i)], model.args['batch'], model.args['conv_winre' + str(i)], dimIn, 'convAlternate', model.container['params'], model.container['names'])
    
    _x = _x.dimshuffle(1,0,2)
    _ix = _x[::-1]
    
    dimIn = model.args['conv_feature_map' + str(i)] * len(model.args['conv_win_feature_map' + str(i)])
    
    _x = gruBidirectCore(_x, _ix, dimIn, model.args['nh' + str(i)], model.args['batch'], 'rnnAlternate', 'irnnAlternate', model.container['params'], model.container['names'])
    
    fRnn = T.cast(T.max(_x, axis=1), dtype=theano.config.floatX)
        
    dim_rnn = 2 * model.args['nh' + str(i)]
    
    return fRnn, dim_rnn

def alternateMaxForward(model, i):

    dimIn = model.container['dimIn' + str(i)]
    _x = getConcatenation(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], tranpose=False, outer=model.args['outer'])
    
    _x = convContextLs(_x, model.args['conv_feature_map' + str(i)], model.args['conv_win_feature_map' + str(i)], model.args['batch'], model.args['conv_winre' + str(i)], dimIn, 'convAlternate', model.container['params'], model.container['names'])
    
    _x = _x.dimshuffle(1,0,2)
    
    dimIn = model.args['conv_feature_map' + str(i)] * len(model.args['conv_win_feature_map' + str(i)])
    
    _x = rnn_gru(_x, dimIn, model.args['nh' + str(i)], model.args['batch'], 'rnnAlternateForward', model.container['params'], model.container['names'])
    
    _x = T.cast(_x.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    fRnn = T.cast(T.max(_x, axis=1), dtype=theano.config.floatX)
        
    dim_rnn = model.args['nh' + str(i)]
    
    return fRnn, dim_rnn
    
def alternateMaxBackward(model, i):

    dimIn = model.container['dimIn' + str(i)]
    _x = getConcatenation(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], tranpose=False, outer=model.args['outer'])
    
    _x = convContextLs(_x, model.args['conv_feature_map' + str(i)], model.args['conv_win_feature_map' + str(i)], model.args['batch'], model.args['conv_winre' + str(i)], dimIn, 'convAlternate', model.container['params'], model.container['names'])
    
    _x = _x.dimshuffle(1,0,2)[::-1]
    
    dimIn = model.args['conv_feature_map' + str(i)] * len(model.args['conv_win_feature_map' + str(i)])
    
    _x = rnn_gru(_x, dimIn, model.args['nh' + str(i)], model.args['batch'], 'rnnAlternateBackward', model.container['params'], model.container['names'])
    
    _x = T.cast(_x[::-1].dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    fRnn = T.cast(T.max(_x, axis=1), dtype=theano.config.floatX)
        
    dim_rnn = model.args['nh' + str(i)]
    
    return fRnn, dim_rnn
    
def alternateConv(model, i):

    _x = gruBiDirect(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'alternateConvRnn', model.container['params'], model.container['names'], outer=model.args['outer'])
    
    dimIn = 2 * model.args['nh' + str(i)]
    
    fConv = convContext(_x, model.args['conv_feature_map' + str(i)], model.args['conv_win_feature_map' + str(i)], model.args['batch'], model.args['conv_winre' + str(i)], dimIn, 'alternateConvConv', model.container['params'], model.container['names'])

    dim_conv = model.args['conv_feature_map' + str(i)] * len(model.args['conv_win_feature_map' + str(i)])
    
    return fConv, dim_conv
    
def alternateConvForward(model, i):

    _x = gruForward(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'alternateConvRnnForward', model.container['params'], model.container['names'], outer=model.args['outer'])
    
    dimIn = model.args['nh' + str(i)]
    
    fConv = convContext(_x, model.args['conv_feature_map' + str(i)], model.args['conv_win_feature_map' + str(i)], model.args['batch'], model.args['conv_winre' + str(i)], dimIn, 'alternateConvConv', model.container['params'], model.container['names'])

    dim_conv = model.args['conv_feature_map' + str(i)] * len(model.args['conv_win_feature_map' + str(i)])
    
    return fConv, dim_conv
    
def alternateConvBackward(model, i):

    _x = gruBackward(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'alternateConvRnnBackward', model.container['params'], model.container['names'], outer=model.args['outer'])
    
    dimIn = model.args['nh' + str(i)]
    
    fConv = convContext(_x, model.args['conv_feature_map' + str(i)], model.args['conv_win_feature_map' + str(i)], model.args['batch'], model.args['conv_winre' + str(i)], dimIn, 'alternateConvConv', model.container['params'], model.container['names'])

    dim_conv = model.args['conv_feature_map' + str(i)] * len(model.args['conv_win_feature_map' + str(i)])
    
    return fConv, dim_conv

def convolute(model, i):
    _x = getConcatenation(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], tranpose=False, outer=model.args['outer'])
        
    fConv = convContext(_x, model.args['conv_feature_map' + str(i)], model.args['conv_win_feature_map' + str(i)], model.args['batch'], model.args['conv_winre' + str(i)], model.container['dimIn' + str(i)], 'convolutionLs', model.container['params'], model.container['names'])
        
    dim_conv = model.args['conv_feature_map' + str(i)] * len(model.args['conv_win_feature_map' + str(i)])
    
    return fConv, dim_conv
    
def convoluteSum(model, i):
    dimIn = model.container['dimIn' + str(i)]
    _x = getConcatenation(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], tranpose=False, outer=model.args['outer'])
    
    _x = convContextLs(_x, model.args['conv_feature_map' + str(i)], model.args['conv_win_feature_map' + str(i)], model.args['batch'], model.args['conv_winre' + str(i)], dimIn, 'convoluteSum', model.container['params'], model.container['names'])
    
    fConv = _x.mean(1)
    
    dim_conv = model.args['conv_feature_map' + str(i)] * len(model.args['conv_win_feature_map' + str(i)])
    
    return fConv, dim_conv
    
def convoluteSumDep(model, i):
    dimIn = model.container['dimIn' + str(i)]
    _x = getConcatenation(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], tranpose=False, outer=model.args['outer'])
    
    _x = convContextLs(_x, model.args['conv_feature_map' + str(i)], model.args['conv_win_feature_map' + str(i)], model.args['batch'], model.args['conv_winre' + str(i)], dimIn, 'convoluteSumDep', model.container['params'], model.container['names'])
    
    def recurrence(x_i, dep_i):
        fet = (x_i * T.addbroadcast(dep_i, 1).dimshuffle(0,'x')).sum(0)
        return [fet]
    
    fConv, _ = theano.scan(fn=recurrence, \
            sequences=[_x, model.container['iidep' + str(i)]], outputs_info=[None], n_steps=_x.shape[0])
    
    dim_conv = model.args['conv_feature_map' + str(i)] * len(model.args['conv_win_feature_map' + str(i)])
    
    return fConv, dim_conv

def rnnHead(model, i):
    _x = gruBiDirect(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnHead', model.container['params'], model.container['names'], outer=model.args['outer'])
    return rnnHeadIn(model, _x, i, 4)
    
def rnnHeadForward(model, i):
    _x = gruForward(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnHeadForward', model.container['params'], model.container['names'], outer=model.args['outer'])
    return rnnHeadIn(model, _x, i, 2)

def rnnHeadBackward(model, i):
    _x = gruBackward(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnHeadBacward', model.container['params'], model.container['names'], outer=model.args['outer'])
    return rnnHeadIn(model, _x, i, 2)

def rnnHeadFf(model, i):
    _x = ffBiDirect(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnHeadFf', model.container['params'], model.container['names'], outer=model.args['outer'])
    return rnnHeadIn(model, _x, i, 4)

def rnnHeadFfForward(model, i):
    _x = ffForward(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnHeadFfForward', model.container['params'], model.container['names'], outer=model.args['outer'])
    return rnnHeadIn(model, _x, i, 2)

def rnnHeadFfBackward(model, i):
    _x = ffBackward(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnHeadFfBackward', model.container['params'], model.container['names'], outer=model.args['outer'])
    return rnnHeadIn(model, _x, i, 2)
    
def rnnHeadIn(model, _x, i, num):
        
    def recurrence(x_i, pos1, pos2):
        fet = T.cast(T.concatenate([x_i[pos1], x_i[pos2]]), dtype=theano.config.floatX)
        return [fet]
        
    fRnn, _ = theano.scan(fn=recurrence, \
            sequences=[_x, model.container['pos1' + str(i)], model.container['pos2' + str(i)]], outputs_info=[None], n_steps=_x.shape[0])
        
    dim_rnn = num * model.args['nh' + str(i)]
    
    return fRnn, dim_rnn

def rnnMax(model, i):
    _x = gruBiDirect(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnMax', model.container['params'], model.container['names'], outer=model.args['outer'])
    return rnnMaxIn(model, _x, i, 2)
    
def rnnMaxForward(model, i):
    _x = gruForward(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnMaxForward', model.container['params'], model.container['names'], outer=model.args['outer'])
    return rnnMaxIn(model, _x, i, 1)

def rnnMaxBackward(model, i):
    _x = gruBackward(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnMaxBacward', model.container['params'], model.container['names'], outer=model.args['outer'])
    return rnnMaxIn(model, _x, i, 1)
    
def rnnMaxFf(model, i):
    _x = ffBiDirect(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnMaxFf', model.container['params'], model.container['names'], outer=model.args['outer'])
    return rnnMaxIn(model, _x, i, 2)

def rnnMaxFfForward(model, i):
    _x = ffForward(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnMaxFfForward', model.container['params'], model.container['names'], outer=model.args['outer'])
    return rnnMaxIn(model, _x, i, 1)

def rnnMaxFfBackward(model, i):
    _x = ffBackward(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnMaxFfBackward', model.container['params'], model.container['names'], outer=model.args['outer'])
    return rnnMaxIn(model, _x, i, 1)

def rnnMaxIn(model, _x, i, num):
    fRnn = T.cast(T.max(_x, axis=1), dtype=theano.config.floatX)
        
    dim_rnn = num * model.args['nh' + str(i)]
    
    return fRnn, dim_rnn
    
def rnnSum(model, i):
    _x = gruBiDirect(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnSum', model.container['params'], model.container['names'], outer=model.args['outer'])
        
    fRnn = _x.mean(1)
        
    dim_rnn = 2 * model.args['nh' + str(i)]
    
    return fRnn, dim_rnn
    
def rnnSumDep(model, i):
    _x = gruBiDirect(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnSumDep', model.container['params'], model.container['names'], outer=model.args['outer'])
        
    def recurrence(x_i, dep_i):
        fet = (x_i * T.addbroadcast(dep_i, 1).dimshuffle(0,'x')).sum(0)
        return [fet]
    
    fRnn, _ = theano.scan(fn=recurrence, \
            sequences=[_x, model.container['iidep' + str(i)]], outputs_info=[None], n_steps=_x.shape[0])
        
    dim_rnn = 2 * model.args['nh' + str(i)]
    
    return fRnn, dim_rnn
    
def rnnAtt(model, i):
    _x = gruBiDirect(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnAtt', model.container['params'], model.container['names'], outer=model.args['outer'])
    
    IW = theano.shared(randomMatrix(2 * model.args['nh' + str(i)], 1))
    Ib = theano.shared(numpy.zeros(1, dtype=theano.config.floatX))
        
    model.container['params'] += [IW, Ib]
    model.container['names'] += ['rnnAt_IW', 'rnnAT_Ib']     
        
    def recurrence(x_i):
        alpha = T.dot(x_i, IW) + Ib
        alpha = T.exp(alpha)
        alpha = alpha / T.sum(alpha)
        fet = (x_i * T.addbroadcast(alpha, 1).dimshuffle(0,'x')).sum(0)
        return [fet]
        
    fRnn, _ = theano.scan(fn=recurrence, \
            sequences=_x, outputs_info=[None], n_steps=_x.shape[0])
                
    dim_rnn = 2 * model.args['nh' + str(i)]
    
    return fRnn, dim_rnn
    
def rnnAttHead(model, i):
    _x = gruBiDirect(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnAtt', model.container['params'], model.container['names'], outer=model.args['outer'])
    
    IW = theano.shared(randomMatrix(2 * model.args['nh' + str(i)], 1))
    Ib = theano.shared(numpy.zeros(1, dtype=theano.config.floatX))
        
    model.container['params'] += [IW, Ib]
    model.container['names'] += ['rnnAt_IW', 'rnnAT_Ib']     
        
    def recurrenceAtt(x_i):
        alpha = T.dot(x_i, IW) + Ib
        alpha = T.exp(alpha)
        alpha = alpha / T.sum(alpha)
        fet = (x_i * T.addbroadcast(alpha, 1).dimshuffle(0,'x')).sum(0)
        return [fet]
        
    fRnnAtt, _ = theano.scan(fn=recurrenceAtt, \
            sequences=_x, outputs_info=[None], n_steps=_x.shape[0])
            
    def recurrenceHead(x_i, pos1, pos2):
        fet = T.cast(T.concatenate([x_i[pos1], x_i[pos2]]), dtype=theano.config.floatX)
        return [fet]
        
    fRnnHead, _ = theano.scan(fn=recurrenceHead, \
            sequences=[_x, model.container['pos1' + str(i)], model.container['pos2' + str(i)]], outputs_info=[None], n_steps=_x.shape[0])
    
    fRnn = T.cast(T.concatenate([fRnnAtt, fRnnHead], axis=1), dtype=theano.config.floatX)
                
    dim_rnn = 6 * model.args['nh' + str(i)]
    
    return fRnn, dim_rnn

#####################################

class dynamicpooling(BaseModel):
    
    def __init__(self, args):

        BaseModel.__init__(self, args)
        
        self.generateInput()
        
        def recurrence(x_i, pos1, pos2):
            beforeZero, betweenZero, afterZero = T.zeros_like(x_i), T.zeros_like(x_i), T.zeros_like(x_i)
            before = T.switch(pos1 <= 0, beforeZero, T.set_subtensor(beforeZero[0:pos1], x_i[0:pos1]))
            between = T.switch(pos1 > pos2-2, betweenZero, T.set_subtensor(betweenZero[(pos1+1):(pos2)], x_i[(pos1+1):(pos2)]))
            after = T.switch(pos2+1 >= x_i.shape[0], afterZero, T.set_subtensor(afterZero[(pos2+1):], x_i[(pos2+1):]))
            els = [T.max(before, axis=0), x_i[pos1], T.max(between, axis=0), x_i[pos2], T.max(after, axis=0)]
            fet = T.cast(T.concatenate(els), dtype=theano.config.floatX)
            return [fet]
        
        fetre, _ = theano.scan(fn=recurrence, \
                sequences=[self.container['x'], self.container['pos1'], self.container['pos2']], outputs_info=[None], n_steps=self.container['x'].shape[0])
        dim_inter = 5 * self.container['dimIn']
        
        hids = [dim_inter] + self.container['multilayerNN']
        
        fetre = MultiHiddenLayers(fetre, hids, self.container['params'], self.container['names'])
        
        fetre_dropout = _dropout_from_layer(self.container['rng'], fetre, self.container['dropout'])
        
        dim_inter = hids[len(hids)-1]
        
        self.container['W'] = theano.shared(randomMatrix(dim_inter, self.container['nc']))
        self.container['b'] = theano.shared(numpy.zeros(self.container['nc'], dtype=theano.config.floatX))
        
        self.container['params'] += [self.container['W'], self.container['b']]
        self.container['names'] += ['W', 'b']
        
        p_y_given_x_dropout = T.nnet.softmax(T.dot(fetre_dropout, self.container['W']) + self.container['b'])
        
        p_y_given_x = T.nnet.softmax(T.dot(fetre , (1.0 - self.container['dropout']) * self.container['W']) + self.container['b'])
        
        self.buildFunctions(p_y_given_x, p_y_given_x_dropout)
        
class convRnnDynamicMax(BaseModel):

    def __init__(self, args):

        BaseModel.__init__(self, args)
        
        if self.container['encoding'] != 'basicRep':
            self.container['dimIn'] = self.container['bi_dimIn']
            self.container['bi_dimIn'] = 0
        self.container['encoding'] = 'basicRep'
        self.generateInput()
        
        fConv = convContext(self)
        dim_conv = self.container['conv_feature_map'] * len(self.container['conv_win_feature_map'])
        
        self.container['bi_dimIn'] = self.container['dimIn']
        self.container['dimIn'] = 0
        self.container['encoding'] = 'gruBiDirect'
        self.generateInput()
        
        def recurrence(x_i, pos1, pos2):
            beforeZero, betweenZero, afterZero = T.zeros_like(x_i), T.zeros_like(x_i), T.zeros_like(x_i)
            before = T.switch(pos1 <= 0, beforeZero, T.set_subtensor(beforeZero[0:pos1], x_i[0:pos1]))
            between = T.switch(pos1 > pos2-2, betweenZero, T.set_subtensor(betweenZero[(pos1+1):(pos2)], x_i[(pos1+1):(pos2)]))
            after = T.switch(pos2+1 >= x_i.shape[0], afterZero, T.set_subtensor(afterZero[(pos2+1):], x_i[(pos2+1):]))
            els = [T.max(before, axis=0), x_i[pos1], T.max(between, axis=0), x_i[pos2], T.max(after, axis=0)]
            fet = T.cast(T.concatenate(els), dtype=theano.config.floatX)
            return [fet]
        
        fRnn, _ = theano.scan(fn=recurrence, \
                sequences=[self.container['x'], self.container['pos1'], self.container['pos2']], outputs_info=[None], n_steps=self.container['x'].shape[0])
        dim_rnn = 5 * self.container['dimIn']
        
        fetre = T.cast(T.concatenate([fConv, fRnn], axis=1), dtype=theano.config.floatX)
        dim_inter = dim_conv + dim_rnn
        
        hids = [dim_inter] + self.container['multilayerNN']
        
        fetre = MultiHiddenLayers(fetre, hids, self.container['params'], self.container['names'])
        
        fetre_dropout = _dropout_from_layer(self.container['rng'], fetre, self.container['dropout'])
        
        dim_inter = hids[len(hids)-1]
        
        print '.......... Feature Dimensions in the combined model(convRnnDynamicMax): ', dim_inter
        
        self.container['W'] = theano.shared(randomMatrix(dim_inter, self.container['nc']))
        self.container['b'] = theano.shared(numpy.zeros(self.container['nc'], dtype=theano.config.floatX))
        
        self.container['params'] += [self.container['W'], self.container['b']]
        self.container['names'] += ['W', 'b']
        
        p_y_given_x_dropout = T.nnet.softmax(T.dot(fetre_dropout, self.container['W']) + self.container['b'])
        
        p_y_given_x = T.nnet.softmax(T.dot(fetre , (1.0 - self.container['dropout']) * self.container['W']) + self.container['b'])
        
        self.buildFunctions(p_y_given_x, p_y_given_x_dropout)
        
class rnnFfMaxPieces(BaseModel):

    def __init__(self, args):

        BaseModel.__init__(self, args)
        if self.container['encoding'] != 'basicRep':
            self.container['dimIn'] = self.container['bi_dimIn']
            self.container['bi_dimIn'] = 0
        self.container['encoding'] = 'basicRep'
        
        ix = getConcatenation(self)
        
        def recurrence(x_i, pos1, pos2):
            before = T.cast(T.set_subtensor(T.zeros_like(x_i)[0:(pos1+1)], x_i[0:(pos1+1)]), dtype=theano.config.floatX)
            between = T.cast(T.set_subtensor(T.zeros_like(x_i)[(pos1+1):(pos2+1)], x_i[(pos1+1):(pos2+1)]), dtype=theano.config.floatX)
            after = T.cast(T.switch(pos2+1 >= x_i.shape[0], T.zeros_like(x_i), T.set_subtensor(T.zeros_like(x_i)[(pos2+1):], x_i[(pos2+1):])), dtype=theano.config.floatX)
            ibefore = before[::-1]
            ibetween = between[::-1]
            iafter = after[::-1]
            return before, between, after, ibefore, ibetween, iafter
        
        ixp, _ = theano.scan(fn=recurrence, \
                sequences=[ix, self.container['pos1'], self.container['pos2']], outputs_info=[None, None, None, None, None, None], n_steps=ix.shape[0])
        
        tixp = [T.cast(ip.dimshuffle(1,0,2), dtype=theano.config.floatX) for ip in ixp]
        
        befo = ffBidirectCore(tixp[0], tixp[3], self.container['dimIn'], self.container['nh'], self.container['batch'], 'ffbiBefore', 'ffibiBefore', self.container['params'], self.container['names'])
        beto = ffBidirectCore(tixp[1], tixp[4], self.container['dimIn'], self.container['nh'], self.container['batch'], 'ffbiBetween', 'ffibiBetween', self.container['params'], self.container['names'])
        afto = ffBidirectCore(tixp[2], tixp[5], self.container['dimIn'], self.container['nh'], self.container['batch'], 'ffbiAfter', 'ffibiAfter', self.container['params'], self.container['names'])
        
        befo = T.cast(T.max(befo, axis=1), dtype=theano.config.floatX)
        beto = T.cast(T.max(beto, axis=1), dtype=theano.config.floatX)
        afto = T.cast(T.max(afto, axis=1), dtype=theano.config.floatX)
        
        fetre = T.cast(T.concatenate([befo, beto, afto], axis=1), dtype=theano.config.floatX)
        dim_inter = 6 * self.container['nh']
        
        hids = [dim_inter] + self.container['multilayerNN']
        
        fetre = MultiHiddenLayers(fetre, hids, self.container['params'], self.container['names'])
        
        fetre_dropout = _dropout_from_layer(self.container['rng'], fetre, self.container['dropout'])
        
        dim_inter = hids[len(hids)-1]
        
        print '.......... Feature Dimensions in the combined model(rnnFfMaxPieces): ', dim_inter
        
        self.container['W'] = theano.shared(randomMatrix(dim_inter, self.container['nc']))
        self.container['b'] = theano.shared(numpy.zeros(self.container['nc'], dtype=theano.config.floatX))
        
        self.container['params'] += [self.container['W'], self.container['b']]
        self.container['names'] += ['W', 'b']
        
        p_y_given_x_dropout = T.nnet.softmax(T.dot(fetre_dropout, self.container['W']) + self.container['b'])
        
        p_y_given_x = T.nnet.softmax(T.dot(fetre , (1.0 - self.container['dropout']) * self.container['W']) + self.container['b'])
        
        self.buildFunctions(p_y_given_x, p_y_given_x_dropout)
        
class rnnGruMaxPieces(BaseModel):

    def __init__(self, args):

        BaseModel.__init__(self, args)
        if self.container['encoding'] != 'basicRep':
            self.container['dimIn'] = self.container['bi_dimIn']
            self.container['bi_dimIn'] = 0
        self.container['encoding'] = 'basicRep'
        
        ix = getConcatenation(self)
        
        def recurrence(x_i, pos1, pos2):
            before = T.cast(T.set_subtensor(T.zeros_like(x_i)[0:(pos1+1)], x_i[0:(pos1+1)]), dtype=theano.config.floatX)
            between = T.cast(T.set_subtensor(T.zeros_like(x_i)[(pos1+1):(pos2+1)], x_i[(pos1+1):(pos2+1)]), dtype=theano.config.floatX)
            after = T.cast(T.switch(pos2+1 >= x_i.shape[0], T.zeros_like(x_i), T.set_subtensor(T.zeros_like(x_i)[(pos2+1):], x_i[(pos2+1):])), dtype=theano.config.floatX)
            ibefore = before[::-1]
            ibetween = between[::-1]
            iafter = after[::-1]
            return before, between, after, ibefore, ibetween, iafter
        
        ixp, _ = theano.scan(fn=recurrence, \
                sequences=[ix, self.container['pos1'], self.container['pos2']], outputs_info=[None, None, None, None, None, None], n_steps=ix.shape[0])
        
        tixp = [T.cast(ip.dimshuffle(1,0,2), dtype=theano.config.floatX) for ip in ixp]
        
        befo = gruBidirectCore(tixp[0], tixp[3], self.container['dimIn'], self.container['nh'], self.container['batch'], 'grubiBefore', 'gruibiBefore', self.container['params'], self.container['names'])
        beto = gruBidirectCore(tixp[1], tixp[4], self.container['dimIn'], self.container['nh'], self.container['batch'], 'grubiBetween', 'gruibiBetween', self.container['params'], self.container['names'])
        afto = gruBidirectCore(tixp[2], tixp[5], self.container['dimIn'], self.container['nh'], self.container['batch'], 'grubiAfter', 'gruibiAfter', self.container['params'], self.container['names'])
        
        befo = T.cast(T.max(befo, axis=1), dtype=theano.config.floatX)
        beto = T.cast(T.max(beto, axis=1), dtype=theano.config.floatX)
        afto = T.cast(T.max(afto, axis=1), dtype=theano.config.floatX)
        
        fetre = T.cast(T.concatenate([befo, beto, afto], axis=1), dtype=theano.config.floatX)
        dim_inter = 6 * self.container['nh']
        
        hids = [dim_inter] + self.container['multilayerNN']
        
        fetre = MultiHiddenLayers(fetre, hids, self.container['params'], self.container['names'])
        
        fetre_dropout = _dropout_from_layer(self.container['rng'], fetre, self.container['dropout'])
        
        dim_inter = hids[len(hids)-1]
        
        print '.......... Feature Dimensions in the combined model(rnnGruMaxPieces): ', dim_inter
        
        self.container['W'] = theano.shared(randomMatrix(dim_inter, self.container['nc']))
        self.container['b'] = theano.shared(numpy.zeros(self.container['nc'], dtype=theano.config.floatX))
        
        self.container['params'] += [self.container['W'], self.container['b']]
        self.container['names'] += ['W', 'b']
        
        p_y_given_x_dropout = T.nnet.softmax(T.dot(fetre_dropout, self.container['W']) + self.container['b'])
        
        p_y_given_x = T.nnet.softmax(T.dot(fetre , (1.0 - self.container['dropout']) * self.container['W']) + self.container['b'])
        
        self.buildFunctions(p_y_given_x, p_y_given_x_dropout)