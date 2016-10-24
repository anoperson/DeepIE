from rnnRE import train
from collections import OrderedDict

def main(params):
    print params
    train(fold = params['fold'],
          outer = params['outer'],
          model = params['model'],
          seqType = params['seqType'],
          collapsed = params['collapsed'],
          expected_features1 = params['expected_features1'],
          expected_features2 = params['expected_features2'],
          kernelFets = params['kernelFets'],
          sharedEmbs = params['sharedEmbs'],
          withEmbs = params['withEmbs'],
          updateEmbs = params['updateEmbs'],
          optimizer = params['optimizer'],
          lr = params['lr'],
          dropout = params['dropout'],
          regularizer = params['regularizer'],
          norm_lim = params['norm_lim'],
          verbose = params['verbose'],
          decay = params['decay'],
          batch = params['batch'],
          binaryCutoff = params['binaryCutoff'],
          useHeadEmbedding = params['useHeadEmbedding'],
          multilayerNN1 = params['multilayerNN1'],
          multilayerNN2 = params['multilayerNN2'],
          nhidden1 = params['nhidden1'],
          nhidden2 = params['nhidden2'],
          conv_winre1 = params['conv_winre1'],
          conv_winre2 = params['conv_winre2'],
          conv_feature_map1 = params['conv_feature_map1'],
          conv_feature_map2 = params['conv_feature_map2'],
          conv_win_feature_map1 = params['conv_win_feature_map1'],
          conv_win_feature_map2 = params['conv_win_feature_map2'],
          seed = params['seed'],
          #emb_dimension=300, # dimension of word embedding
          nepochs = params['nepochs'],
          folder = params['folder'])
def fetStr(ef):
    res = ''
    for f in ef:
        res += str(ef[f])
    return res

def fmStr(ft):
    res = ''
    for f in ft:
        res += str(f)
    return res

if __name__=='__main__':
    pars={'fold' : 'all',
          'outer' : False,
          'model' : '#MultiNN', # convolute, convoluteSum, convoluteSumDep # rnnHead, rnnMax, rnnHeadFf, rnnMaxFf, rnnHeadForward, rnnHeadBackward, rnnMaxForward, rnnMaxBackward, rnnHeadFfForward, rnnHeadFfBackward, rnnMaxFfForward, rnnMaxFfBackward, rnnAtt, rnnSum, rnnSumDep, # alternateHead, alternateMax, alternateConv,  #MultiNN
          'seqType' : '-dep', #-dep
          'collapsed' : False,
          'expected_features1' : OrderedDict([('dist1', -1),
                                              ('dist2', -1),
                                              ('type', 0),
                                              ('subtype', -1),
                                              ('order', -1),
                                              ('constit1', -1),
                                              ('constit2', -1),
                                              ('preter', -1),
                                              ('prepreter', 0),
                                              ('grammar', 1),
                                              ('gov', -1),
                                              ('indep', 1)]),
                                             
          'expected_features2' : OrderedDict([('dist1', -1),
                                              ('dist2', -1),
                                              ('type', -1),
                                              ('subtype', -1),
                                              ('order', -1),
                                              ('constit1', -1),
                                              ('constit2', -1),
                                              ('preter', -1),
                                              ('prepreter', -1),
                                              ('grammar', -1),
                                              ('gov', -1),
                                              ('indep', -1)]),
                                              
          'sharedEmbs' :         OrderedDict([('word', 1),
                                              ('dist1', 0),
                                              ('dist2', 0),
                                              ('type', 0),
                                              ('subtype', 0),
                                              ('order', 0),
                                              ('constit1', 0),
                                              ('constit2', 0),
                                              ('preter', 0),
                                              ('prepreter', 0),
                                              ('grammar', 0),
                                              ('gov', 0),
                                              ('indep', 0)]),
                                              
          'kernelFets' :         OrderedDict([('kernelPred', 0),
                                              ('kernelScore', 0)]),
                                              
          'withEmbs' : True,
          'updateEmbs' : True,
          'optimizer' : 'adadelta',
          'lr' : 0.01,
          'dropout' : 0.5,
          'regularizer' : 0.0,
          'norm_lim' : 9.0,
          'verbose' : 1,
          'decay' : False,
          'batch' : 50,
          'binaryCutoff' : 2,
          'useHeadEmbedding' : False,
          'multilayerNN1' : [300],
          'multilayerNN2' : [],
          'nhidden1' : 300,
          'nhidden2' : 300,
          'conv_winre1' : 20,
          'conv_winre2' : 20,
          'conv_feature_map1' : 150,
          'conv_feature_map2' : 150,
          'conv_win_feature_map1' : [2,3,4,5],
          'conv_win_feature_map2' : [2,3,4,5],
          'seed' : 3435,
          'nepochs' : 20,
          'folder' : './res'}
    folder = 'fold_' + pars['fold'] \
             + '.model_' + pars['model'] \
             + '.st_' + pars['seqType'] \
             + '.cl_' + ('1' if pars['collapsed'] else '0') \
             + '.h1_' + str(pars['nhidden1']) \
             + '.h2_' + str(pars['nhidden2']) \
             + '.outer_' + str(pars['outer']) \
             + '.embs_' + str(pars['withEmbs']) \
             + '.upd_' + str(pars['updateEmbs']) \
             + '.batch_' + str(pars['batch']) \
             + '.cut_' + str(pars['binaryCutoff']) \
             + '.he_' + ('1' if pars['useHeadEmbedding'] else '0') \
             + '.mul1_' + fmStr(pars['multilayerNN1']) \
             + '.mul2_' + fmStr(pars['multilayerNN2']) \
             + '.opt_' + pars['optimizer'] \
             + '.drop_' + str(pars['dropout']) \
             + '.reg_' + str(pars['regularizer']) \
             + '.fet1_' + fetStr(pars['expected_features1']) \
             + '.ke_' + fetStr(pars['kernelFets']) \
             + '.cvw1_' + str(pars['conv_winre1']) \
             + '.cvw2_' + str(pars['conv_winre2']) \
             + '.cvft1_' + str(pars['conv_feature_map1']) \
             + '.cvft2_' + str(pars['conv_feature_map2']) \
             + '.cvfm1_' + fmStr(pars['conv_win_feature_map1']) \
             + '.nm_' + str(pars['norm_lim'])
    pars['folder'] =  '' + folder
    main(pars)

#+ '.fet2_' + fetStr(pars['expected_features2']) \
#+ '.cvfm2_' + fmStr(pars['conv_win_feature_map2']) \
#+ '.lr_' + str(pars['lr']) \
#+ '.norm_' + str(pars['norm_lim']
