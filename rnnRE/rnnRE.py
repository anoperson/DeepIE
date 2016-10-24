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
from rnnReModels import *

dataset_path = './word2vec.full_rnnRE_Plank.pkl'

##################################################################

def getTypeDict(numType):
    res = {}
    id = 1
    for i in range(numType):
        for j in range(numType):
            res[str(i+1) + 'x' + str(j+1)] = id
            id += 1
    return res

def _generatePartialDataInstance(rev, dictionaries, embeddings, features, mLen):

    idx2label = dict((k,v) for v,k in dictionaries['label'].iteritems())
    goldRel = idx2label[ (rev["y"] if 'expanded_idx2label' not in dictionaries else dictionaries['label'][dictionaries['expanded_idx2label'][rev["y"]].replace('(e1,e2)','').replace('(e2,e1)','')]) ]
    orderId = 0
    if goldRel.endswith('(e2,e1)'): orderId = 1
    if "order" in rev:
        if rev["order"] != 0 and rev["order"] != 1:
            print 'wrong order input'
            exit()
        orderId = rev["order"]

    numPosition = embeddings['dist11'].shape[0]-1
    numDepPosition = embeddings['dep_dist11'].shape[0]-1
    numType = embeddings['type1'].shape[0]-1
    numSubtype = embeddings['subtype1'].shape[0]-1
    numOrder = embeddings['order1'].shape[0]-1
    numConstit = embeddings['constit11'].shape[0]-1
    numPreter = embeddings['preter1'].shape[0]-1
    numPrepreter = embeddings['prepreter1'].shape[0]-1
    numDeprel = embeddings['deprel1'].shape[0]-1
    numIndep = embeddings['indep1'].shape[0]-1

    x = []
    dist1 = []
    dist2 = []
    type = []
    subtype = []
    order = []
    constit1 = []
    constit2 = []
    preter = []
    prepreter = []
    grammar = []
    gov = []
    indep = []
    iidep = []
    
    #typeDic = getTypeDict(numType)
    
    id = -1
    for word, go in zip(rev["text"].split(), rev["gov"].split()):
        id += 1
        word = ' '.join(word.split('_'))
        go = ' '.join(go.split('_'))
        c1, c2, pr, ppr = rev["cons1"][id], rev["cons2"][id], rev["preter"][id], rev["prepreter"][id]
        if word in dictionaries["word"] and go in dictionaries["word"]: #and c1 in dictionaries["constit"] and c2 in dictionaries["constit"] and pr in dictionaries["preter"] and ppr in dictionaries["prepreter"]
            x.append(dictionaries["word"][word])
            
            gov.append(dictionaries["word"][go])
            
            graFet = [0] * numDeprel
            for gid in rev["grammar"][id]:
                graFet[gid-1] = 1
            grammar.append(graFet)
            
            indepFet = [0] * numIndep
            if id in rev["depIdx"]:
                idid = 2
            else:
                idid = 1
            indepFet[idid-1] = 1
            indep.append((indepFet if features['indep'] == 1 else idid))
            iidep.append(idid-1)
            
            conFet1, conFet2 = [0] * numConstit, [0] * numConstit
            conFet1[c1-1] = 1
            conFet2[c2-1] = 1
            constit1.append((conFet1 if features['constit1'] == 1 else c1))
            constit2.append((conFet2 if features['constit2'] == 1 else c2))
            
            prFet, pprFet = [0] * numPreter, [0] * numPrepreter
            prFet[pr-1] = 1
            pprFet[ppr-1] = 1
            preter.append((prFet if features['preter'] == 1 else pr))
            prepreter.append((pprFet if features['prepreter'] == 1 else ppr))
            
            #######pos
            lpos1 = numPosition / 2 + id - rev["pos1"]
            lpos2 = numPosition / 2 + id - rev["pos2"]
            scalar_dist1, scalar_dist2 = (lpos1+1), (lpos2+1)
            vector_dist1 = [0] * numPosition
            vector_dist2 = [0] * numPosition
            vector_dist1[lpos1] = 1
            vector_dist2[lpos2] = 1
                       
            dist1.append((vector_dist1 if features['dist1'] == 1 else scalar_dist1))
            dist2.append((vector_dist2 if features['dist2'] == 1 else scalar_dist2))
            
            #######
            typeFet = [0] * numType
            #typeFet = [0] * (numType + numType*numType + 1)
            subtypeFet = [0] * numSubtype
            
            orderVec = [0] * numOrder
            
            ty1, ty2 = rev["type1"], rev["type2"]
            sty1, sty2 = rev["subtype1"], rev["subtype2"]
            
            #ty12 = str(ty1) + 'x' + str(ty2)
            #ty21 = str(ty2) + 'x' + str(ty1)
            
            if id == rev["pos1"]:
                typeFet[ty1-1] = 1
                #typeFet[ty1-1] = 1
                #typeFet[ty1+typeDic[ty12]] = 1
                subtypeFet[sty1-1] = 1
                orderVec[orderId] = 1
                
                type.append((typeFet if features['type'] == 1 else ty1))
                #type.append(typeFet)
                subtype.append((subtypeFet if features['subtype'] == 1 else sty1))
                
                order.append((orderVec if features['order'] == 1 else orderId+1))
                
            elif id == rev["pos2"]:
                typeFet[ty2-1] = 1
                #typeFet[ty2-1] = 1
                #typeFet[ty2+typeDic[ty12]] = 1
                subtypeFet[sty2-1] = 1
                orderVec[1-orderId] = 1
                
                type.append((typeFet if features['type'] == 1 else ty2))
                #type.append(typeFet)
                subtype.append((subtypeFet if features['subtype'] == 1 else sty2))
                
                order.append((orderVec if features['order'] == 1 else 2-orderId))
            
            else:
                typeFet[0] = 1
                #typeFet[0] = 1
                #typeFet[numType] = 1
                subtypeFet[0] = 1
                orderVec[2] = 1
                
                type.append((typeFet if features['type'] == 1 else 1))
                #type.append(typeFet)
                subtype.append((subtypeFet if features['subtype'] == 1 else 1))
                
                order.append((orderVec if features['order'] == 1 else 3))
                
        else:
            print 'unrecognized features '
            exit()
    
    if len(x) > mLen:
        print 'incorrect length!'
        exit()
    
    if len(x) < mLen:
        graFet = [0] * numDeprel
        indepFet = [0] * numIndep
        conFet1, conFet2 = [0] * numConstit, [0] * numConstit
        prFet, pprFet = [0] * numPreter, [0] * numPrepreter
        vector_dist1 = [0] * numPosition
        vector_dist2 = [0] * numPosition
        typeFet = [0] * numType
        #typeFet = [0] * (numType + numType*numType + 1)
        subtypeFet = [0] * numSubtype
        orderVec = [0] * numOrder
        while len(x) < mLen:
            x.append(0)
            gov.append(0)
            grammar.append(graFet)
            indep.append((indepFet if features['indep'] == 1 else 0))
            iidep.append(0)
            constit1.append((conFet1 if features['constit1'] == 1 else 0))
            constit2.append((conFet2 if features['constit2'] == 1 else 0))
            preter.append((prFet if features['preter'] == 1 else 0))
            prepreter.append((pprFet if features['prepreter'] == 1 else 0))
            dist1.append((vector_dist1 if features['dist1'] == 1 else 0))
            dist2.append((vector_dist2 if features['dist2'] == 1 else 0))
            type.append((typeFet if features['type'] == 1 else 0))
            #type.append(typeFet)
            subtype.append((subtypeFet if features['subtype'] == 1 else 0))
            order.append((orderVec if features['order'] == 1 else 0))
    
    ret = {'word' : x, 'dist1' : dist1, 'dist2' : dist2, 'type' : type, 'subtype' : subtype, 'order' : order, 'constit1' : constit1, 'constit2' : constit2, 'preter' : preter, 'prepreter' : prepreter, 'grammar' : grammar, 'gov' : gov, 'indep' : indep, 'iidep' : iidep}
    
    return ret

def dep_generatePartialDataInstance(rev, dictionaries, embeddings, features, mDepLen):
    
    idx2label = dict((k,v) for v,k in dictionaries['label'].iteritems())
    goldRel = idx2label[ (rev["y"] if 'expanded_idx2label' not in dictionaries else dictionaries['label'][dictionaries['expanded_idx2label'][rev["y"]].replace('(e1,e2)','').replace('(e2,e1)','')]) ]
    orderId = 0
    if goldRel.endswith('(e2,e1)'): orderId = 1
    if "order" in rev:
        if rev["order"] != 0 and rev["order"] != 1:
            print 'wrong order input'
            exit()
        orderId = rev["order"]
    
    numPosition = embeddings['dist11'].shape[0]-1
    numDepPosition = embeddings['dep_dist11'].shape[0]-1
    numType = embeddings['type1'].shape[0]-1
    numSubtype = embeddings['subtype1'].shape[0]-1
    numOrder = embeddings['order1'].shape[0]-1
    numConstit = embeddings['constit11'].shape[0]-1
    numPreter = embeddings['preter1'].shape[0]-1
    numPrepreter = embeddings['prepreter1'].shape[0]-1
    numDeprel = embeddings['deprel1'].shape[0]-1
    numIndep = embeddings['indep1'].shape[0]-1
    
    depx = []
    depdist1 = []
    depdist2 = []
    deptype = []
    depsubtype = []
    deporder = []
    depconstit1 = []
    depconstit2 = []
    deppreter = []
    depprepreter = []
    depgrammar = []
    depgov = []
    depindep = []
    depiidep = []
    
    #typeDic = getTypeDict(numType)
    
    id = -1
    for word, go in zip(rev["depSent"].split(), rev["dep_gov"].split()):
        id += 1
        word = ' '.join(word.split('_'))
        go = ' '.join(go.split('_'))
        c1, c2, pr, ppr = rev["dep_cons1"][id], rev["dep_cons2"][id], rev["dep_preter"][id], rev["dep_prepreter"][id]
        if word in dictionaries["word"] and go in dictionaries["word"]: #and c1 in dictionaries["constit"] and c2 in dictionaries["constit"] and pr in dictionaries["preter"] and ppr in dictionaries["prepreter"]
            depx.append(dictionaries["word"][word])
            
            depgov.append(dictionaries["word"][go])
            
            graFet = [0] * numDeprel
            for gid in rev["dep_grammar"][id]:
                graFet[gid-1] = 1
            depgrammar.append(graFet)
            
            indepFet = [0] * numIndep
            idid = 0
            depindep.append((indepFet if features['indep'] == 1 else idid))
            depiidep.append(1)
            
            conFet1, conFet2 = [0] * numConstit, [0] * numConstit
            conFet1[c1-1] = 1
            conFet2[c2-1] = 1
            depconstit1.append((conFet1 if features['constit1'] == 1 else c1))
            depconstit2.append((conFet2 if features['constit2'] == 1 else c2))
            
            prFet, pprFet = [0] * numPreter, [0] * numPrepreter
            prFet[pr-1] = 1
            pprFet[ppr-1] = 1
            deppreter.append((prFet if features['preter'] == 1 else pr))
            depprepreter.append((pprFet if features['prepreter'] == 1 else ppr))
            
            #######pos
            lpos1 = numDepPosition / 2 + id - rev["dep_pos1"]
            lpos2 = numDepPosition / 2 + id - rev["dep_pos2"]
            scalar_dist1, scalar_dist2 = (lpos1+1), (lpos2+1)
            vector_dist1 = [0] * numDepPosition
            vector_dist2 = [0] * numDepPosition
            vector_dist1[lpos1] = 1
            vector_dist2[lpos2] = 1
                       
            depdist1.append((vector_dist1 if features['dist1'] == 1 else scalar_dist1))
            depdist2.append((vector_dist2 if features['dist2'] == 1 else scalar_dist2))
            
            #######
            typeFet = [0] * numType
            subtypeFet = [0] * numSubtype
            
            orderVec = [0] * numOrder
            
            ty1, ty2 = rev["type1"], rev["type2"]
            sty1, sty2 = rev["subtype1"], rev["subtype2"]
            
            if id == rev["dep_pos1"]:
                typeFet[ty1-1] = 1
                subtypeFet[sty1-1] = 1
                orderVec[orderId] = 1
                
                deptype.append((typeFet if features['type'] == 1 else ty1))
                depsubtype.append((subtypeFet if features['subtype'] == 1 else sty1))
                
                deporder.append((orderVec if features['order'] == 1 else orderId+1))
                
            elif id == rev["dep_pos2"]:
                typeFet[ty2-1] = 1
                subtypeFet[sty2-1] = 1
                orderVec[1-orderId] = 1
                
                deptype.append((typeFet if features['type'] == 1 else ty2))
                depsubtype.append((subtypeFet if features['subtype'] == 1 else sty2))
                
                deporder.append((orderVec if features['order'] == 1 else 2-orderId))
            
            else:
                typeFet[0] = 1
                subtypeFet[0] = 1
                orderVec[2] = 1
                
                deptype.append((typeFet if features['type'] == 1 else 1))
                depsubtype.append((subtypeFet if features['subtype'] == 1 else 1))
                
                deporder.append((orderVec if features['order'] == 1 else 3))
                
        else:
            print 'unrecognized dep features '
            exit()
    
    if len(depx) > mDepLen:
        print 'incorrect length!'
        exit()
    
    if len(depx) < mDepLen:
        graFet = [0] * numDeprel
        indepFet = [0] * numIndep
        conFet1, conFet2 = [0] * numConstit, [0] * numConstit
        prFet, pprFet = [0] * numPreter, [0] * numPrepreter
        vector_dist1 = [0] * numDepPosition
        vector_dist2 = [0] * numDepPosition
        typeFet = [0] * numType
        subtypeFet = [0] * numSubtype
        orderVec = [0] * numOrder
        while len(depx) < mDepLen:
            depx.append(0)
            depgov.append(0)
            depgrammar.append(graFet)
            depindep.append((indepFet if features['indep'] == 1 else 0))
            depiidep.append(0)
            depconstit1.append((conFet1 if features['constit1'] == 1 else 0))
            depconstit2.append((conFet2 if features['constit2'] == 1 else 0))
            deppreter.append((prFet if features['preter'] == 1 else 0))
            depprepreter.append((pprFet if features['prepreter'] == 1 else 0))
            depdist1.append((vector_dist1 if features['dist1'] == 1 else 0))
            depdist2.append((vector_dist2 if features['dist2'] == 1 else 0))
            deptype.append((typeFet if features['type'] == 1 else 0))
            depsubtype.append((subtypeFet if features['subtype'] == 1 else 0))
            deporder.append((orderVec if features['order'] == 1 else 0))
    
    ret = {'word' : depx, 'dist1' : depdist1, 'dist2' : depdist2, 'type' : deptype, 'subtype' : depsubtype, 'order' : deporder, 'constit1' : depconstit1, 'constit2' : depconstit2, 'preter' : deppreter, 'prepreter' : depprepreter, 'grammar' : depgrammar, 'gov' : depgov, 'indep' : depindep, 'iidep' : depiidep}
    
    return ret

def generateDataInstance(rev, dictionaries, embeddings, features1, features2, mLen, mDepLen, seqRep, model):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    ret = {}
    for i, srp in enumerate(seqRep):
        if len(model.split('-')) < 2 and i >= 1: continue
        mle = mDepLen if srp == 'dep' else mLen
        datGen = srp + '_generatePartialDataInstance'
        dt = eval(datGen)(rev, dictionaries, embeddings, eval('features' + str(i+1)), mle)
        for kd in dt:
            ret[kd + str(i+1)] = dt[kd]
    
    return ret

def make_data(revs, dictionaries, embeddings, features1, features2, fold, seqRep, model):

    mLen = -1
    mDepLen = -1
    for rev in revs:
        if len(rev["text"].split()) > mLen:
            mLen = len(rev["text"].split())
        if len(rev["depSent"].split()) > mDepLen:
            mDepLen = len(rev["depSent"].split())
    
    print 'maximum of length in the dataset: ', mLen, mDepLen
    
    mla = [-1, -1]
    for i, sqr in enumerate(seqRep):
        mla[i] = mDepLen if sqr == 'dep' else mLen
    
    #mLen += 1

    #stext, rel1, rel2, tys, stys = {}, {}, {}, {}, {}
    res = {}
    allCorpus = ['bn_nw', 'bc0', 'bc1', 'cts', 'wl']
    partCorpus = ['bn_nw_train', 'bc0', 'bn_nw_test']
    partCorpus[0] += fold
    partCorpus[2] += fold
    for rev in revs:
        if fold == 'all':
            if rev["corpus"] not in allCorpus: continue
        else:
            if rev["corpus"] not in partCorpus: continue 
        ists = generateDataInstance(rev, dictionaries, embeddings, features1, features2, mLen, mDepLen, seqRep, model)
        
        for kk in ists:
            if (kk.endswith('1') and len(ists[kk]) != mla[0]) or (kk.endswith('2') and len(ists[kk]) != mla[1]):
                print 'wrong length!'
                exit()
         
        if rev["corpus"] not in res: res[rev["corpus"]] = defaultdict(list)
        
        for kk in ists:
            res[rev["corpus"]][kk] += [ists[kk]]
        
        res[rev["corpus"]]['label'] += [ rev["y"] if 'expanded_idx2label' not in dictionaries else dictionaries['label'][dictionaries['expanded_idx2label'][rev["y"]].replace('(e1,e2)','').replace('(e2,e1)','')] ]
        res[rev["corpus"]]['pos1'] += [rev["pos1"]]
        res[rev["corpus"]]['pos2'] += [rev["pos2"]]
        res[rev["corpus"]]['deppos1'] += [rev["dep_pos1"]]
        res[rev["corpus"]]['deppos2'] += [rev["dep_pos2"]]
        res[rev["corpus"]]['id'] += [rev["id"]]
        res[rev["corpus"]]['binaryFeatures'] += [rev["binaryFeatures"]]
        res[rev["corpus"]]['kernelPred'] += [rev["kernelPred"]]
        res[rev["corpus"]]['kernelScore'] += [rev["kernelScore"]]
    
    #for corpus in res:
    #    if not useBinaryFeatures:
    #        for prop in res[corpus]:
    #            res[corpus][prop] = np.array(res[corpus][prop], dtype="int32")
    #    else:
    #        for prop in res[corpus]:
    #            if prop == 'word':
    #                res[corpus][prop] = np.array(res[corpus][prop], dtype="int32")
    #            else:
    #                res[corpus][prop] = np.array(res[corpus][prop], dtype="float32")
    return res

def makeBinaryDictionary(dat, dats, kernelFets, cutoff=1):
    if cutoff < 0: return None, None
    print '-------creating binary feature dictionary on the training data--------'
    
    bfdCounter = defaultdict(int)
    for rev in dat['binaryFeatures']:
        for fet in rev:
            bfdCounter[fet] += 1
    print 'binary feature cutoff: ', cutoff
    bfd = {}
    for fet in bfdCounter:
        if bfdCounter[fet] >= cutoff:
            if fet not in bfd: bfd[fet] = len(bfd)
    
    if kernelFets['kernelPred'] > 0:
        print '+++using kernel prediction features ...'
        for kerpred in dat['kernelPred']:
            kepr = 'kernelPred=' + str(kerpred)
            if kepr not in bfd:
                bfd[kepr] = len(bfd)
    
    print 'size of dictionary: ', len(bfd)
    
    maxBiLen = -1
    for corpus in dats:
        for rev in dats[corpus]['binaryFeatures']:
            if len(rev) > maxBiLen: maxBiLen = len(rev)
    if kernelFets['kernelPred'] > 0: maxBiLen += 1
    print 'maximum number of binary features: ', maxBiLen
    
    return maxBiLen, bfd

def convertBinaryFeatures(dat, kernelFets, maxBiLen, bfd):
    if not bfd:
        for corpus in dat: del dat[corpus]['binaryFeatures']
        return -1
    print 'converting binary features to vectors ...'
    for corpus in dat:
        for i in range(len(dat[corpus]['word1'])):
            dat[corpus]['binaryFeatures'][i] = getBinaryVector(dat[corpus]['binaryFeatures'][i], dat[corpus]['kernelPred'][i] if kernelFets['kernelPred'] > 0 else -1, maxBiLen, bfd)
            
    return len(bfd)

def getBinaryVector(fets, kerpred, maxBiLen, dic):
    res = [-1] * (maxBiLen + 1)
    id = 0
    for fet in fets:
        if fet in dic:
            id += 1
            res[id] = dic[fet]
            
    if kerpred >= 0:
        fet = 'kernelPred=' + str(kerpred)
        if fet in dic:
            id += 1
            res[id] = dic[fet]
            
    res[0] = id
    return res

def predict(corpus, batch, reModel, idx2word, idx2label, features1, features2, model, seqRep, kernelFets):
    evaluateCorpus = {}
    extra_data_num = -1
    nsen = corpus['word1'].shape[0]
    if nsen % batch > 0:
        extra_data_num = batch - nsen % batch
        for ed in corpus:  
            extra_data = corpus[ed][:extra_data_num]
            evaluateCorpus[ed] = numpy.append(corpus[ed],extra_data,axis=0)
    else:
        for ed in corpus: 
            evaluateCorpus[ed] = corpus[ed]
        
    numBatch = evaluateCorpus['word1'].shape[0] / batch
    predictions_corpus = numpy.array([], dtype='int32')
    probs_corpus = []
    for i in range(numBatch):
        zippedCorpus = [ evaluateCorpus[ed + '1'][i*batch:(i+1)*batch] for ed in features1 if features1[ed] >= 0 ]
        zippedCorpus += [ evaluateCorpus[seqRep[0] + 'pos1'][i*batch:(i+1)*batch], evaluateCorpus[seqRep[0] + 'pos2'][i*batch:(i+1)*batch] ]
        if len(model.split('-')) == 2:
            zippedCorpus += [ evaluateCorpus[ed + '2'][i*batch:(i+1)*batch] for ed in features2 if features2[ed] >= 0 ] 
            zippedCorpus += [ evaluateCorpus[seqRep[1] + 'pos1'][i*batch:(i+1)*batch], evaluateCorpus[seqRep[1] + 'pos2'][i*batch:(i+1)*batch] ]
        
        if 'binaryFeatures' in evaluateCorpus:
            zippedCorpus += [ evaluateCorpus['binaryFeatures'][i*batch:(i+1)*batch] ]
        
        if kernelFets['kernelScore'] > 0:
            zippedCorpus += [ evaluateCorpus['kernelScore'][i*batch:(i+1)*batch] ]
        
        for prei, mmodel in enumerate(model.split('-')):
            zippedCorpus += [ evaluateCorpus['iidep' + str(prei+1)][i*batch:(i+1)*batch] ]
        
        clas, probs = reModel.classify(*zippedCorpus)
        predictions_corpus = numpy.append(predictions_corpus, clas)
        probs_corpus.append(probs)
    
    probs_corpus = numpy.concatenate(probs_corpus, axis=0)
    
    if extra_data_num > 0:
        predictions_corpus = predictions_corpus[0:-extra_data_num]
        probs_corpus = probs_corpus[0:-extra_data_num]
    
    groundtruth_corpus = corpus['label']
    
    if predictions_corpus.shape[0] != groundtruth_corpus.shape[0]:
        print 'length not matched!'
        exit()
    #words_corpus = [ map(lambda x: idx2word[x], w) for w in corpus['word']]

    #return predictions_corpus, groundtruth_corpus, words_corpus
    return predictions_corpus, probs_corpus, groundtruth_corpus

def score(predictions, groundtruths):

    zeros = numpy.zeros(predictions.shape, dtype='int')
    numPred = numpy.sum(numpy.not_equal(predictions, zeros))
    numKey = numpy.sum(numpy.not_equal(groundtruths, zeros))
    
    predictedIds = numpy.nonzero(predictions)
    preds_eval = predictions[predictedIds]
    keys_eval = groundtruths[predictedIds]
    correct = numpy.sum(numpy.equal(preds_eval, keys_eval))
    
    #numPred, numKey, correct = 0, 0, 0
    
    precision = 100.0 * correct / numPred if numPred > 0 else 0.0
    recall = 100.0 * correct / numKey
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0. else 0.0
    
    return {'p' : precision, 'r' : recall, 'f1' : f1}

def saving(corpus, predictions, probs, groundtruths, seqRep, idx2word, idx2label, idx2type, address):
    
    def determineType(type, pos1, pos2, idx2type):
        #return idx2type[0], idx2type[0]
        type1 = type[pos1]
        type2 = type[pos2]
        if type.ndim == 2:
            nty1, nty2 = -1, -1
            for i, v in enumerate(type1):
                if v == 1:
                    nty1 = i + 1
                    break
            for i, v in enumerate(type2):
                if v == 1:
                    nty2 = i + 1
                    break
            if nty1 < 0 or nty2 < 0:
                print 'negative type index'
                exit()
            type1 = nty1
            type2 = nty2
        return idx2type[type1], idx2type[type2]
    
    def generateSent(rid, sent, pos1, pos2, type1, type2, pred, gold, idx2word, idx2label):
        res = str(rid) + '\t'
        for i, w in enumerate(sent):
            if w == 0: continue
            w = idx2word[w]
            w = '_'.join(w.split())
            if i == pos1:
                res += '<ent1-type=' + type1 + '>' + w + '</ent1>' + ' '
            elif i == pos2:
                res += '<ent2-type=' + type2 + '>' + w + '</ent2>' + ' '
            else:
                res += w + ' '
        
        res = res.strip()
        res += '\t' + idx2label[gold] + '\t' + idx2label[pred] + '\t' + ('__TRUE_' if pred == gold else '__FALSE_')
        
        return res
    
    def generateProb(rid, pro, gold, idx2label):
        res = str(rid) + '\t'
        for i in range(pro.shape[0]):
            res += idx2label[i] + ':' + str(pro[i]) + ' '
        res = res.strip() + '\t' + idx2label[gold]
        return res
    
    fout = open(address, 'w')
    fprobOut = open(address + '.prob', 'w')
    
    for rid, sent, pos1, pos2, type, pred, pro, gold in zip(corpus['id'], corpus['word1'], corpus[seqRep[0] + 'pos1'], corpus[seqRep[0] + 'pos2'], corpus['type1'], predictions, probs, groundtruths):
        type1, type2 = determineType(type, pos1, pos2, idx2type)
        fout.write(generateSent(rid, sent, pos1, pos2, type1, type2, pred, gold, idx2word, idx2label) + '\n')
        fprobOut.write(generateProb(rid, pro, gold, idx2label) + '\n')
    
    fout.close()
    fprobOut.close()

def collapseTypes(td):
    res = {'Other':0}
    for t in td:
        t = t.replace('(e1,e2)','').replace('(e2,e1)','')
        if t not in res: res[t] = len(res)
    return res

def train(fold='0',
          outer=False,
          model='basic',
          seqType='-dep',
          collapsed=False,
          #encoding='ffBiDirect',
          expected_features1 = OrderedDict([('dist1', -1), ('dist2', -1), ('type', -1), ('subtype', -1), ('order', -1), ('constit1', -1), ('constit2', -1), ('preter', -1), ('prepreter', -1), ('grammar', -1), ('gov', -1), ('indep', -1)]), #values must be -1 (not used), 0 (embeddings) or 1 (binary)
          expected_features2 = OrderedDict([('dist1', -1), ('dist2', -1), ('type', -1), ('subtype', -1), ('order', -1), ('constit1', -1), ('constit2', -1), ('preter', -1), ('prepreter', -1), ('grammar', -1), ('gov', -1), ('indep', -1)]),
          sharedEmbs = OrderedDict([('word', 1), ('dist1', 0), ('dist2', 0), ('type', 0), ('subtype', 0), ('order', 0), ('constit1', 0), ('constit2', 0), ('preter', 0), ('prepreter', 0), ('grammar', 0), ('gov',0), ('indep', 0)]),
          kernelFets = OrderedDict([('kernelPred', 0), ('kernelScore', 0)]),
          withEmbs=False, # using word embeddings to initialize the network or not
          updateEmbs=True,
          optimizer='adadelta',
          lr=0.01,
          dropout=0.05,
          regularizer=0.5,
          norm_lim = -1.0,
          verbose=1,
          decay=False,
          batch=50,
          binaryCutoff=1,
          useHeadEmbedding=False,
          multilayerNN1=[1200, 600],
          multilayerNN2=[1200, 600],
          nhidden1=100,
          nhidden2=100,
          conv_winre1=20,
          conv_winre2=20,
          conv_feature_map1=100,
          conv_feature_map2=100,
          conv_win_feature_map1=[2,3,4,5],
          conv_win_feature_map2=[2,3,4,5],
          seed=3435,
          #emb_dimension=300, # dimension of word embedding
          nepochs=50,
          folder='./res'):
          
    folder = '/home/thn235/projects/minibatch/MoMatt/res/fixedExp/' + folder

    if not os.path.exists(folder): os.mkdir(folder)

    print 'loading dataset: ', dataset_path, ' ...'
    revs, embeddings, dictionaries = cPickle.load(open(dataset_path, 'rb'))
    
    if collapsed:
        dictionaries['expanded_idx2label'] = dict((k,v) for v,k in dictionaries['label'].iteritems())
        dictionaries['label'] = collapseTypes(dictionaries['label'])
    
    idx2label = dict((k,v) for v,k in dictionaries['label'].iteritems())
    idx2word  = dict((k,v) for v,k in dictionaries['word'].iteritems())
    idx2type = dict((k,v) for v,k in dictionaries['type'].iteritems())

    if not withEmbs:
        wordEmbs1 = embeddings['randomWord1']
        wordEmbs2 = embeddings['randomWord2']
    else:
        print 'using word embeddings to initialize the network ...'
        wordEmbs1 = embeddings['word1']
        wordEmbs2 = embeddings['word2']
    emb_dimension = wordEmbs1.shape[1]

    seqRep = seqType.split('-')
    
    print 'sequence types: ', seqType
    
    embs1 = {'word' : wordEmbs1,
             'gov' : wordEmbs1,
             'dist1' : embeddings['dist11'] if not seqRep[0] else embeddings['dep_dist11'],
             'dist2' : embeddings['dist21'] if not seqRep[0] else embeddings['dep_dist21'],
             #'depdist1' : embeddings['dep_dist11'],
             #'depdist2' : embeddings['dep_dist21'],
             'type' : embeddings['type1'],
             'subtype' : embeddings['subtype1'],
             'order' : embeddings['order1'],
             'constit1' : embeddings['constit11'],
             'constit2' : embeddings['constit21'],
             'preter' : embeddings['preter1'],
             'prepreter' : embeddings['prepreter1'],
             'grammar' : embeddings['deprel1'],
             'indep' : embeddings['indep1']}
    
    embs2 = {'word' : wordEmbs2,
             'gov' : wordEmbs2,
             'dist1' : embeddings['dist12'] if not seqRep[1] else embeddings['dep_dist12'],
             'dist2' : embeddings['dist22'] if not seqRep[1] else embeddings['dep_dist22'],
             #'depdist1' : embeddings['dep_dist12'],
             #'depdist2' : embeddings['dep_dist22'],
             'type' : embeddings['type2'],
             'subtype' : embeddings['subtype2'],
             'order' : embeddings['order2'],
             'constit1' : embeddings['constit12'],
             'constit2' : embeddings['constit22'],
             'preter' : embeddings['preter2'],
             'prepreter' : embeddings['prepreter2'],
             'grammar' : embeddings['deprel2'],
             'indep' : embeddings['indep2']}
             
    expected_features1['grammar'] = 1 if expected_features1['grammar'] >= 0 else -1
    expected_features2['grammar'] = 1 if expected_features2['grammar'] >= 0 else -1
    expected_features1['gov'] = 0 if expected_features1['gov'] >= 0 else -1
    expected_features2['gov'] = 0 if expected_features2['gov'] >= 0 else -1

    features1 = OrderedDict([('word', 0)])

    for ffin in expected_features1:
        features1[ffin] = expected_features1[ffin]
        if expected_features1[ffin] == 0:
            print 'using feature1: ', ffin, ' : embeddings'
        elif expected_features1[ffin] == 1:
            print 'using feature1: ', ffin, ' : binary'
    
    if len(model.split('-')) == 2:
        features2 = OrderedDict([('word', 0)])

        for ffin in expected_features2:
            features2[ffin] = expected_features2[ffin]
            if expected_features2[ffin] == 0:
                print 'using feature2: ', ffin, ' : embeddings'
            elif expected_features2[ffin] == 1:
                print 'using feature2: ', ffin, ' : binary'
    else:
        features2 = OrderedDict([('word', -1)])
        for ffin in expected_features2: features2[ffin] = -1
    
    if model == '#MultiNN':
        for ffin in features1: features1[ffin] = -1
        for ffin in features2: features2[ffin] = -1
        if useHeadEmbedding:
            features1['word'] = 0
    
    #if encoding == 'basicRep' and decoding == 'maxpooling':
    #    print '------cannot have both basicRep and maxpooling'
    #    exit()
        
    datasets = make_data(revs, dictionaries, embeddings, features1, features2, fold, seqRep, model)
    
    dimCorpus = datasets['bn_nw'] if fold == 'all' else datasets['bn_nw_train' + fold]
    
    maxBinaryFetDim, binaryFeatureDict = makeBinaryDictionary(dimCorpus, datasets, kernelFets, binaryCutoff)
    binaryFeatureDim = convertBinaryFeatures(datasets, kernelFets, maxBinaryFetDim, binaryFeatureDict)
    
    vocsize = len(idx2word)
    nclasses = len(idx2label)
    nsentences = len(dimCorpus['word1'])

    print 'vocabsize = ', vocsize, ', nclasses = ', nclasses, ', nsentences = ', nsentences, ', word embeddings dim = ', emb_dimension
    
    features_dim1 = OrderedDict([('word', emb_dimension)])
    for ffin in expected_features1:
        features_dim1[ffin] = ( len(dimCorpus[ffin + '1'][0][0]) if (features1[ffin] == 1) else embs1[ffin].shape[1] )
        
    features_dim2 = OrderedDict([('word', emb_dimension)])
    for ffin in expected_features2:
        features_dim2[ffin] = ( len(dimCorpus[ffin + '2'][0][0]) if (features2[ffin] == 1) else embs2[ffin].shape[1] )
    
    conv_winre1 = len(dimCorpus['word1'][0])
    if 'word2' in dimCorpus:
        conv_winre2 = len(dimCorpus['word2'][0])
    
    print '------- length of the instances: ', conv_winre1, conv_winre2
    #binaryFeatureDim = -1
    
    params = {'outer' : outer,
              #'encoding' : encoding,
              'model' : model,
              'nh1' : nhidden1,
              'nh2' : nhidden2,
              'nc' : nclasses,
              'ne' : vocsize,
              'batch' : batch,
              'embs1' : embs1,
              'embs2' : embs2,
              'dropout' : dropout,
              'regularizer': regularizer,
              'norm_lim' : norm_lim,
              'updateEmbs' : updateEmbs,
              'features1' : features1,
              'features_dim1' : features_dim1,
              'features2' : features2,
              'features_dim2' : features_dim2,
              'sharedEmbs': sharedEmbs,
              'kernelFets' : kernelFets,
              'optimizer' : optimizer,
              'binaryCutoff' : binaryCutoff,
              'useHeadEmbedding' : useHeadEmbedding,
              'binaryFeatureDim' : binaryFeatureDim,
              'multilayerNN1' : multilayerNN1,
              'multilayerNN2' : multilayerNN2,
              'conv_winre1' : conv_winre1,
              'conv_winre2' : conv_winre2,
              'conv_feature_map1' : conv_feature_map1,
              'conv_feature_map2' : conv_feature_map2,
              'conv_win_feature_map1' : conv_win_feature_map1,
              'conv_win_feature_map2' : conv_win_feature_map2}
    
    for corpus in datasets:
        for ed in datasets[corpus]:
            if ed == 'label' or ed == 'id' or ed == 'kernelPred':
                datasets[corpus][ed] = numpy.array(datasets[corpus][ed], dtype='int32')
            else:
                dty = 'float32' if numpy.array(datasets[corpus][ed][0]).ndim == 2 else 'int32'
                if ed.startswith('iidep') or ed == 'kernelScore': dty = 'float32'
                datasets[corpus][ed] = numpy.array(datasets[corpus][ed], dtype=dty)
    
    trainCorpus = {} #evaluatingDataset['train']
    augt = datasets['bn_nw'] if fold == 'all' else datasets['bn_nw_train' + fold]
    if nsentences % batch > 0:
        extra_data_num = batch - nsentences % batch
        for ed in augt:
            numpy.random.seed(3435)
            permuted = numpy.random.permutation(augt[ed])   
            extra_data = permuted[:extra_data_num]
            trainCorpus[ed] = numpy.append(augt[ed],extra_data,axis=0)
    else:
        for ed in augt:
            trainCorpus[ed] = augt[ed]
    
    number_batch = trainCorpus['word1'].shape[0] / batch
    
    print '... number of batches: ', number_batch
    
    # instanciate the model
    print 'building model ...'
    numpy.random.seed(seed)
    random.seed(seed)
    if model.startswith('#'):
        model = model[1:]
        params['model'] = model
        if model != 'MultiNN':
            reModel = eval('hybridModel')(params)
        else:
            reModel = eval(params['model'])(params)
    elif '_' in model:
        params['embs2']['dist1'] = embeddings['dist12'] if not seqRep[0] else embeddings['dep_dist12']
        params['embs2']['dist2'] = embeddings['dist22'] if not seqRep[0] else embeddings['dep_dist22']
        reModel = eval('ensembleModel')(params)
    else: reModel = eval('mainModel')(params) #decoding
    print 'done'
    
    if fold != 'all':
        evaluatingDataset = OrderedDict([('train', datasets['bn_nw_train' + fold]),
                                         ('valid', datasets['bc0']),
                                         ('test', datasets['bn_nw_test' + fold])
                                         ])
    else:
        evaluatingDataset = OrderedDict([('train', datasets['bn_nw']),
                                     ('valid', datasets['bc0']),
                                     ('bc', datasets['bc1']),
                                     ('cts', datasets['cts']),
                                     ('wl', datasets['wl'])
                                     ])
    
    _predictions, _probs, _groundtruth, _perfs = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict() #, _words

    # training model
    best_f1 = -numpy.inf
    clr = lr
    s = OrderedDict()
    for e in xrange(nepochs):
        s['_ce'] = e
        tic = time.time()
        #nsentences = 5
        print '-------------------training in epoch: ', e, ' -------------------------------------'
        # for i in xrange(nsentences):
        miniId = -1
        for minibatch_index in numpy.random.permutation(range(number_batch)):
            miniId += 1
            trainIn1, trainIn2 = OrderedDict(), OrderedDict()
            for ed in features1:
                if features1[ed] >= 0:
                    if (ed + '1') not in trainCorpus:
                        print 'cannot find data in train 1 for: ', ed + '1'
                        exit()
                    
                    trainIn1[ed] = trainCorpus[ed + '1'][minibatch_index*batch:(minibatch_index+1)*batch]
            for ed in features2:
                if features2[ed] >= 0:
                    if (ed + '2') not in trainCorpus:
                        print 'cannot find data in train 2 for: ', ed + '2'
                        exit()
                    
                    trainIn2[ed] = trainCorpus[ed + '2'][minibatch_index*batch:(minibatch_index+1)*batch]

            trainPos11, trainPos21 = trainCorpus[seqRep[0] + 'pos1'][minibatch_index*batch:(minibatch_index+1)*batch], trainCorpus[seqRep[0] + 'pos2'][minibatch_index*batch:(minibatch_index+1)*batch]
            trainPos12, trainPos22 = trainCorpus[seqRep[1] + 'pos1'][minibatch_index*batch:(minibatch_index+1)*batch], trainCorpus[seqRep[1] + 'pos2'][minibatch_index*batch:(minibatch_index+1)*batch]

            zippedData = [ trainIn1[ed] for ed in trainIn1 ]

            zippedData += [trainPos11, trainPos21]
            
            if len(model.split('-')) == 2:
                zippedData += [ trainIn2[ed] for ed in trainIn2 ]
                zippedData += [trainPos12, trainPos22]
            
            if 'binaryFeatures' in trainCorpus:
                zippedData += [trainCorpus['binaryFeatures'][minibatch_index*batch:(minibatch_index+1)*batch]]
                
            if kernelFets['kernelScore'] > 0:
                zippedData += [trainCorpus['kernelScore'][minibatch_index*batch:(minibatch_index+1)*batch]]
            
            for i, mmodel in enumerate(model.split('-')):
                zippedData += [ trainCorpus['iidep' + str(i+1)][minibatch_index*batch:(minibatch_index+1)*batch] ]

            zippedData += [trainCorpus['label'][minibatch_index*batch:(minibatch_index+1)*batch]]
            
            reModel.f_grad_shared(*zippedData)
            reModel.f_update_param(clr)
            
            for ed in reModel.container['embDict1']:
                reModel.container['setZero'][ed + '1'](reModel.container['zeroVecs'][ed + '1'])
            for ed in reModel.container['embDict2']:
                reModel.container['setZero'][ed + '2'](reModel.container['zeroVecs'][ed + '2'])
                
            if verbose:
                if miniId % 50 == 0:
                    print 'epoch %i >> %2.2f%%'%(e,(miniId+1)*100./number_batch),'completed in %.2f (sec) <<'%(time.time()-tic)
                    sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        print 'evaluating in epoch: ', e

        for elu in evaluatingDataset:
            _predictions[elu], _probs[elu], _groundtruth[elu] = predict(evaluatingDataset[elu], batch, reModel, idx2word, idx2label, features1, features2, model, seqRep, kernelFets)
            _perfs[elu] = score(_predictions[elu], _groundtruth[elu])# folder + '/' + elu + '.txt'

        # evaluation // compute the accuracy using conlleval.pl

        #res_train = {'f1':'Not for now', 'p':'Not for now', 'r':'Not for now'}
        perPrint(_perfs)
        
        if _perfs['valid']['f1'] > best_f1:
            #rnn.save(folder)
            best_f1 = _perfs['valid']['f1']
            print '*************NEW BEST: epoch: ', e
            if verbose:
                perPrint(_perfs, len('Current Performance')*'-')

            for elu in evaluatingDataset:
                s[elu] = _perfs[elu]
            s['_be'] = e
            
            print 'saving output ...'
            for elu in evaluatingDataset:
                saving(evaluatingDataset[elu], _predictions[elu], _probs[elu], _groundtruth[elu], seqRep, idx2word, idx2label, idx2type, folder + '/' + elu + '.best.txt')
            #subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
            #subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
        else:
            print ''
        
        # learning rate decay if no improvement in 10 epochs
        if decay and abs(s['_be']-s['_ce']) >= 10: clr *= 0.5 
        if clr < 1e-5: break

    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    print 'BEST RESULT: epoch: ', s['_be']
    perPrint(s, len('Current Performance')*'-')
    print ' with the model in ', folder

def perPrint(perfs, mess='Current Performance'):
    print '------------------------------%s-----------------------------'%mess
    for elu in perfs:
        if elu.startswith('_'):
            continue
        pri = elu + ' : ' + str(perfs[elu]['p']) + '\t' + str(perfs[elu]['r'])+ '\t' + str(perfs[elu]['f1'])
        print pri
    
    print '------------------------------------------------------------------------------'
    
if __name__ == '__main__':
    pass
