import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import random

#thien's version
extra = 5
maximumDistance = 100000000
testingDataCorpus = ['bn_nw', 'bc0', 'bc1', 'cts', 'wl' ]

def build_data_cv(fullOrNot, data_folder):
    """
    Loads data.
    """
    labelDict = {'NONE':0}
    typeDict = {'Other':1}
    subTypeDict = {'Other':1}
    constitDict = {'Other':1}
    preterDict = {'Other':1}
    prepreterDict = {'Other':1}
    depRelDict = {'Other':1}
    vocab = defaultdict(float)
    revs = []
    corpus = ''
    corpusCountIns = defaultdict(int)
    maxLength = -1
    maxDist = -1
    maxDepLength = -1
    distCounter = defaultdict(int)
    tooLong = 0
    ignoredDueTo1 = 0
    
    for ed in data_folder:
        if fullOrNot == 'full' and 'full' not in ed: continue
        if fullOrNot != 'full' and 'full' in ed: continue 
        print 'Loading file: ', ed
        if ed.endswith('.full'):
            cpu = ed[0:ed.rfind('.')]
        else:
            cpu = ed
        with open(data_folder[ed], 'r') as f:
            for line in f:
                line = line.strip()
                relId, features, detectorLabel, classLabel, type1, subtype1, cons1, pos1, type2, subtype2, cons2, pos2, preter, prepreter, grs, gov, sentence, depSent, depRel, depIdx, kernelPred, kernelScore = parseLine(cpu, line)
                
                depPos1, depPos2, depCons1, depCons2, depPreter, depPrepreter, depGrs, depGov = retrieveDepFeatures(cons1, cons2, preter, prepreter, grs, gov, depIdx)
                
                if classLabel == None:
                    ignoredDueTo1 += 1
                    continue
                
                if not labelDict.has_key(classLabel):
                    labelDict[classLabel] = len(labelDict)
                    print 'label: ', classLabel, ' --> id = ', labelDict[classLabel]
                classId = labelDict[classLabel]
                
                if abs(pos2 - pos1) > maximumDistance:
                    tooLong += 1
                    continue
                
                #sentence, pos1, pos2, cons1, cons2, preter, prepreter, grs, gov, depIdx = fitSentenceToWindow(sentence, pos1, pos2, cons1, cons2, preter, prepreter, grs, gov, depIdx, extra=extra)
                
                if len(sentence.split()) > maxLength:
                    maxLength = len(sentence.split())
            
                if abs(pos2 - pos1) > maxDist:
                    maxDist = abs(pos2-pos1)
                    
                if len(depSent.split()) > maxDepLength:
                    maxDepLength = len(depSent.split())
                    
                distCounter[abs(pos2-pos1)] += 1
            
                corpusCountIns[ed] += 1
            
                words = set(sentence.split())
                for word in words:
                    word = ' '.join(word.split('_'))
                    vocab[word] += 1
                    
                depWords = set(depSent.split())
                for word in depWords:
                    word = ' '.join(word.split('_'))
                    vocab[word] += 1
                
                govWords = set(gov.split())
                for word in govWords:
                    word = ' '.join(word.split('_'))
                    vocab[word] += 1
                
                depGovWords = set(depGov.split())
                for word in depGovWords:
                    word = ' '.join(word.split('_'))
                    vocab[word] += 1
            
                #######Constituent
                for cons in set(cons1.split()):
                    if not constitDict.has_key(cons):
                        constitDict[cons] = len(constitDict) + 1
                        print 'constitType: ', cons, ' --> id = ', constitDict[cons]
                        
                for cons in set(cons2.split()):
                    if not constitDict.has_key(cons):
                        constitDict[cons] = len(constitDict) + 1
                        print 'constitType: ', cons, ' --> id = ', constitDict[cons]
                
                ####################Constituent for Dep
                        
                for cons in set(depCons1.split()):
                    if not constitDict.has_key(cons):
                        constitDict[cons] = len(constitDict) + 1
                        print 'constitType: ', cons, ' --> id = ', constitDict[cons]
                        
                for cons in set(depCons2.split()):
                    if not constitDict.has_key(cons):
                        constitDict[cons] = len(constitDict) + 1
                        print 'constitType: ', cons, ' --> id = ', constitDict[cons]
                        
                cons1 = [ constitDict[cons] for cons in cons1.split() ]
                cons2 = [ constitDict[cons] for cons in cons2.split() ]
                depCons1 = [ constitDict[cons] for cons in depCons1.split() ]
                depCons2 = [ constitDict[cons] for cons in depCons2.split() ] 
                
                #######Preterminal
                
                for pre in set(preter.split()):
                    if not preterDict.has_key(pre):
                        preterDict[pre] = len(preterDict) + 1
                        print 'preterType: ', pre, ' --> id = ', preterDict[pre]
                
                for pre in set(depPreter.split()):
                    if not preterDict.has_key(pre):
                        preterDict[pre] = len(preterDict) + 1
                        print 'preterType: ', pre, ' --> id = ', preterDict[pre]
                        
                preter = [ preterDict[pre] for pre in preter.split() ]
                depPreter = [ preterDict[pre] for pre in depPreter.split() ]
                
                #######Prepreterminal
                
                for prepre in set(prepreter.split()):
                    if not prepreterDict.has_key(prepre):
                        prepreterDict[prepre] = len(prepreterDict) + 1
                        print 'prepreterType: ', prepre, ' --> id = ', prepreterDict[prepre]
                
                for prepre in set(depPrepreter.split()):
                    if not prepreterDict.has_key(prepre):
                        prepreterDict[prepre] = len(prepreterDict) + 1
                        print 'prepreterType: ', prepre, ' --> id = ', prepreterDict[prepre]
                        
                prepreter = [ prepreterDict[prepre] for prepre in prepreter.split() ]
                depPrepreter = [ prepreterDict[prepre] for prepre in depPrepreter.split() ]
                
                ########Grammatical relations
                for gs in grs:
                    for g in gs:
                        if not depRelDict.has_key(g):
                            depRelDict[g] = len(depRelDict) + 1
                            print 'depRelType: ', g, ' --> id = ', depRelDict[g]
                
                for gs in depGrs:
                    for g in gs:
                        if not depRelDict.has_key(g):
                            depRelDict[g] = len(depRelDict) + 1
                            print 'depRelType: ', g, ' --> id = ', depRelDict[g]
                            
                nngs = []
                for gs in grs:
                    nng = [ depRelDict[g] for g in gs ]
                    nngs += [nng]
                grs = nngs
                
                nngs = []
                for gs in depGrs:
                    nng = [ depRelDict[g] for g in gs ]
                    nngs += [nng]
                depGrs = nngs
                        
                #for dere in set(depRel.split()):
                #    dere = dere.replace('\'', '')
                #    if not depRelDict.has_key(dere):
                #        depRelDict[dere] = len(depRelDict) + 1
                #        print 'depRelType: ', dere, ' --> id = ', depRelDict[dere] 
                
                #depRel = [ depRelDict[dere] for dere in depRel.split() ]                   
                
                if not typeDict.has_key(type1):
            	    typeDict[type1] = len(typeDict)+1
            	    print 'entityType: ', type1, ' --> id = ', typeDict[type1]
                type1 = typeDict[type1]
                if not typeDict.has_key(type2):
            	    typeDict[type2] = len(typeDict)+1
            	    print 'entityType: ', type2, ' --> id = ', typeDict[type2]
                type2 = typeDict[type2]
            
                if not subTypeDict.has_key(subtype1):
            	    subTypeDict[subtype1] = len(subTypeDict)+1
            	    print 'entitySubType: ', subtype1, ' --> id = ', subTypeDict[subtype1]
                subtype1 = subTypeDict[subtype1]
                if not subTypeDict.has_key(subtype2):
            	    subTypeDict[subtype2] = len(subTypeDict)+1
            	    print 'entitySubType: ', subtype2, ' --> id = ', subTypeDict[subtype2]
                subtype2 = subTypeDict[subtype2]
                
                if cpu not in testingDataCorpus: continue
            
                datum = {"id": relId,
                         "y":classId, 
                         
                         "text": sentence,
                         "pos1": pos1,
                         "pos2": pos2,
                         "cons1": cons1,
                         "cons2": cons2,
                         "preter": preter,
                         "prepreter": prepreter,
                         "grammar": grs,
                         "gov": gov,
                         "depIdx": depIdx,
                         
                         "depSent": depSent,
                         "dep_pos1": depPos1,
                         "dep_pos2": depPos2,
                         "dep_cons1": depCons1,
                         "dep_cons2": depCons2,
                         "dep_preter": depPreter,
                         "dep_prepreter": depPrepreter,
                         "dep_grammar": depGrs,
                         "dep_gov": depGov,
                         
                         "kernelPred": kernelPred,
                         "kernelScore": kernelScore,
                         
                         "depRel": depRel,
                         
                         "type1": type1,
                         "subtype1": subtype1,
                         "type2": type2,
                         "subtype2": subtype2,
                         
                         "binaryFeatures": features,
                         "corpus": cpu}
                revs.append(datum)
    
    print 'instances in corpus'
    for corpus in corpusCountIns:
    	print corpus, ' : ', corpusCountIns[corpus]
    print '---------------'
    print 'distance distribution'
    for di in distCounter:
    	print di, ' : ', distCounter[di]
    print '---------------'
    print "maximum length of sentences: ", maxLength
    print "maximum distances of instances: ", maxDist
    print "maximum length of depSent: ", maxDepLength
    print "number of too long: ", tooLong
    print "number of ignored due 1: ", ignoredDueTo1
    
    for rev in revs:
        #kernelType = rev["kernelPred"]
        #if kernelType in labelDict:
        #    rev["kernelPred"] = labelDict[kernelType]
        #else: rev["kernelPred"] = -1
        
        kers = [-1.0] * len(labelDict)
        #for kernelType in rev["kernelScore"]:
        #    if kernelType not in labelDict:
        #        print 'cannot find label in dict: ', kernelType
        #        exit()
        #    kers[labelDict[kernelType]] = rev["kernelScore"][kernelType]
        rev["kernelScore"] = kers
        
        rev["kernelPred"] = -1
        
        if rev["kernelPred"] == -1 and rev["kernelScore"][0] != -1.0:
            print 'There are mismatches for kernel'
            exit()
    
    return maxLength, maxDist, maxDepLength, revs, vocab, labelDict, typeDict, subTypeDict, constitDict, preterDict, prepreterDict, depRelDict

def parseLine(cpu, line):
    els = line.split('\t')
    if cpu in testingDataCorpus:
        relId = els[0]
        classLabel = els[1]
        ans = els[5].split()
        constit = els[6].split()
        preter = els[7]
        prepreter = els[8]
        dep = els[9]
        gra = els[10]
        gov = els[11].strip()
        
        features = els[4]
        
        #triples: els[12]
    else:
        relId = els[0]
        classLabel = els[1]
        ans = els[4].split()
        constit = els[5].split()
        preter = els[6]
        prepreter = els[7]
        dep = els[8]
        gra = els[9]
        gov = els[10].strip()
        
        features = els[2]
    
    if classLabel.startswith('PHYS'): classLabel = 'PHYS'
    if classLabel.startswith('PER-SOC'): classLabel = 'PER-SOC'
        
    if len(els) >= 14:
        kernel = els[13]
        kernelPred = kernel[0:kernel.find(' ')]
        kernelScore = parseScore( kernel[ (kernel.find(' ')+1): ] )
    else:
        kernelPred = '__UNDEFINED__'
        kernelScore = {}
    
    if len(ans) != len(constit) and constit[0] != '__NULL__':
        print 'annotation and constituent lengths no matched!'
        exit()
        
    if len(ans) != len(preter.split()) and preter != '__NULL__':
        print 'annotation and preterminal lengths no matched!'
        exit()
    
    qid = features[0:features.find(' ')]
    features = features[(features.find(' ')+1):].split()
    
    if classLabel == 'NONE':
        detectorLabel = 'O'
    else:
        detectorLabel = '1'
    
    type1 = els[3].split('@')[0]
    type2 = els[3].split('@')[1]
    
    subtype1 = 'O'
    subtype2 = 'O'
    
    sentence = ''
    e1arrs, e2arrs = [], []
    id = -1
    for an in ans:
        id += 1
        sep = an.rfind('/')
        if sep < 0:
            print 'cannot find the / separator: ', an
            exit()
        w = an[0:sep]
        ty = an[(sep+1):]
        sty = 'O'
        ears = None
        if ty != 'O':
            if ty.rfind('#') < 0:
                print 'cannot find the # separator: ', an
                exit()
            ears = int(ty[(ty.rfind('#')+1):])
            if ears != 1 and ears != 2:
                print 'not entity indicator: ', an
                exit()
            type = ty[0:ty.rfind('#')]
            if type.rfind('.') >= 0:
                sty = type[(type.rfind('.')+1):]
                type = type[0:type.rfind('.')]
            if type.find('-') >= 0:
                type = type[(type.find('-')+1):]
            if ears == 1:
                if type != type1:
                    print 'type1 not matched: ', an, type1
                    exit()
                e1arrs += [id]
                subtype1 = sty
            elif ears == 2:
                if type != type2:
                    print 'type2 not matched: ', an, type2
                    exit()
                e2arrs += [id]
                subtype2 = sty
        sentence += w + ' '
    
    if len(e1arrs) == 0 and len(e2arrs) == 0:
        print 'cannot find entity indexes: ', line
        exit()
    
    if len(e1arrs) == 0: e1arrs = e2arrs
    elif len(e2arrs) == 0: e2arrs = e1arrs
    #if len(e1arrs) == 0 or len(e2arrs) == 0:
    #    return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    pos1 = e1arrs[len(e1arrs)-1]
    pos2 = e2arrs[len(e2arrs)-1]
    
    #if type1 == 'O' and type2 == 'O' and pos1 == 0 and pos2 == id: pos2 = 1 if id > 0 else 0
    
    sentence = sentence.strip()
    
    #if classLabel == 'NONE': classLabel = 'Other'
    if type1 == 'O': type1 = 'Other'
    if type2 == 'O': type2 = 'Other'
    if subtype1 == 'O': subtype1 = 'Other'
    if subtype2 == 'O': subtype2 = 'Other'
    
    if constit[0] != '__NULL__':
        cons1, cons2 = '', ''
        for con in constit:
            if len(con.split('--')) != 2:
                print 'constit not have two parameters: ', con
                exit()
            cons1 += con.split('--')[0] + ' '
            cons2 += con.split('--')[1] + ' '
    else:
        cons1 = 'Other ' * len(sentence.split())
        cons2 = 'Other ' * len(sentence.split())
    
    cons1 = cons1.strip()
    cons2 = cons2.strip()
    
    if len(sentence.split()) != len(cons1.split()) or len(sentence.split()) != len(cons2.split()):
        print 'Length of sentence and constit not matched!!!'
        exit()
    
    if preter == '__NULL__':
        preter = 'Other ' * len(sentence.split())
    
    preter = preter.strip()
    
    if prepreter == '__NULL__':
        prepreter = 'Other ' * len(sentence.split())
    
    prepreter = prepreter.strip()
    
    depSent, depRel, depIdx = analyzeDepPath(dep)
    if depSent == None:
        #depSent = sentence.split()[pos1] + ' ' + sentence.split()[pos2]
        #depIdx = [pos1, pos2]
        if pos1 == pos2:
            depSent = sentence.split()[pos1]
            depIdx = [pos1]
            depRel = 'Other'
        else:
            depSent = sentence.split()[pos1] + ' ' + sentence.split()[pos2]
            depIdx = [pos1, pos2]
            depRel = 'Other'
    
    if depIdx[0] != pos1 or depIdx[len(depIdx)-1] != pos2:
        print 'Wrong depIdx assumption with pos1 or pos2: ', depIdx[0], depIdx[1], pos1, pos2
        exit()
        
    grs = []
    for ga in gra.split():
        g = ga.split('@')
        grs += [g]
    
    if len(grs) != len(sentence.split()):
        print 'Length of sentence and grammatical relations not matched!!!'
        exit()
    
    return relId, features, detectorLabel, classLabel, type1, subtype1, cons1, pos1, type2, subtype2, cons2, pos2, preter, prepreter, grs, gov, sentence, depSent, depRel, depIdx, kernelPred, kernelScore

def parseScore(scoStr):
    res = {}
    els = scoStr.split()
    for e in els:
        k = e[0:e.rfind(':')]
        score = float(e[(e.rfind(':')+1):])
        res[k] = score
    return res

def retrieveDepFeatures(cons1, cons2, preter, prepreter, grs, gov, depIdx):
    
    depCons1, depCons2, depPreter, depPrepreter, depGov = '', '', '', '', ''
    depGrs = []
    
    cc1, cc2, pre, prepre, go = cons1.split(), cons2.split(), preter.split(), prepreter.split(), gov.split()
    for id in depIdx:
        c1, c2, p, pp, g = cc1[id], cc2[id], pre[id], prepre[id], go[id]
        depCons1 += c1 + ' '
        depCons2 += c2 + ' '
        depPreter += p + ' '
        depPrepreter += pp + ' '
        depGrs += [grs[id]]
        depGov += g + ' '
    
    return 0, len(depIdx)-1, depCons1.strip(), depCons2.strip(), depPreter.strip(), depPrepreter.strip(), depGrs, depGov.strip()

def analyzeDepPath(dep):
    if not dep or dep == '__NULL__': return None, None, None
    dep = dep.split()
    sent, rel = [], []
    idxs = []
    for i, el in enumerate(dep):
        if i % 2 == 0:
            offset = el.rfind('-')
            if offset < 0:
                print 'cannot find - separator in depPath: ', dep
                exit()
            word = el[0:offset]
            id = int(el[(offset+1):])-1
            sent.append(word)
            idxs.append(id)
        else:
            rel.append(el)
    if idxs[len(idxs)-1] < idxs[0]:
        sent = sent[::-1]
        rel = rel[::-1]
        idxs = idxs[::-1]
    return ' '.join(sent).strip(), ' '.join(rel).strip(), idxs

def fitSentenceToWindow(sentence, pos1, pos2, cons1, cons2, preter, prepreter, grs, gov, depIdx, extra=5):
    #window = abs(pos2 - pos1) + 1 + 2 * extra
    #lower = ((pos1 + pos2 + window) / 2) - window + 1
    lower = (pos1 - extra) if pos1 < pos2 else (pos2 - extra)
    
    if lower >= 0:
        npos1 = pos1 - lower
        npos2 = pos2 - lower
        under = lower
        nDepIdx = [ ni - lower for ni in depIdx ]
    else:
        npos1 = pos1
        npos2 = pos2
        under = 0
        nDepIdx = depIdx
    
    above = (pos1 + extra) if pos1 > pos2 else (pos2 + extra)
    words = sentence.split()
    cc1 = cons1.split()
    cc2 = cons2.split()
    pf = preter.split()
    ppf = prepreter.split()
    ggo = gov.split()
    
    if above > (len(words)-1):
        above = len(words)-1
    
    nsent, ncons1, ncons2, npreter, nprepreter, ngov = '', '', '', '', '', ''
    ngrs = []
    for i, w in enumerate(words):
        if i < under or i > above: continue
        nsent += w + ' '
        ncons1 += cc1[i] + ' '
        ncons2 += cc2[i] + ' '
        npreter += pf[i] + ' '
        nprepreter += ppf[i] + ' '
        ngrs += [grs[i]]
        ngov += ggo[i] + ' '
    
    lenSent = len(nsent.strip()) - 1
    nDepIdx = [ ni for ni in nDepIdx if ni <= lenSent ]
    
    return nsent.strip(), npos1, npos2, ncons1.strip(), ncons2.strip(), npreter.strip(), nprepreter.strip(), ngrs, ngov.strip(), nDepIdx

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W1 = np.zeros(shape=(vocab_size+1, k))
    W2 = np.zeros(shape=(vocab_size+1, k))
    W1[0] = np.zeros(k)
    W2[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W1[i] = word_vecs[word]
        W2[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W1, W2, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    dim = 0
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
               dim = word_vecs[word].shape[0]
            else:
                f.read(binary_len)
    print 'dim: ', dim
    return dim, word_vecs
    
def load_text_vec(fname, vocab):
    word_vecs = {}
    count = 0
    dim = 0
    with open(fname, 'r') as f:
        for line in f:
            count += 1
            line = line.strip()
            if count == 1:
                if len(line.split()) < 10:
                    dim = int(line.split()[1])
                    print 'dim: ', dim
                    continue
                else:
                    dim = len(line.split()) - 1
                    print 'dim: ', dim
            word = line.split()[0]
            emStr = line[(line.find(' ')+1):]
            if word in vocab:
                word_vecs[word] = np.fromstring(emStr, dtype='float32', sep=' ')
                if word_vecs[word].shape[0] != dim:
                    print 'mismatch dimensions: ', dim, word_vecs[word].shape[0]
                    exit()
    print 'loaded ', len(word_vecs), ' words in word embeddings'
    return dim, word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

if __name__=="__main__":
    np.random.seed(8989)
    random.seed(8989)
    fullOrNot = sys.argv[1]
    embType = sys.argv[2]
    w2v_file = sys.argv[3]
    data_file = sys.argv[4]
    data_folder = {}
    dataCorpus = ['bn_nw', 'bn_nw.full', 'bc0', 'bc0.full', 'bc1', 'bc1.full', 'cts', 'cts.full', 'wl', 'wl.full', 'bn_nw_train0', 'bn_nw_train0.full', 'bn_nw_train1', 'bn_nw_train1.full', 'bn_nw_train2', 'bn_nw_train2.full', 'bn_nw_train3', 'bn_nw_train3.full', 'bn_nw_train4', 'bn_nw_train4.full', 'bn_nw_test0', 'bn_nw_test0.full', 'bn_nw_test1', 'bn_nw_test1.full', 'bn_nw_test2', 'bn_nw_test2.full', 'bn_nw_test3', 'bn_nw_test3.full', 'bn_nw_test4', 'bn_nw_test4.full' ]
    
    #dataCorpus = ['bn_nw.full', 'bc0.full', 'bc1.full', 'cts.full', 'wl.full' ]
    
    for ed in dataCorpus:
        data_folder[ed] = data_file + ed + '.txt'
    print "loading data...\n"
    maxLength, maxDist, maxDepLength, revs, vocab, labelDict, typeDict, subTypeDict, constitDict, preterDict, prepreterDict, depRelDict = build_data_cv(fullOrNot, data_folder)
    #print "max distance between entities: " + str(maxDist)
    print "data loaded!"
    print "vocab size: " + str(len(vocab))
    print "loading word embeddings...",
    dimEmb = 300
    if embType == 'word2vec':
    	dimEmb, w2v = load_bin_vec(w2v_file, vocab)
    else:
    	dimEmb, w2v = load_text_vec(w2v_file, vocab)
    print "word embeddings loaded!"
    print "num words already in word embeddings: " + str(len(w2v))
    add_unknown_words(w2v, vocab, 1, dimEmb)
    W1, _W1, word_idx_map = get_W(w2v, dimEmb)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab, 1, dimEmb)
    W2, _W2, _ = get_W(rand_vecs, dimEmb)
    #mmax_l = 20
    
    dictionaries = {}
    #dictionaries['vocab'] = vocab
    dictionaries['word'] = word_idx_map
    dictionaries['label'] = labelDict
    dictionaries['type'] = typeDict
    dictionaries['subtype'] = subTypeDict
    dictionaries['constit'] = constitDict
    dictionaries['preter'] = preterDict
    dictionaries['prepreter'] = prepreterDict
    dictionaries['deprel'] = depRelDict
    
    #toReturn = [ vocab, word_idx_map, labelDict, typeDict, subTypeDict, constitDict, preterDict, prepreterDict, depRelDict ]
    
    embeddings = {}
    
    dist_size = 2*maxLength - 1
    dist_dim = 50
    D1 = np.random.uniform(-0.25,0.25,(dist_size+1,dist_dim))
    D2 = np.random.uniform(-0.25,0.25,(dist_size+1,dist_dim))
    D1[0] = np.zeros(dist_dim)
    D2[0] = np.zeros(dist_dim)
    
    dist_dep_size = 2*maxDepLength - 1
    dist_dep_dim = 50
    depD1 = np.random.uniform(-0.25,0.25,(dist_dep_size+1,dist_dep_dim))
    depD2 = np.random.uniform(-0.25,0.25,(dist_dep_size+1,dist_dep_dim))
    depD1[0] = np.zeros(dist_dep_dim)
    depD2[0] = np.zeros(dist_dep_dim)
    
    type_dim = 50
    T = np.random.uniform(-0.25,0.25,(len(typeDict)+1,type_dim))
    T[0] = np.zeros(type_dim)
    
    subtype_dim = 50
    ST = np.random.uniform(-0.25,0.25,(len(subTypeDict)+1,subtype_dim))
    ST[0] = np.zeros(subtype_dim)
    
    constit_dim = 50
    CONSTIT1 = np.random.uniform(-0.25,0.25,(len(constitDict)+1,constit_dim))
    CONSTIT2 = np.random.uniform(-0.25,0.25,(len(constitDict)+1,constit_dim))
    CONSTIT1[0] = np.zeros(constit_dim)
    CONSTIT2[0] = np.zeros(constit_dim)
    
    preter_dim = 50
    PRETER = np.random.uniform(-0.25,0.25,(len(preterDict)+1,preter_dim))
    PRETER[0] = np.zeros(preter_dim)
    
    prepreter_dim = 50
    PREPRETER = np.random.uniform(-0.25,0.25,(len(prepreterDict)+1,prepreter_dim))
    PREPRETER[0] = np.zeros(prepreter_dim)
    
    depRel_dim = 50
    DEPREL = np.random.uniform(-0.25,0.25,(len(depRelDict)+1,depRel_dim))
    DEPREL[0] = np.zeros(depRel_dim)
    
    order_dim = 50
    OR = np.random.uniform(-0.25,0.25,(3+1,order_dim))
    OR[0] = np.zeros(order_dim)
    
    indep_dim = 50
    INDEP = np.random.uniform(-0.25,0.25,(2+1,indep_dim))
    INDEP[0] = np.zeros(indep_dim)
    
    #######
    _dist_size = 2*maxLength - 1
    _dist_dim = 50
    _D1 = np.random.uniform(-0.25,0.25,(_dist_size+1,_dist_dim))
    _D2 = np.random.uniform(-0.25,0.25,(_dist_size+1,_dist_dim))
    _D1[0] = np.zeros(_dist_dim)
    _D2[0] = np.zeros(_dist_dim)
    
    _dist_dep_size = 2*maxDepLength - 1
    _dist_dep_dim = 50
    _depD1 = np.random.uniform(-0.25,0.25,(_dist_dep_size+1,_dist_dep_dim))
    _depD2 = np.random.uniform(-0.25,0.25,(_dist_dep_size+1,_dist_dep_dim))
    _depD1[0] = np.zeros(_dist_dep_dim)
    _depD2[0] = np.zeros(_dist_dep_dim)
    
    _type_dim = 50
    _T = np.random.uniform(-0.25,0.25,(len(typeDict)+1,_type_dim))
    _T[0] = np.zeros(_type_dim)
    
    _subtype_dim = 50
    _ST = np.random.uniform(-0.25,0.25,(len(subTypeDict)+1,_subtype_dim))
    _ST[0] = np.zeros(_subtype_dim)
    
    _constit_dim = 50
    _CONSTIT1 = np.random.uniform(-0.25,0.25,(len(constitDict)+1,_constit_dim))
    _CONSTIT2 = np.random.uniform(-0.25,0.25,(len(constitDict)+1,_constit_dim))
    _CONSTIT1[0] = np.zeros(_constit_dim)
    _CONSTIT2[0] = np.zeros(_constit_dim)
    
    _preter_dim = 50
    _PRETER = np.random.uniform(-0.25,0.25,(len(preterDict)+1,_preter_dim))
    _PRETER[0] = np.zeros(_preter_dim)
    
    _prepreter_dim = 50
    _PREPRETER = np.random.uniform(-0.25,0.25,(len(prepreterDict)+1,_prepreter_dim))
    _PREPRETER[0] = np.zeros(_prepreter_dim)
    
    _depRel_dim = 50
    _DEPREL = np.random.uniform(-0.25,0.25,(len(depRelDict)+1,_depRel_dim))
    _DEPREL[0] = np.zeros(_depRel_dim)
    
    _order_dim = 50
    _OR = np.random.uniform(-0.25,0.25,(3+1,_order_dim))
    _OR[0] = np.zeros(_order_dim)
    
    _indep_dim = 50
    _INDEP = np.random.uniform(-0.25,0.25,(2+1,_indep_dim))
    _INDEP[0] = np.zeros(_indep_dim)
    #######
    
    embeddings['word1'] = W1
    embeddings['randomWord1'] = W2
    embeddings['dist11'] = D1
    embeddings['dist21'] = D2
    embeddings['dep_dist11'] = depD1
    embeddings['dep_dist21'] = depD2
    embeddings['type1'] = T
    embeddings['subtype1'] = ST
    embeddings['constit11'] = CONSTIT1
    embeddings['constit21'] = CONSTIT2
    embeddings['preter1'] = PRETER
    embeddings['prepreter1'] = PREPRETER
    embeddings['deprel1'] = DEPREL
    embeddings['order1'] = OR
    embeddings['indep1'] = INDEP
    
    embeddings['word2'] = _W1
    embeddings['randomWord2'] = _W2
    embeddings['dist12'] = _D1
    embeddings['dist22'] = _D2
    embeddings['dep_dist12'] = _depD1
    embeddings['dep_dist22'] = _depD2
    embeddings['type2'] = _T
    embeddings['subtype2'] = _ST
    embeddings['constit12'] = _CONSTIT1
    embeddings['constit22'] = _CONSTIT2
    embeddings['preter2'] = _PRETER
    embeddings['prepreter2'] = _PREPRETER
    embeddings['deprel2'] = _DEPREL
    embeddings['order2'] = _OR
    embeddings['indep2'] = _INDEP
    
    #toReturn = [ vocab, word_idx_map, labelDict, typeDict, subTypeDict, constitDict, preterDict, prepreterDict, depRelDict ]
    #toReturn += [ W1, W2, D1, D2, depD1, depD2, T, ST, CONSTIT1, CONSTIT2, PRETER, PREPRETER, DEPREL, OR, INDEP ]
    #toReturn += [ _W1, _W2, _D1, _D2, _depD1, _depD2, _T, _ST, _CONSTIT1, _CONSTIT2, _PRETER, _PREPRETER, _DEPREL, _OR, _INDEP ]
    
    for di in dictionaries:
        print 'size of ', di, ': ', len(dictionaries[di])
    
    if fullOrNot == 'full':
        embType += '.full'
    cPickle.dump([revs, embeddings, dictionaries], open(embType + "_rnnRE_Plank.pkl", "wb"))
    #cPickle.dump([revs] + toReturn, open(embType + "_rnnRE_Plank.pkl", "wb"))
    print "dataset created!"   
