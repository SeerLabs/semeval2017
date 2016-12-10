from __future__ import division
import numpy as np
import copy
import random
from pprint import pprint

def getPhraseTokens(bDs,iDs,senLength):
    bpTokens=[]
    for i in range(len(bDs)):
        start = bDs[i][0]
        end = senLength
        if i < len(bDs)-1:
            end = bDs[i+1][0]  
        phrase =  ' '.join([x[1] for x in [bDs[i]]+[x for x in iDs if x[0] > start and x[0] < end]])
        bpTokens.append(phrase)
        
    return bpTokens

def getPhraseTokensWithIndex(bDs,iDs,senLength):
    bpTokens=[]
    for i in range(len(bDs)):
        start = bDs[i][0]
        end = senLength
        if i < len(bDs)-1:
            end = bDs[i+1][0]  
        phrase =  ' '.join([x[1] for x in [bDs[i]]+[x for x in iDs if x[0] > start and x[0] < end]])
        phraseStart = min([x[-2] for x in [bDs[i]]+[x for x in iDs if x[0] > start and x[0] < end]])
        phraseEnd =  max([x[-1] for x in [bDs[i]]+[x for x in iDs if x[0] > start and x[0] < end]])
        bpTokens.append((phrase,phraseStart,phraseEnd))
        
    return bpTokens
            
def phrasesFromTestSen(sen):
    bDs = sorted([(i,x[0]) for (i,x) in enumerate(sen) if x[-1] == "B_D"], key = lambda x:x[0])
    iDs = sorted([(i,x[0]) for (i,x) in enumerate(sen) if x[-1] == "I_D"], key = lambda x:x[0])
    bFs = sorted([(i,x[0]) for (i,x) in enumerate(sen) if x[-1] == "B_F"], key = lambda x:x[0])
    iFs = sorted([(i,x[0]) for (i,x) in enumerate(sen) if x[-1] == "I_F"], key = lambda x:x[0])
    bTs = sorted([(i,x[0]) for (i,x) in enumerate(sen) if x[-1] == "B_T"], key = lambda x:x[0])
    iTs = sorted([(i,x[0]) for (i,x) in enumerate(sen) if x[-1] == "I_T"], key = lambda x:x[0])
    
    tSen = copy.deepcopy(sen)
    tSen.append(
    {
    'dps': getPhraseTokens(bDs,iDs,len(sen)),
    'tps': getPhraseTokens(bTs,iTs,len(sen)),
    'fps': getPhraseTokens(bFs,iFs,len(sen))
    }
    )
    return tSen

def phrasesFromTestSenJustExtractionWithIndex(sen,tokenIndices):
    sen = [(word,pos,chunk,label,wsi,wei) for ((word,pos,chunk,label),(wsi,wei)) in zip(sen,tokenIndices)]
    bS = sorted([(i,w,wsi,wei) for (i,(w,p,c,l,wsi,wei)) in enumerate(sen) if l == "B-KP"], key = lambda x:x[0])
    iS = sorted([(i,w,wsi,wei) for (i,(w,p,c,l,wsi,wei)) in enumerate(sen) if l == "I-KP"], key = lambda x:x[0])

    tSen = copy.deepcopy(sen)
    tSen.append(
    {
    'phrases': getPhraseTokensWithIndex(bS,iS,len(sen)),
    }
    )
    return tSen

def phrasesFromTestSenJustExtractionWithIndexCoNLLBIO(sen,tokenIndices):
    tsen = [list(t) for t in sen]
    if tsen[0][-1].startswith("I-"):
        tsen[0][-1] = tsen[0][-1].replace("I-","B-")
    for i in range(1,len(tsen)):
        if tsen[i][-1].startswith("I-") and tsen[i-1][-1]=="O":
            tsen[i][-1] = tsen[i][-1].replace("I-","B-")
    tsen = [tuple(t) for t in tsen]
    return phrasesFromTestSenJustExtractionWithIndex(tsen,tokenIndices)


def phrasesFromTestSenJustExtraction(sen):
    bS = sorted([(i,x[0]) for (i,x) in enumerate(sen) if x[-1] == "B-KP"], key = lambda x:x[0])
    iS = sorted([(i,x[0]) for (i,x) in enumerate(sen) if x[-1] == "I-KP"], key = lambda x:x[0])
   
    tSen = copy.deepcopy(sen)
    tSen.append(
    {
    'phrases': getPhraseTokens(bS,iS,len(sen)),
    }
    )
    return tSen

def phrasesFromTestSenJustExtractionCoNLLBIO(sen):
    tsen = [list(t) for t in sen]
    if tsen[0][-1].startswith("I-"):       
        tsen[0][-1] = tsen[0][-1].replace("I-","B-")
    for i in range(1,len(tsen)):
        if tsen[i][-1].startswith("I-") and tsen[i-1][-1]=="O":
            tsen[i][-1] = tsen[i][-1].replace("I-","B-")
    tsen = [tuple(t) for t in tsen]
    return phrasesFromTestSenJustExtraction(tsen)  


def matchAbs(i1,i2):
    return i1.lower() == i2.lower()

def matchIn(iP,iG):
    return iG.lower() in iP.lower()

def calc_result(gl,pl,style):
    #gL = [a,b,c], pL = [d,c,a,e]. match = 2, precision = 2/4, recall = 2/3
    if not gl and not pl:
        return None
    if (gl and not pl) or (pl and not gl):
        return (0.0,0.0)        
    nmatch = 0
    for i1 in pl:
        for i2 in gl:
            if match(i1,i2):
                nmatch += 1
    return (nmatch/len(pl),nmatch/len(gl))     
    
def phrase_classification_report(sentListG,sentListP):
    slGphrases=[x[-1] for x in sentListG]
    slPphrases=[x[-1] for x in sentListP]
    if len(slGphrases) != len(slPphrases):
        return None
    else:
        tRs = [x for x in [calc_result(sG['tps'],sP['tps'],"technique") for (sG,sP) in zip(slGphrases,slPphrases)] if x is not None]
        fRs = [x for x in [calc_result(sG['fps'],sP['fps'],"focus") for (sG,sP) in zip(slGphrases,slPphrases)] if x is not None]
        dRs = [x for x in [calc_result(sG['dps'],sP['dps'],"domain") for (sG,sP) in zip(slGphrases,slPphrases)] if x is not None]
        results = {
            'technique': {
                'precision':round(np.mean([x[0] for x in tRs]),2),
                'recall':round(np.mean([x[1] for x in tRs]),2)  
                },
            'focus': {
                'precision':round(np.mean([x[0] for x in fRs]),2),
                'recall':round(np.mean([x[1] for x in fRs]),2)  
                },
            'domain': {
                'precision':round(np.mean([x[0] for x in dRs]),2),
                'recall':round(np.mean([x[1] for x in dRs]),2)  
                }    
        }
        return results  

def phrase_extraction_report(gl,pl):
    gl = list(set([x.lower() for x in gl]))
    pl = list(set([x.lower() for x in pl])) 
    print "phrase extraction results: gold standard: ",len(gl),"phrases, predicted: ",len(pl),"phrases" 
    nmatch = 0
    matches = []
    for i1 in pl:
        for i2 in gl:
            if matchAbs(i1,i2): 
                matches.append({'predicted':i1,'gold':i2})
                nmatch += 1
                break
    random.shuffle(matches)
    print nmatch,"phrases matched"
    #pprint(matches[:10])
    return {'precision':nmatch/len(pl),'recall':nmatch/len(gl)}

