from nltk import pos_tag

def convertCONLLFormJustExtraction(loc):
    dT=open(loc).read().split("\n")[:-2]
    sI = [-1] + [i for i, x in enumerate(dT) if not x.strip()] + [len(dT)]
    sT1s = [dT[sI[i]+1:sI[i+1]] for i in range(len(sI)-1)]
    sTs = []
    for s in sT1s:
        xp = [x.split(" ") for x in s] 
        ts = [(x[0],x[1],x[2],x[3]) for x in xp]
        sTs.append(ts)
    return sTs

def convertCONLLFormJustExtractionSemEval(loc):
    dT=open(loc).read().split("\n")[:-2]
    sI = [-1] + [i for i, x in enumerate(dT) if not x.strip()] + [len(dT)]
    sT1s = [dT[sI[i]+1:sI[i+1]] for i in range(len(sI)-1)]
    sTs = []
    for s in sT1s:
        ts= [(x.split("\t")[0],x.split("\t")[1]) for x in s]
        tokens = [(x[0],y[1],x[1]) for (x,y) in zip(ts,pos_tag([x[0] for x in ts])) ]
        tokens = [(x,y,z[0]) for (x,y,z) in tokens]
        sTs.append(tokens)
    return sTs

def convertCONLLFormJustExtractionSemEvalPerfile(loc):
    #assumes we have a file with token indices in the form `x,y` and the end of each line.
    dT=open(loc).read().split("\n")[:-2]
    sI = [-1] + [i for i, x in enumerate(dT) if not x.strip()] + [len(dT)]
    sT1s = [dT[sI[i]+1:sI[i+1]] for i in range(len(sI)-1)]
    sTs = []
    sTIs = []
    for s in sT1s:
        xp = [x.split(" ") for x in s]
        tokens = [(word,pos,chunk,label) for (word,pos,chunk,label,ti) in xp]
        tokenindices = [(int(ti.split(",")[0]),int(ti.split(",")[1])) for (word,pos,chunk,label,ti) in xp]
        sTs.append(tokens)
        sTIs.append(tokenindices)
    return (sTs,sTIs)
