import random
from pprint import pprint
import os
import sys
import nltk.data
import nltk.chunk
from nltk import pos_tag
from copy import deepcopy

#needed because code failes on a word like -LRB
def filterUnderscores(tokens):
     return [x for x in tokens if "-" not in x[0]]

def changeBeginTags(tokens,p): #expects a list of tokens of the form (word,tags**), for example, (word, POS-TAG, Chunk-Tag, Label)
    # and a position, 1,2 etc.. The position determine which labels to change.
    tags = set([x[p] for x in tokens])
    if len(tags) == 1 and list(tags)[0] == "O": #just one tag, O
        return tokens
    else:
        '''
        A token sequence
        [('An', 'DT', u'B-NP', 'O'), ('Experiment', 'NN', u'I-NP', 'O'), ('In', 'IN', u'B-PP', 'O'), ('Semantic', 'NNP', u'B-NP', 'B_D'), ('Tagging', 'NNP', u'I-NP', 'I_D'), ('Using', 'NNP', u'I-NP', 'O'), ('Hidden', 'NNP', u'I-NP', 'B_T'), ('Markov', 'NNP', u'I-NP', 'I_T'), ('Model', 'NNP', u'I-NP', 'I_T'), ('Tagging', 'VBG', u'B-VP', 'B_D')]
        should be changed to:
        [('An', 'DT', u'I-NP', 'O'), ('Experiment', 'NN', u'I-NP', 'O'), ('In', 'IN', u'B-PP', 'O'), ('Semantic', 'NNP', u'I-NP', 'I_D'), ('Tagging', 'NNP', u'I-NP', 'I_D'), ('Using', 'NNP', u'I-NP', 'O'), ('Hidden', 'NNP', u'I-NP', 'I_T'), ('Markov', 'NNP', u'I-NP', 'I_T'), ('Model', 'NNP', u'I-NP', 'I_T'), ('Tagging', 'VBG', u'I-VP', 'I_D')]
        following the CoNLL guideline: http://www.cnts.ua.ac.be/conll2003/ner/
        Only if two phrases of the same type immediately follow each other, the first word of the second phrase will have tag B-TYPE to show that it starts a new phrase. A word with tag O is not part of a phrase. 
        '''
        bindices = [x for (x,y) in enumerate(tokens) if y[p].startswith("B")]
        ntokens = [list(x) for x in tokens]
        for bindex in bindices:
            if bindex != 0:
                leftTokenLabel = tokens[bindex-1][p]
                thisTokenLabel = tokens[bindex][p]
                if leftTokenLabel == "O" or leftTokenLabel.split("-")[1] != thisTokenLabel.split("-")[1]:
                    ntokens[bindex][p] = thisTokenLabel.replace("B-","I-")
            else:
                ntokens[0][p] = ntokens[0][p].replace("B-","I-")            
        return filterUnderscores([tuple(x) for x in ntokens])              
   
    

def randomOneLabelExtractionfromTwoLabels(ts_with_tags_chunks):
    i  = 0
    tokens = []
    while i < len(ts_with_tags_chunks):
        t  = ts_with_tags_chunks[i] 
        i += 1
        tag = t[-2]
        if len(tag) == 4 and tag.startswith("B-"):               
            r = random.random()
            chosentag  = ""  
            if r > 0.5 :
                chosentag = tag.split("-")[1][1]
            else:
                chosentag  = tag.split("-")[1][0]
            tag = "B-"+chosentag
            tokens.append((t[0],t[1],t[2],tag,t[4]))
            while i < len(ts_with_tags_chunks) and ts_with_tags_chunks[i][-2].startswith("I-"):
                t  = ts_with_tags_chunks[i]
                tag  = "I-"+chosentag
                i += 1
                tokens.append((t[0],t[1],t[2],tag,t[4]))
        else:
            tokens.append((t[0],t[1],t[2],tag,t[4]))    
    return changeBeginTags(changeBeginTags(tokens,2),3)



def NoLabelExtractionfromTwoLabels(ts_with_tags_chunks):
    tmp = [] 
    for x in changeBeginTags(changeBeginTags(ts_with_tags_chunks,2),3):
        if x[3] == "O":
            tmp.append(x)
        else:
            tmp.append((x[0],x[1],x[2],x[3][0]+"-KP",x[4]))   
    return tmp      
      
def createFile(sens,loc):
    with open(loc,"w") as f:
        for sen in sens:
            for wts in sen: 
                f.write((" ".join([x.decode("utf-8") for x in wts])+"\n").encode("utf-8"))
            f.write("\n")
   

def main():
    loc = sys.argv[1]
    #loc = "malletformatfeaturesmultilabel/train/S0003491615000433__output.txt"

    base  = loc.split("_")[0] 
      
    chunker = nltk.data.load("chunkers/conll2000_ub.pickle")

    dT=open(loc).read().split("\n")[:-1]
    sI = [-1] + [i for i, x in enumerate(dT) if not x.strip()] + [len(dT)]
    sT1s = [x for x in [dT[sI[i]+1:sI[i+1]] for i in range(len(sI)-1)] if x]
    sensdict = {
        'nolabel-withouttokenindex':[],
        'nolabel-withtokenindex':[],
        'onelabel-withouttokenindex':[],
        'onelabel-withtokenindex':[],
        'multilabel-withouttokenindex':[],
        'multilabel-withtokenindex':[]         
    }
    for s in sT1s:
        ts = [(x.split("\t")[0],x.split("\t")[1].split(" ")[0], ",".join(x.split("\t")[1].split(" ")[1:])[:-1]) for x in s] #if the tag contains _, replace with - 
        ts_with_pos_tags = [(x[0],y[1],x[1],x[2]) for (x,y) in zip(ts,pos_tag([x[0] for x in ts])) ]
        
        ts_with_tags_chunks = [(x[0],x[1],y[2],x[2],x[3]) for (x,y) in\
        zip(ts_with_pos_tags,nltk.chunk.tree2conlltags(chunker.parse([(x[0],x[1]) for x in ts_with_pos_tags])))]
 
        multilabeltsc = changeBeginTags(changeBeginTags(ts_with_tags_chunks,2),3)

        sensdict['multilabel-withtokenindex'].append([(word,pos,chunk,label,index) for (word,pos,chunk,label,index) in multilabeltsc])
        sensdict['multilabel-withouttokenindex'].append([(word,pos,chunk,label) for (word,pos,chunk,label,index) in multilabeltsc])

        onelabeltsc = randomOneLabelExtractionfromTwoLabels(ts_with_tags_chunks) 
        sensdict['onelabel-withtokenindex'].append([(word,pos,chunk,label,index) for (word,pos,chunk,label,index) in onelabeltsc])
        sensdict['onelabel-withouttokenindex'].append([(word,pos,chunk,label) for (word,pos,chunk,label,index) in onelabeltsc])

        nolabeltsc = NoLabelExtractionfromTwoLabels(multilabeltsc)
        sensdict['nolabel-withtokenindex'].append([(word,pos,chunk,label,index) for (word,pos,chunk,label,index) in nolabeltsc])
        sensdict['nolabel-withouttokenindex'].append([(word,pos,chunk,label) for (word,pos,chunk,label,index) in nolabeltsc])
 
    for k in sensdict:
        #print k
        #print "----------------"
        #pprint(sensdict[k][0])
        #print "----------------"    
        createFile(sensdict[k],base+"-"+k+".txt") 
     
        
        
        
          

if __name__ == "__main__":
    main()    
