##############################
#Agnese Chiatti Nov., 21 2016#
##############################

import os
import io


#Returns a List containing annotated phrases for a given txt
#(Assumes filename notation is the same and only formats change)



def get_ann_phrases(target_folder, twintxt):

    twinann= twintxt.split(".")[0]+".ann"

    phrases=[]
        
    with io.open(os.path.join(target_folder, twinann), 'r', errors="ignore", encoding="utf-8") as f_targ:
        
        lines= f_targ.read().splitlines()
        
        #lcount=0
        for l in lines:
            
            line=l.split("\t")
            
            # handle lines formatted for synonym relations (i.e. no keyphrases in third place)
            try:
                phrases.append(line[2].split())        
            except:
                phrases.append("")
            
    f_targ.close()
    return phrases        


#Returns a List containing annotated labels/categories
def get_ann_labels(target_folder, twintxt):

    twinann= twintxt.split(".")[0]+".ann"

    labels=[]
    labelsout=[]
        
    with io.open(os.path.join(target_folder, twinann), 'r', errors="ignore", encoding="utf-8") as f_targ:
        
        lines= f_targ.read().splitlines()
        
        #lcount=0
        for l in lines:
            
            line=l.split("\t")
            
            try:
                labels.append(line[1]) 
            
            except:
                print("Invalid ann file!"+twinann)
                labels=[]
                return labels
                break
 
               
        for lab in labels:
            
            tokens=lab.split()
            labelsout.append(tokens[0])
            
        #for labb in labelsout:
            #print(labb)
            
    f_targ.close()
    return labelsout

#Returns a List containing original offsets
def get_offs(target_folder, twintxt):

    twinann= twintxt.split(".")[0]+".ann"

    labels=[]
    offs=[]
        
    with io.open(os.path.join(target_folder, twinann), 'r', errors="ignore", encoding="utf-8") as f_targ:
        
        lines= f_targ.read().splitlines()
        
        #lcount=0
        for l in lines:
            
            line=l.split("\t")
            
            try:
                labels.append(line[1]) 
            
            except:
                #print("Invalid ann file!")
                labels=[]
                return labels
                break
 
            
               
        for lab in labels:
            
            tokens=lab.split()
            offs.append((tokens[1]+" "+tokens[2]).split())
            
        #for o in offs:
            #print(o)
            
    f_targ.close()
    return offs


def get_kw(train_folder, f):
    keyphrases= get_ann_phrases(train_folder, f)
    #labels= get_ann_labels(train_folder, f)
    offsets= get_offs(train_folder, f)
    #print(keyphrases)
    #print(offsets)
    keywords=[]        
    for c in keyphrases:
        j=keyphrases.index(c)
        
        #retrieve same position in offsets list 
        rangeof=offsets[j]
        #print(rangeof)
        try:
            #length=float(range[1])-float(range[0])
            newstart=rangeof[0]
            #print(length)
            i=0
            for d in c:
                kw=[]
                kw.append(d)
                if i==0:
                    kw.append('B-KP')
                    start=rangeof[0]
                    end=str(int(rangeof[0])+len(d))
                    #current position + blank space
                    newstart=int(rangeof[0])+len(d)+1 
                    i+=1
                else:
                    kw.append('I-KP')
                    start=str(newstart)
                    end=str(newstart+len(d))
                    #if newstart+len(d) == float(range[1]):
                        #end of phrase
                    newstart=newstart+len(d)+1   
                    i+=1
                kw.append(start)
                kw.append(end)
                keywords.append(kw)
        except:
            #relational line reached - no keyphrase line
            continue
                
    return keywords
