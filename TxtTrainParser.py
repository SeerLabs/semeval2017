##############################################################
#Agnese Chiatti 
#Creation Date: Nov., 21 2016 
#Modified: Nov., 24 2016
#added level-1 POS tags, phrase-level tags, IOB labels
#Modified: Nov, 28 2016
#replaced regex-based tokenizer with WhitespaceTokenizer 
#to better handle symbols and units. This solved span conflicts.
#Added offsets/spans to the output files
##############################################################

import os
import io
import unicodedata
import sys
import re
#import string
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tokenize import word_tokenize,WhitespaceTokenizer
from nltk.tag import StanfordPOSTagger
#Java configuration
nltk.internals.config_java("C:/Program Files/Java/jdk1.8.0_101/bin/java.exe",options='-xmx4G')
java_path = "C:/Program Files/Java/jdk1.8.0_101/bin/java.exe"
os.environ['JAVAHOME'] = java_path


from annParser import get_kw

#same as EKE 3.0, only extended to include phrase labels and PP and VP types

def get_leaves(ptree):
    
    for subtree in ptree.subtrees(filter = lambda t: t.label()=='NP'or t.label()=='PP' or t.label()=='VP' ):
        yield (subtree.treeposition(),subtree.label(),subtree.leaves())
    
#TO-DO: folders referring to Jian's config.py after local laptop tests
stanford_bidirectional_tagger_path ="C:/semeval/stanford-postagger-full-2015-12-09/models/english-bidirectional-distsim.tagger"
stanford_postagger_jar_path ="C:/semeval/stanford-postagger-full-2015-12-09/stanford-postagger-3.6.0.jar"

#TO-DO: Folders will be changed to taos paths after debugging
train_folder= "C:/semeval/sub-train2/" #folder containing training data
output_folder= "C:/semeval/out_sub/" #folder for pre-processed output - will contain files in the right format for CRF 
dev_folder="C:/semeval/dev/" # folder containing development data

#Define stopword list
stopwords = nltk.corpus.stopwords.words('english')

#Initialize Stanford POS Tagger
stan= StanfordPOSTagger(stanford_bidirectional_tagger_path,stanford_postagger_jar_path,encoding="utf8",java_options="-mx8g")

# Nltk sentence split based on periods. 
punkt_param= PunktParameters()
#avoid i.e., e.g., Fig. or Tab. being split in separate sentences
punkt_param.abbrev_types = set([ 'i.e', 'e.g', 'fig', 'tab', 'no', 'et', 'al'])
sentence_splitter = PunktSentenceTokenizer(punkt_param)

flist_train= os.listdir(train_folder)

#For debugging or data quality check in case not all .ann files in the training set follow brat annotation format
buggyfiles=0

n_count=0
for f in flist_train:
    #Considering only txt files here
    if not str(f).endswith(".txt"):
        continue
    #print(f)
    
    
    with io.open(os.path.join(train_folder, f), mode='r', encoding="utf-8") as f_train:
    
        text= f_train.read()
        
        #Retrieve annotated features for the currently-open training file
        keywords=get_kw(train_folder,f)
        #origin_offs=get_offs(train_folder,f)  
        #print(keywords)
        
        
        #Split each sentence on a separate line 
        toktext=sentence_splitter.tokenize(text)
        s_spans=sentence_splitter.span_tokenize(text)
        sentence_spans=[]
        for ss in s_spans:
            sss=[]
            start=ss[0]
            end=ss[1]
            
            sss.append(start)
            sss.append(end)
            sentence_spans.append(sss)
                  
        #print(keywords)
        #Create output files with a similar name as the input files
        outputfile = f.split(".")[0] + "__output.txt"
    
    with io.open(os.path.join(output_folder,outputfile),'w', encoding="utf-8") as outf:
        
        
        #each tokenized word within each sentence is placed on a separate line in the output file
        z=0
        for s in toktext:
            #outf.write("---BOS---\n") #Beginning of sentence
            
            
            #Remove punctuation
            '''table = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
            sout=s.translate(table)'''
            
            tokenizer = WhitespaceTokenizer()
            tokenwords = WhitespaceTokenizer().tokenize(s)
            t_spans = tokenizer.span_tokenize(s)
            t_spans_l=[]
            
            #tokenwords2=word_tokenize(s)
            word_spans_=[]
            
            
            for w in t_spans:
                t_spans_l.append(w)
            #print(t_spans_l)
            
            for t in tokenwords:
                
                index=tokenwords.index(t)
                w=t_spans_l[index]
                try:
                    #previous word
                    t_prev=tokenwords[index-1]
                    #for w in t_spans:
                    wss=[]
                    length=w[1]-w[0]
                    if z==0:  #first sentence
                        
                            start=w[0]
                            end=w[1]
                            end=start +length
                    else:
                        
                        if w[0]==0:
                            #if t_prev!=',':
                            start=sentence_spans[z][0]
                            end= start+w[1]
                                
                            #else:
                                #start=sentence_spans[z][0]-1
                                #end= start+w[1]
                            newstart=end+1    
                        else:
                            start=newstart
                            end=start+length
                            newstart=end+1
                        
                    wss.append(start)
                    wss.append(end)
                    word_spans_.append(wss)
                except:
                    continue
            #print(tokenwords)
            #print(tokenwords2)
            #print(word_spans_)
            #print(keywords)
            #print(origin_offs)
           
            
            #For each cleaned sentence: chunk phrases and parse tree
            #same module of Hung-Hsuan Chen & Jian Wu code (EKE 3.0)
            #---NP rationale taken from Su Nam Kim paper
            #But extended to simple PPs and VPs---
            grammar = r"""
            NP: {<NN.*|JJ>*<NN.*>} # NP
            PP: {<IN> <NP>}      # PP -> P NP
            VP: {<V.*> <NP|PP>*}  # VP -> V (NP|PP)*
            """
            chunker = nltk.RegexpParser(grammar)
            # Stanford POS tagging
            postag=stan.tag(tokenwords)
            
            tree= chunker.parse(postag)
            partree = nltk.ParentedTree.convert(tree)
            
            positions=()
            phrases=[]
            temp=-1
            
            #generator wrapping data for leaves
            leaves=get_leaves(partree)
            #copy-generator for iterations and next() call
            cleaves=get_leaves(partree)
            try:
                next(cleaves)
            
            #Going from tree leaves to one phrase per word (only phrases of type NP, PP, VP)
                for l in leaves: 
                    #print(l)
                    current=l[0][0]
                    #print(l)
                    try:
                        m=next(cleaves)
                        #print(m)
                        temp=m[0][0]
                        #print(temp)
                                   
                        try:
                            check=l[0][1]
                            current2=l[0][1]
                            temp2=m[0][1]
                            if current==temp and current2==temp2:
                                try:
                                    check=l[0][2]
                                    for n in l[2]:
                                        phrases.append(l[1])
                                        #print(n)
                                        phrases.append(n)
                                    
                                except:
                                    phrases.append(l[1])
                                    phrases.append(l[2][0])
                            else:        
                                for n in l[2]:
                                    phrases.append(l[1])
                                    #print(n)
                                    phrases.append(n)
                        except:
                            #Target phrase has been already reached, no further levels
                            if current != temp:
                                
                                for n in l[2]:
                                    phrases.append(l[1])
                                    #print(n)
                                    phrases.append(n)
                                
                            else:
                                phrases.append(l[1])
                                phrases.append(l[2][0])
                    #last phrase reached
                    except:
                        for n in l[2]:
                            phrases.append(l[1])
                            #print(n)
                            phrases.append(n)
                #print(phrases)
            except:
                print(outf)
                #print(sout)
                print(tokenwords)
                print(partree)    
            phraseout=[]
            ptagout=[]
            
            #List for only words that have phrase-level tags
            for j in range(1,len(phrases), 2):
                try:
                    phraseout.append(phrases[j][0])
                    
                except:
                    break
            
            #List for only phrase-level tags at same position of the previous list
            for j in range(0,len(phrases), 2):
                try:
                    ptagout.append(phrases[j])
                    
                except:
                    break
            
            #print(ptagout)
            
            #Remove stopwords from set before outputting
            '''for w in tokenwords:
                
                if (w.lower() in stopwords):
                    #print(w)
                    index=tokenwords.index(w)
                    tokenwords.remove(w)
                    del word_spans_[index]
                    #del word_spans[index]
                    #del word_spans_np[index]
            '''
                        
            #print(tokenwords)
            #Retag after stopword and punctuation removal
            #postag=stan.tag(tokenwords)
            #print(postag)
            #print(tokenwords)
            #print(word_spans_)
            #print(keywords)
            
            
            wordcount=1
            i=0
            for w in tokenwords:
                
                #Write word and relative POS tag
                tags='\t'.join(postag[i])
                outf.write(tags)
                
                               
                #Write relative phrase-level tag if found in subset NP, PP, VP or label as "Other"
                if w in phraseout:
                    #get relative tag in the associated tag list for same index
                    ptag=ptagout[phraseout.index(w)]
                    outf.write("\t"+ptag)
                else:
                    outf.write("\tOTHER")
                
                
                #add ground-truth labels
                label="O"
                
                for c in keywords:
                    if w==c[0] and (str(word_spans_[i][0])==c[2]) and (str(word_spans_[i][1])==c[3]):
                        
                        label= c[1]
                       
                        
                        
                    else:
                        continue 
                       
                outf.write("\t"+label)
                
                outf.write(" "+str(word_spans_[i][0]))
                outf.write(" "+str(word_spans_[i][1]))
                
                word_length=len(w)
               
                #Compute word placement from the beginning of a sentence
                outf.write("\t["+ str(wordcount)+"]")
                outf.write("\t["+ str(word_length)+"]")
                
                #a simple check for Upper/Lower case in the initial letter
                letters=list(w)
                
                if letters[0].isupper():
                    outf.write("\tCAPITALIZED")
                else:
                    outf.write("\tLOWERCASE")
                    
                if not(w.isalpha()):
                    outf.write("\tContainSymbol")
                else:
                    outf.write("\tallAlpha")
                
                outf.write("\n")
                wordcount +=1 
                i+=1
            #outf.write("---EOS---\n") #End of sentence
            outf.write("\n") #End of sentence
            z+=1
            
    outf.close()
    f_train.close()
    n_count +=1
    print("No. of processed files: "+str(n_count))
print("COMPLETE --- ")
#print("COMPLETE --- Total no. of broken ann. files:"+str(buggyfiles))
#print(n_count)
    
