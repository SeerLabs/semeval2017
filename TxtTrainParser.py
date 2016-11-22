
import os
import io
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tokenize import word_tokenize



#folders will be changed to taos paths after debugging

train_folder= "C:/semeval/train2/" #folder containing training data
output_folder= "C:/semeval/out/" #folder for pre-processed output - will contain files in the right format for CRF 
dev_folder="C:/semeval/dev/" # folder containing development data


punkt_param= PunktParameters()

#avoid i.e., e.g., Fig. or Tab. being splitted in separate sentences
punkt_param.abbrev_types = set([ 'i.e', 'e.g', 'fig', 'tab', 'no'])
sentence_splitter = PunktSentenceTokenizer(punkt_param)

flist_train= os.listdir(train_folder)

for f in flist_train:
    #considering only txt files right now
    if not str(f).endswith(".txt"):
        continue
    
    with io.open(os.path.join(train_folder, f), 'r', encoding="utf-8") as f_train:
    
        text= f_train.read()
        
        #split each sentence on a separate line 
        
        toktext=sentence_splitter.tokenize(text)
            
        outputfile = f.split(".")[0] + "__output.txt"
    
    #creates output files with a similar name as the input files
    with io.open(os.path.join(output_folder,outputfile),'w', encoding="utf-8") as outf:
        #print("test")
        
        #each tokenized word within each sentence is placed on a separate line in the output file
        for s in toktext:
            outf.write("---BOS---\n") #Beginning of sentence
            
            tokenwords=word_tokenize(s)
            wordcount=1
        
            for w in tokenwords:
                outf.write(w)
                
                word_length=len(w)
                #outf.write(s) 
                
                #Compute word placement from the beginning of a sentence
                outf.write("\t["+ str(wordcount)+"]")
                outf.write("\t["+ str(word_length)+"]")
                
                #a simple check for Upper/Lower case in the initial letter
                letters=list(w)
                
                if letters[0].isupper():
                    outf.write("\tCAPITALIZED")
                else:
                    outf.write("\tLOWERCASE")
                    
                if (w=='.')|(w==','):
                    outf.write("\tallPunct")
                elif (w=='[')|(w==']')|(w=='(')|(w==')'):
                    outf.write("\tallParenth")
                    
                elif not((w=='.')|(w==','))and not((w=='[')|(w==']')|(w=='(')|(w==')'))and not(w.isalpha()):
                    outf.write("\tContainSymbol")
                else:
                    outf.write("\tallAlpha")
                
                outf.write("\n")
                wordcount +=1 
            outf.write("---EOS---\n") #End of sentence
        
        outf.close()
  

