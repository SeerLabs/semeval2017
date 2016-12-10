import pickle
import os,sys
from pprint import pprint

from DataExtraction import convertCONLLFormJustExtractionSemEvalPerfile
from PhraseEval import phrasesFromTestSenJustExtractionWithIndexCoNLLBIO

def noLabelPhraseExtraction(fileinLoc):
    (sents,sents_indices) = convertCONLLFormJustExtractionSemEvalPerfile(fileinLoc)
    test_sents_pls_phrases=[x for x in [phrasesFromTestSenJustExtractionWithIndexCoNLLBIO(x,y)[-1]['phrases'] for (x,y) in zip(sents,sents_indices)] if x]
    phraseDict = {}
    for sen in test_sents_pls_phrases:
        for (phrase,pis,pie) in sen:
            pti = str(pis)+","+str(pie)
            if pti in phraseDict:
                phraseDict[pti].append((phrase,"KEYPHRASE_NOTYPES")) 
            else:
                phraseDict[pti] = [(phrase,"KEYPHRASE_NOTYPES")]
    return phraseDict       

def writeFile(fileoutLoc,phraseDict): # a key in phraseDict is the position of the phrase 120,235. Each value is a list: [(phrase,phrase type)]. 
    i = 0
    with open(fileoutLoc,"w") as f:
        for tokenloc in phraseDict:
            pis = tokenloc.split(",")[0]
            pie = tokenloc.split(",")[1] 
            for (phrase,phrasetype) in phraseDict[tokenloc]:
                f.write("T{0}\t{1} {2} {3}\t{4}\n".format(str(i),phrasetype,pis,pie,phrase))
            i+=1
        
def main():
    fileInLoc = sys.argv[1]
    PREDICTEDANNDIR = sys.argv[2]
    fileOutLoc = os.path.join(PREDICTEDANNDIR,os.path.split(fileInLoc)[-1].split("-")[0]+".ann")
    
    pdnolabel = noLabelPhraseExtraction(fileInLoc)
    writeFile(fileOutLoc,pdnolabel)   
    print "file writtem at",fileOutLoc

if __name__ == "__main__":
    main()
