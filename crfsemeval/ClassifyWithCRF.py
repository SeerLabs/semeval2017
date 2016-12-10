import pickle
import os,sys
from pprint import pprint

from sklearn_crfsuite import metrics

from DataExtraction import convertCONLLFormJustExtractionSemEvalPerfile
from FeatureExtraction import sent2labels,sent2features
from PhraseEval import phrasesFromTestSenJustExtractionWithIndex

fileinLoc = sys.argv[1]
CRFPREDICTIONRESULTSDIR = sys.argv[2]
fileoutLoc = os.path.join(CRFPREDICTIONRESULTSDIR,os.path.split(fileinLoc)[-1].split("-")[0]+"-crfprediction.txt")

crf = pickle.load(open("linear-chain-crf.model.pickle"))
(test_sents,test_sents_indices) = convertCONLLFormJustExtractionSemEvalPerfile(fileinLoc)

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

y_pred = crf.predict(X_test)

labels = list(crf.classes_)
labels.remove('O')

print labels
sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))
print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))

test_sents_pls = []  #test sentences with predicted labels
for index,testsent in enumerate(test_sents):
    sent=[]
    pls = y_pred[index]
    for ((word,pos,chunk,glabel),pl) in zip(testsent,pls):
        nt=(word,pos,chunk,pl)
        sent.append(nt)
    test_sents_pls.append(sent)

with open(fileoutLoc,"w") as f:
    for (sen,senindex) in zip(test_sents_pls,test_sents_indices):
        for ((word,pos,chunk,plabel),index) in zip(sen,senindex):
            f.write("{0} {1} {2} {3} {4}\n".format(word,pos,chunk,plabel,str(index[0])+","+str(index[1]))) 
        f.write("\n") 
            
print "classified file written at",fileoutLoc  
      

