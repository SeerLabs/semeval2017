#from itertools import chain

#import nltk
#import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
import scipy
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from pprint import pprint
import random
import pickle

from FeatureExtraction import sent2labels,sent2features
from PhraseEval import phrasesFromTestSenJustExtraction,phrase_extraction_report
from DataExtraction import convertCONLLFormJustExtractionSemEval

def main():
    train_sents = convertCONLLFormJustExtractionSemEval("semeval-ner-train.txt")
    test_sents = convertCONLLFormJustExtractionSemEval("semeval-ner-test.txt")
    
    pprint(train_sents[0])
    pprint(test_sents[0])
        
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    crf = sklearn_crfsuite.CRF(\
    algorithm='lbfgs',\
    c1=0.1,\
    c2=0.1,\
    max_iterations=100,\
    all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    
    labels = list(crf.classes_)
    labels.remove('O')
    print labels
    pickle.dump(crf,open("linear-chain-crf.model.pickle","wb"))
    y_pred = crf.predict(X_test)
     
    # Use this if you need to do grid search on training data for parameter optimization.
    '''    
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
       all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=labels)

    # search
    rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=50,
                        scoring=f1_scorer)
    rs.fit(X_train, y_train)
    print "classification done"
    crf = rs.best_estimator_
    y_pred = crf.predict(X_test)
    '''
    sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))
    print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
    
    # Use this if you want to see how the phrase extraction works. This does NOT produce the ann files. 
    '''
    test_sents_phrases=[phrasesFromTestSenJustExtraction(x) for x in test_sents]
    pprint(test_sents_phrases[:10]) 
    print "gold standard phrases for test sentences created"
    
    test_sents_pls = []  #test sentences with predicted labels
    for index,testsent in enumerate(test_sents):
        sent=[]
        pls = y_pred[index]
        for (token,pl) in zip(testsent,pls):
            nt=(token[0],token[1],pl) 
            sent.append(nt)
        test_sents_pls.append(sent)  
    
    test_sents_pls_phrases=[phrasesFromTestSenJustExtraction(x) for x in test_sents_pls]
    print "predicted phrases for test sentences created"
  
    gps = []
    pps = []
    for sent in test_sents_phrases:
        for p in sent[-1]['phrases']:
            gps.append(p)
    for sent in test_sents_pls_phrases:
        for p in sent[-1]['phrases']:
            pps.append(p)
    
    print gps[0],pps[0]  
    pprint (phrase_extraction_report(gps,pps)) 
    '''
if __name__ == "__main__":
    main()

