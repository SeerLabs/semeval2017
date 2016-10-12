# taking code from:
#Terry Peng <pengtaoo@gmail.com>
#Mikhail Korobov <kmike84@gmail.com>
#Bundled CRFSuite C/C++ library is by Naoaki Okazaki & contributors.
#http://www.chokkan.org/software/crfsuite/manual.html

import pycrfsuite

#The training data provided for SemEval have to be adapted to this module

########## PARSING TO BE ADDED##################################
#for each annotated file in the training folder
# mark the first keyphrase in the sequence with "__BOS__"
# the first label refers to the current token and has 
#to be of type B-/I-/O- and then the related Brill POS tag
# then the relative position of each word in the window is indicated
# then the Brill tag for each position is indicate
# mark the first keyphrase in the sequence with "__EOS__"
# at the end of each file add an empty line to separate blocks

# Brill tag set
#http://www.ling.gu.se/~lager/mogul/brill-tagger/penn.html

###EXAMPLE####
#T1 Process 5 14  oxidation
#T2 Material  69  84  Ti-based alloys
#T3 Material  186 192 alloys

###has to become:######################
#B-NP w[0]=oxidation pos[0]=NNP  __BOS__
#I-NP w[-1]=Ti-based  w[0]=alloys pos[-1]=JJ  pos[0]=NNPS 
#B-NP w[0]=alloys pos[0]=NNPS __EOS__

# Both use tab command to separate values placed on the same line
#R (relationship) lines in the SemEval .ann files should be skipped here

#just an example with the first file of the training set
inputfile = open('CRFtrain/S0010938X1500195X.ann')
outputfile = open('CRFtrain/S0010938X1500195X.txt', 'w')

for line in inputfile:
	target_text=inputfile.readlines()
	outputfile.writelines(target_text)

inputfile=close()
outputfile=close()



############PART OF THE TUTORIAL CODE################################################
#Feature extraction
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]    
  
# Train the model
%%time
trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)
    
# Here L-BFGS training algorithm (it is default) with Elastic Net (L1 + L2) regularization is used. 
  
  trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})
 



