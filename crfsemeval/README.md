## Dependencies

1. `sklearn-crfsuite`: https://pypi.python.org/pypi/sklearn-crfsuite
2. nltk
3. scipy, numpy, scikit-learn. 

## Description

1. Traindata from SemEval was converted to mallet format by Agnese. The original files are at `semevaltrainingdata`. Mallet formatted files are at `malletformatfeatures`.
To convert data from semeval to BIO-style tags for classification, the `semeval_to_BIO.py` script can be used (example outputs can be found under malletformatfeaturesBIO, which can also be given as input to `convert.py`, similarly to what explained in the next step).  

2. The data was split randomly into train and test data. The file lists are in `training.txt` and `test.txt`. Also, for each original file, three fields were taken, the word (token), the tag (`B_KP`, `I_KP`) and the index. The files are in `malletformat/training` and `malletformat/test`. `convert.py` was used.

3. These files were combined using `awk` [`awk -F"\t" {print $1"\t"$2} malletformat/test/* > semeval-ner-test.txt`] to produce the training data (`semeval-ner-train.txt`) and test data (`semeval-ner-test.txt`). Note these files contain only two columns (tab separated): the token and the tag.

4. CRFNER.py trains a linear chain CRF model and outputs the model as a pickle file (`linear-chain-crf.model.pickle`). You can do a hyper parameter optimization on the training data.

5. `DataExtraction.py` and `FeatureExtraction.py` contains the code to prepare the data and extract features. Both are used by CRFNER.py. Note the pos tags are extracted during the data extraction step. 

6. `ClassifyCRFtoANN.py` uses the trained model to predict the token classes and output the predicted ann file. An example input to the code is a file in `malletformat/test`, e.g., `python ClassifyCRFtoANN.py malletformat/test/S0166218X14003011-mallet.txt`. Output is `malletformat/test/S0166218X14003011-predicted.ann`.

##TODO

1. Change the training data, test data and see if there is any significant effect.

2. New features.

3. Error analysis.   
