## Dependencies

1. `sklearn-crfsuite`: https://pypi.python.org/pypi/sklearn-crfsuite
2. nltk
3. scipy, numpy, scikit-learn. 

## Description

1. Traindata from SemEval was converted to basic CoNLL format by Agnese. The original files are at `data/semevaltrainingdata`. Proper CoNLL formatted files are at `data/conllformat/`. Follow the README file in that format.

2. CRFNER.py trains a linear chain CRF model and outputs the model as a pickle file (`linear-chain-crf.model.pickle`). You can do a hyper parameter optimization on the training data.

3. `DataExtraction.py` and `FeatureExtraction.py` contains the code to prepare the data and extract features. Both are used by CRFNER.py. Note the pos tags are extracted during the data extraction step. 

4. `ClassifyWithCRF.py` uses the trained model to predict the token classes and output a predicted text file with the predicted labels. An example input to the code is a file in `data/conllformat/nolabel/test/withouttokenindex/S2212671612001291-nolabel-withouttokenindex.txt`, e.g., `python ClassifyWithCRF.py data/conllformat/nolabel/test/withouttokenindex/S2212671612001291-nolabel-withouttokenindex.txt <BaseDirForPredictedText>` where `<BaseDirForPredictedText>` is the base directory for storing txt files with predicted labels. Output for this code would be at `<BaseDirForPredictedText>/S2212671612001291-crfprediction.txt`.

5. `ConvertCoNLLtoANN.py` will take a CoNLL format file as input and output corresponding ANN file. For example, `python ConvertCoNLLtoAnn.py results/crfprediction/nolabel/predictedtxts/S2212671612001291-crfprediction.txt  <BaseDirForPredictedANN>` will produce the required ANN file at the location `BaseDirForPredictedANN`.

6. The whole pipeline for just extraction (or **no label classification**) is at `SemEvalNoLabel.sh`.

##TODO

1. Change the training data, test data and see if there is any significant effect.

2. New features.

3. Error analysis.   
