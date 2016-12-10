We are looking at three problems (each problem takes a document as input):

1. **Extraction**: Extract the phrases from a sentence that can be considered as a scholarly concept.

2. **Single label classification**: Extract and classify the phrases from a sentence in one of the three classes mentioned above.

3. **Multi label classification**: Extract and classify the phrases from a sentence one or multiple of the classes mentioned above.


We have three folders: `dev`, `train` and `test`. Each of these folders have three sub-folders: `nolabel`, `onelabel` and `multilabel`. Each folder corresponds to one of the problems mentioned above. Each folder contains the **CoNLL style annotation** of files. Each folder has two sub-folders: 1. `withtokenindex`, and `withouttokenindex`. The `withouttokenindex` files are to be used during training. The `withtokenindex` files are to be used during prediction, so we can predict phrases with their token indices, as required by the SemEval competition. A file contains both text lines and blank lines. A text line in a `withouttokenindex` file has four columns separated by `space`: 1. A word from a sentence, 2. POS-Tag, 3. Chunk-Tag and 4. Class label. A line in a `withtokenindex` file has all that information + the token index. A blank line denotes the end of a sentence.

The tagging is done according to the [CoNLL NER task guideline](http://www.cnts.ua.ac.be/conll2003/ner/):


The CoNLL-2003 shared task data files contain four columns separated by a single space. Each word has been put on a separate line and there is an empty line after each sentence. The first item on each line is a word, the second a part-of-speech (POS) tag, the third a syntactic chunk tag and the fourth the named entity tag. The chunk tags and the named entity tags have the format I-TYPE which means that the word is inside a phrase of type TYPE. Only if two phrases of the same type immediately follow each other, the first word of the second phrase will have tag B-TYPE to show that it starts a new phrase. A word with tag O is not part of a phrase. Here is an example:

```
   U.N.         NNP  I-NP  I-ORG
   official     NN   I-NP  O
   Ekeus        NNP  I-NP  I-PER
   heads        VBZ  I-VP  O
   for          IN   I-PP  O
   Baghdad      NNP  I-NP  I-LOC
   .            .    O     O
```

The goal is to predict the class labels for the tokens (words).

