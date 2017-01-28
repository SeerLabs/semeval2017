#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Hung-Hsuan Chen <hhchen@ncu.edu.tw>
# Creation Date : 10-06-2016
# Jian Wu <fanchyna@gmail.com>
# Modified: 2016-10-12: output to SemEval 2017 format
# Jian Wu <fanchyna@gmail.com>
# Modified: 2016-10-25: use Stanford CoreNLP Tag parser instead of NLTK 
# Jian Wu <fanchyna@gmail.com>
# Modified:2017-01-28: change POS tags of "[" to "X" rather than "NN"


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import collections
import logging
import config
import nltk
from nltk.tag import StanfordPOSTagger
nltk.internals.config_java(options='-xmx4G')

logger = logging.getLogger("gen_keyphrase_core_bounds")

def leaves(ptree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
#    for subtree in tree.subtrees(filter = lambda t: t.node=='NP'):
    for subtree in ptree.subtrees(filter = lambda t: t.label()=='NP'):
        yield (subtree.treeposition(),subtree.leaves())


def normalize(word, lemmatizer):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    #word = stemmer.stem_word(word)
    word = lemmatizer.lemmatize(word)
    return word


def acceptable_word(word, stopwords):
    """Checks conditions for acceptable word: length, stopword."""
    # Hung-Hsuan used "2 <=", Jian Wu changes to "1 <="
    accepted = bool(1 <= len(word) <= 40 and word.lower() not in stopwords)
    return accepted


# pos_map is a dict keys are leaf_treeposition, values are token spans
def get_terms(ptree, lemmatizer, stopwords, pos_map):
    logger = logging.getLogger("get_terms")
    term_span = []
    for treepos,leaf in leaves(ptree):
        # from positions of the subtree, infer boundaries of the keyphrase, do not lemmentize (keep original form)
        #term = [ normalize(w, lemmatizer) for w,t in leaf if acceptable_word(w, stopwords) ]
        term = [ w for w,t in leaf if acceptable_word(w, stopwords) ]
        span_kph = [1e10,0]
        for pos_w,span_w in pos_map:
            if pos_w.startswith(str(treepos)[0:-1]):
                span_kph[0] = min(span_w[0],span_kph[0])
                span_kph[1] = max(span_w[1],span_kph[1])
        # Hung-Hsuan used "1 < len(term) <= 5", Jian Wu changes to "len(term) >0"
        if len(term) > 0: 
            term_span.append( (" ".join(term),span_kph) )
        else:
            pass
    return term_span

# ptree is a ParentedTree
# span_toks is a generator containing spans of tokens
#@return is a list of tuples containing ("leaf_treeposition",token span)
def generate_pos_map(ptree,span_toks):
    logger = logging.getLogger("generate_pos_map")
    pos_map = []
    for i,span in enumerate(span_toks):
        logger.debug(ptree.leaf_treeposition(i))
        pos_map.append([str(ptree.leaf_treeposition(i)),span])
    return pos_map

def gen_keyphrases(text):
    # Used when tokenizing words
    sentence_re = r'''(?x)        # set flag to allow verbose regexps
        (?:[A-Z])(?:\.[A-Z])+\.?    # abbreviations, e.g. U.S.A.
        | \w+(?:-\w+)*            # words with optional internal hyphens
        | \$?\d+(?:\.\d+)?%?        # currency and percentages, e.g. $12.40, 82%
        | \.\.\.                # ellipsis
        | [][.,;"'?():-_`]        # these are separate tokens
    '''

    lemmatizer = nltk.WordNetLemmatizer()

    #Taken from Su Nam Kim Paper...
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>}    # Nouns and Adjectives, terminated with Nouns

        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}    # Above, connected with in/of/etc...
    """
    chunker = nltk.RegexpParser(grammar)

    tokenizer = nltk.RegexpTokenizer(sentence_re)
    toks = tokenizer.tokenize(text)
    span_toks = tokenizer.span_tokenize(text)
    logger.debug(toks)
    logger.debug("tokens: %(1)d"%{"1":len(toks)})

    # old way of tokenization
    #toks = nltk.regexp_tokenize(text, sentence_re)
    st = StanfordPOSTagger(config.stanford_bidirectional_tagger_path,config.stanford_postagger_jar_path,encoding="utf8",java_options="-mx8g")
    _postoks = st.tag(toks)
    # examine the postags, if "[", then change the tag to "X", create a new list
    postoks = []
    for pt in _postoks:
        if pt[0] == "[":
            postoks.append(('[','X'))
        elif pt[0] == "]":
            postoks.append((']','X'))
        else:
            postoks.append(pt)
    logger.info(postoks)
    
    # NLTK POS Tagger
    #postoks = nltk.tag.pos_tag(toks)
    logger.debug("postoks: %(1)d"%{"1":len(postoks)})
    tree = chunker.parse(postoks)
    # cast a Tree into a ParentedTree
    ptree = nltk.ParentedTree.convert(tree)
    # for each token, record its tree position
    pos_map = generate_pos_map(ptree,span_toks)

    stopwords = nltk.corpus.stopwords.words('english')
    return get_terms(ptree, lemmatizer, stopwords, pos_map)
