import os
import word2vec
import cPickle as pickle
from nltk import word_tokenize
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def extract_raw_data(folder):
    pairs = []
    for fname in os.listdir(folder):
        if fname.endswith('.ann'):
            with open(os.path.join(folder, fname)) as f:
                for line in f:
                    data = line.strip().split('\t')
                    if len(data) == 3:
                        label = data[1].split()[0]
                        pairs.append((label, data[2]))
    return pairs


def extract_feature(pairs):
    mapping = {'Process': 0, 'Material': 1, 'Task': 2}
    X, Y = [], []
    for p in pairs:
        y = mapping[p[0]]
        wds = word_tokenize(p[1])
        x = None
        for wd in wds:
            vec = w2v[wd] if wd in w2v else w2v[UNK]
            if x is None:
                x = vec
            else:
                x += vec
        x /= len(wds)
        X.append(x)
        Y.append(y)
    return X, Y


def get_in_domain_w2v():
    w2v = {}
    if os.path.exists(args.compact_w2v):
        w2v = pickle.load(open(args.compact_w2v, 'rb'))
    else:
        from gensim.models import Word2Vec
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
        print 'loading pretrained word2vec model...'
        model = Word2Vec.load_word2vec_format(args.pretrain_w2v, binary=True)
        print 'done'
        for p in train_pairs + dev_pairs:
            wds = word_tokenize(p[1])
            for wd in wds:
                if wd in model and wd not in w2v:
                    w2v[wd] = model[wd]
        w2v[UNK] = np.random.uniform(-0.25, 0.25, model.vector_size)
        pickle.dump(w2v, open(args.compact_w2v, 'wb'))
    return w2v


def evaluate(y_pred, y_true):
    acc = accuracy_score(y_true, y_pred)
    return acc


def main():
    global train_pairs, dev_pairs, w2v
    train_pairs = extract_raw_data(args.train_folder)
    dev_pairs = extract_raw_data(args.dev_folder)
    w2v = get_in_domain_w2v()

    print 'extracting features for training set...'
    X_train, Y_train = extract_feature(train_pairs)
    print 'extracting features for dev set...'
    X_dev, Y_dev = extract_feature(dev_pairs)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_dev = scaler.transform(X_dev)

    # clf = LogisticRegression()
    clf = RandomForestClassifier(n_estimators=50)
    print 'training classifier'
    clf.fit(X_train, Y_train)

    Y_train_pred = clf.predict(X_train)
    train_acc = evaluate(Y_train_pred, Y_train)
    print 'train accuracy', train_acc

    Y_dev_pred = clf.predict(X_dev)
    dev_acc = evaluate(Y_dev_pred, Y_dev)
    print 'dev accuracy', dev_acc


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-train_folder', default='/data/home/cul226/semeval/train2')
    argparser.add_argument(
        '-dev_folder', default='/data/home/cul226/semeval/dev')
    argparser.add_argument(
        '-pretrain_w2v', default='/data/home/cul226/word2vec/wos_skip_300.bin')
    argparser.add_argument(
        '-compact_w2v',
        default='/data/home/cul226/semeval/w2v.pkl',
        help='Contains embeddings only for words in the dataset.')
    args = argparser.parse_args()

    UNK = '<UNK>'
    np.random.seed(0)
    main()
