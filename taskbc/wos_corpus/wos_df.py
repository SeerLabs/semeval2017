import json
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


def process_word(wd):
    return stemmer.stem(wd.lower())

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-wos_path', default='/data/home/cul226/wos_dump')
    argparser.add_argument('-output', default='/data/home/cul226/wos_df')
    args = argparser.parse_args()

    stemmer = PorterStemmer()

    cnt = 0
    df = Counter()
    with open(args.wos_path) as f:
        for line in f:
            data = json.loads(line.strip())
            text = data['abstract']
            flags = set()
            for sent in sent_tokenize(text):
                for wd in word_tokenize(sent):
                    t = process_word(wd)
                    if t not in flags:
                        flags.add(t)
                        df[t] += 1
            cnt += 1
            if cnt % 1000 == 0:
                print 'processed {}k lines'.format(cnt / 1000)

    with open(args.output, 'w') as out:
        out.write(str(cnt) + '\n')
        for (k, v) in df.most_common():
            out.write('{}\t{}\n'.format(k, v))

