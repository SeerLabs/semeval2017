from nltk.tokenize import sent_tokenize, word_tokenize
import json
import string


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-wos_path', default='/data/home/cul226/wos_dump')
    argparser.add_argument('-output', default='/data/home/cul226/wos_sents')
    args = argparser.parse_args()

    # ignore punctuations
    punkts = set(string.punctuation)
    punkts.add('``')
    punkts.add("''")

    cnt = 0
    with open(args.wos_path) as f, open(args.output, 'w') as out:
        for line in f:
            cnt += 1
            if cnt % 1000 == 0:
                print 'processed {}k lines'.format(cnt / 1000)
            data = json.loads(line.strip())
            text = data['abstract']
            for sent in sent_tokenize(text):
                wds = [wd for wd in word_tokenize(sent)
                       if wd not in punkts]
                out.write(' '.join(wds) + '\n')
