import json
from nltk import sent_tokenize,word_tokenize
import sys

def wikiSentences(phrase,sr):
    if sr is not None and sr['pageContent'] is not None:
        sents=[x.lower() for x in sent_tokenize(sr['pageContent']) if "\n" not in x]
        phraseWords=[x.lower() for x in word_tokenize(phrase)]
        validSentences=[]
        for sent in sents:
            for word in phraseWords:
                if word in word_tokenize(sent):
                    #print "found",word,"in",sent
                    validSentences.append(sent)
        return validSentences
    else:
        return []

def processJson(wikiSearchJsonLoc):
    con=[x for x in json.load(open(wikiSearchJsonLoc)) if x is not None]
    results=[]
    for item in con:
        phrase=item['phrase']
        resultDict={
            "phraseCharStart": item["phraseCharStart"],
            "phraseCharEnd": item["phraseCharEnd"], 
            "phraseIndex": item["phraseIndex"], 
            "phraseGoldStandardTag": item["phraseGoldStandardTag"], 
            "phrase": phrase,
            "wikiSearchResults":[{
                 "snippet":x["snippet"],
                 "pageCategories":x["pageCategories"],
                 "pageSentences":wikiSentences(phrase,x),
                 "titleUrl": x["titleUrl"], 
                 "title": x["title"]                      
            } for x in item['wikiSearchResults']] 
        }
        results.append(resultDict)
    return results

def main():
    wikisearchJsonLoc=sys.argv[1]
    with open(wikisearchJsonLoc[:-5]+"-sentences.json","wb") as f:
        f.write(json.dumps(processJson(wikisearchJsonLoc), indent=4, sort_keys=False))
    print "written results at",wikisearchJsonLoc[:-5]+"-sentences.json"

if __name__ == "__main__":
    main()    
 
                  
            
