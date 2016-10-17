import requests
from pprint import pprint
from bs4 import BeautifulSoup
import wikipedia
import json
import os

def getRequest(url):
    try:
        r=requests.get(url)
    except requests.exceptions.RequestException as e:
        print e
        return None 
    if r.status_code!=200:
        print "request unsuccessful"
        return None
    else:
        return r 
 
def pageCategory(title):
    baseUrl=u"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=categories&cllimt=500&titles={0}"
    url=baseUrl.format(title)
    r=getRequest(url)
    if r == None:
        return None
    else:
        rJson=r.json()
        try:
            return '\n'.join([j['title'].split('Category:')[1] \
            for j in rJson['query']['pages'][rJson['query']['pages'].keys()[0]]['categories']])
        except KeyError:
            return ""
  
def pageContent(title):
    try:
        r=wikipedia.page(title)
        return r.content
    except (wikipedia.exceptions.PageError,wikipedia.exceptions.DisambiguationError):
        print "some errors from wiki modules"
        return None

    
def fullSearch(queryTerm):
    baseUrl=u"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={0}&utf8&format=json"
    url=baseUrl.format("%20".join(queryTerm.split()))
    r=getRequest(url)
    if r==None:
        return None
    else:
        snippets=[BeautifulSoup(j['snippet'],'html.parser').get_text() for j in \
        r.json()['query']['search']]
        titles=[j['title'] for j in r.json()['query']['search']] 
        titleUrls=['https://en.wikipedia.org/wiki/'+'_'.join(t.split()) for t in titles]
        pageCategories=[pageCategory('_'.join(t.split())) for t in titles]
        pageContents=[pageContent('_'.join(t.split())) for t in titles]
        searchResults=[]
        for (t,tU,pC,pCat,sn) in zip(titles,titleUrls,pageContents,pageCategories,snippets):
            searchResult={
                'title':t,
                'titleUrl':tU,
                'pageContent':pC,
                'pageCategories':pCat,
                'snippet':sn   
            }
            searchResults.append(searchResult)
        return searchResults
            

def oneLine(dataLine):
    if not dataLine.startswith("T") : #relation tag
        print "data line tags relations, not extracting features"
        return None

    dataLineContent=dataLine.split("\t")
    phrase=dataLineContent[-1]
    phraseGoldStandardTag=dataLineContent[1].split()[0]
    phraseCharStart=dataLineContent[1].split()[1]
    phraseCharEnd=dataLineContent[1].split()[2]
    phraseIndex=dataLineContent[0]
    
    try:
        print(u"processing phrase: {0}, startIndex: {1},endIndex {2}, termIndex: {3},\
                goldStandard: {4}"\
        .format(phrase,phraseCharStart,phraseCharEnd,phraseIndex,phraseGoldStandardTag))
    except UnicodeDecodeError:
        print "search phrase can not be decoded"
        return  {
            'phrase': phrase,
            'phraseIndex': phraseIndex,
            'phraseCharStart':phraseCharStart,
            'phraseCharEnd':phraseCharEnd,
            'phraseGoldStandardTag':phraseGoldStandardTag,
            'wikiSearchResults':[]
        }

       

    
    results=fullSearch(phrase)
    return {
        'phrase': phrase,
        'phraseIndex': phraseIndex,
        'phraseCharStart':phraseCharStart,
        'phraseCharEnd':phraseCharEnd,
        'phraseGoldStandardTag':phraseGoldStandardTag,
        'wikiSearchResults':results    
    }

def fromFile(annFileLoc):
    print "processing file",annFileLoc
    
    searchResults=[oneLine(x) for x in open(annFileLoc).read().split("\n")[:-1]]
    with open(annFileLoc[:-4]+"-wikisearch.json","wb") as f:
        f.write(json.dumps(searchResults, indent=4, sort_keys=False)) 
    print "wiki search results written to",annFileLoc[:-4]+"-wikisearch.json" 
   

def main():
    dataLine="T1	Process 0 16	Complex Langevin"
    #print oneLine(dataLine)['phrase']
    annDir="/home/sagnik/data/scienceie/devdata/dev/anns/"
    [fromFile(os.path.join(annDir,x)) for x in os.listdir(annDir) if x.endswith('ann') and not os.path.exists(os.path.join(annDir,x)[:-4]+"-wikisearch.json")]
    
     

if __name__ == "__main__":
    main() 
