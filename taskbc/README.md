### KB based classification
-----------------------------------
sentence extraction from Wikipedia: 

1. wikiQuery.py: Extract content for query. Input: ann file from SemEval data. Output: `*-wikisearch.json`. The json file contains dictionaries for each phrase in the ANN file. Each dictionary contains various information from first 10 search results with that query. 

2. procesWikiSearchResults.py : Generate sentences from wiki search results. Each sentence contains one or multiple of the phrases. 

For example, the ann file: `dataExample/S2352179115300041.ann`. `wikiQuery.py` generates `dataExample/S2352179115300041-wikisearch.ann`. `procesWikiSearchResults.py` uses `dataExample/S2352179115300041-wikisearch.ann` and generates `dataExample/S2352179115300041-wikisearch-sentences.ann`.  

