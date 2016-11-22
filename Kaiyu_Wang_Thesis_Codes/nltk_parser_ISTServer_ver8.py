import nltk
import string
from nltk.corpus import stopwords
import os, re, pickle, math, time, codecs
import csv
import multiprocessing as mp
from multiprocessing import Pool
from nltk.tag.stanford import StanfordPOSTagger
from bs4 import BeautifulSoup
nltk.internals.config_java(options='-xmx4G')

#file paths
filepath = "/home/kuw176/Kaiyu_Thesis/paper_files_xml"
picklefilepath = "/data/kuw176/pickles"
csvpath = '/data/kuw176/results_5_19/results_stanford_2'
stanford_tagger_path = '/home/kuw176/lib/stanford-postagger-full-2015-12-09/models/english-left3words-distsim.tagger'
stanford_jar_path = '/home/kuw176/lib/stanford-postagger-full-2015-12-09/stanford-postagger.jar'

#global vairables

def convertUTF(onefilepath):
    f = codecs.open(onefilepath, 'r', 'utf8')
    text_list = []
    while True:
        try:
            line = f.readline()
        except UnicodeDecodeError:
            continue
        if not line:
            break

        #put into list
        text_list.append(line)

    return text_list

def readFile(filepath):
    filelist = os.listdir(filepath)
    #read file
    count = 0

    mgr = mp.Manager()
    #first generate kgrams and compute the tf and df using multiprocessing:
    #dict for storing all (gram, docID) pairs
    df_dic = {}
    #dict for computing PMI, use Manager to share it across processes
    #mgr = mp.Manager()
    pmi_dic = mgr.dict()

    #number of process
    processNum = 20
    pool_precompute = mp.Pool(processes = processNum)

    fileNum = len(filelist)
    offset = fileNum // processNum
    
    #get_kgrams_df_pmi(0, len(filelist), filelist, 0)
    
    ProcessList = []
    for i in range(processNum):
        if (i == processNum - 1):
            start = i * offset
            end = fileNum
        else:
            start = i * offset
            end = start + offset

        #call the function
        print (start, end, i)
        ProcessList.append(pool_precompute.apply_async(get_kgrams_df_pmi, args = (start, end, filelist, i)))
        #pool_precompute.apply_async(get_kgrams_df_pmi, args = (start, end, filelist, i))
    
    pool_precompute.close()
    pool_precompute.join()
    #pool_precompute.terminate()
    
    wordNum = 0
    for process in ProcessList:
        wordNum += process.get()

    #better to add this, terminate all the processes
    pool_precompute.terminate()

    print ('words: ', wordNum)
    #get_kgrams_df_pmi(0, len(filelist), filelist, 0, count)
    print ("begin to merge df and pmi dictionary")
    #merge df_list and pmi_dic separately
    #pmi_dic
    for i in range(processNum):
        print ("pmi_dic: ", i)
        pmi_pickle_name = 'pmi_dic_pickle_' + str(i) + '.txt'
        pmi_pickle_file = open(picklefilepath + '/' + pmi_pickle_name, 'rb')
        cur_pmi_dic = pickle.load(pmi_pickle_file)

        #merge:
        for word in cur_pmi_dic.keys():
            if word in pmi_dic:
                pmi_dic[word] = pmi_dic[word] + cur_pmi_dic[word]
            else:
                pmi_dic[word] = cur_pmi_dic[word]
    
    #df_dic
    for i in range(processNum):
        print("df_dic: ", i)
        df_pickle_name = 'df_list_pickle_' + str(i) + '.txt'
        df_pickle_file = open(picklefilepath + '/' + df_pickle_name, 'rb')
        cur_df_list = pickle.load(df_pickle_file)

        for tup in cur_df_list:
            if (tup[0] in df_dic):
                df_dic[tup[0]] = df_dic[tup[0]] + 1
            else:
                df_dic[tup[0]] = 1

    print('finish merging df and pmi dictionaries', len(df_dic.keys()), len(pmi_dic.keys()))
    '''
    #for testing
    for key in pmi_dic.keys():
        print key, pmi_dic[key]

    
    print "begin to compute pmi dictionary"
    #construct pmi dic
    pmi_dic = {key : 0 for key in pmi_list}
    print 'finish initialization'
    count = 0
    for onefile in filelist:
        print count
        onefilepath = os.path.join(filepath, onefile)
        text = resetFile(open(onefilepath, 'r'))
        constructPMIDic(pmi_dic, text)
        count += 1'''

        
    #acquire the features by multiprocessing
    processNum = 20
    pool = mp.Pool(processes = processNum)
    fileNum = len(filelist)
    offset = fileNum // processNum
    for i in range(processNum):
        if (i == processNum - 1):
            start = i * offset
            end = fileNum
        else:
            start = i * offset
            end = start + offset

        print(str(start) + ' ' + str(end))
        pool.apply_async(get_features, args = (df_dic, pmi_dic, start, end, filelist, wordNum))

    pool.close()
    pool_precompute.terminate()
    pool.join()
    #get_features(df_dic, pmi_dic, 0, len(filelist), filelist, wordNum)

def get_kgrams_df_pmi(start_index, end_index, filelist, group_index):
    #dictionary for pmi
    pmi_dic = {}
    #list for df
    df_list = []
    #a list for counting the amount of words
    wordNumList = []

    #create stanford postagger object
    st = StanfordPOSTagger(stanford_tagger_path, stanford_jar_path, encoding = 'utf8', java_options = '-mx8g')

    for onefile in filelist[start_index : end_index]:
        onefilepath = os.path.join(filepath, onefile)
        print("current: ", onefilepath)
        #important function, generate Ngrams and pmi_dic
        text_list = list()
        KgramDic = onefile_to_grams(onefilepath, pmi_dic, onefile, wordNumList, st, text_list)
        print(len(KgramDic.keys()))
        
        #compute tf
        text = resetFile(text_list)
        compute_tf(KgramDic, onefilepath, text)
        
        #compute df
        df_save_list(KgramDic, onefile, df_list)
    
    #save df_list and pmi_dic as pickles
    #naming rule: df_list_pickle_/pmi_dic_pickle_ + group_index + .txt
    df_list_name = 'df_list_pickle_' + str(group_index) + '.txt'
    pmi_dic_name = 'pmi_dic_pickle_' + str(group_index) + '.txt'
    
    df_pickle_file = open(picklefilepath + '/' + df_list_name, 'wb')
    pmi_pickle_file = open(picklefilepath + '/' + pmi_dic_name, 'wb')
    
    pickle.dump(df_list, df_pickle_file, True)
    pickle.dump(pmi_dic, pmi_pickle_file, True)

    df_pickle_file.close()
    pmi_pickle_file.close()

    wordNum = 0
    for num in wordNumList:
        wordNum += num

    return wordNum
    

def get_features(df_dic, pmi_dic, start_index, end_index, filelist, wordNum):
    for onefile in filelist[start_index : end_index]:
        onefilepath = os.path.join(filepath, onefile)
        #get pickle file path
        onefilepath_pickle = picklefilepath + '/' + onefile.split('.')[0] + '_pickle.txt'

        print (onefilepath_pickle)
        
        #acquire all the resources needed
        f = codecs.open(onefilepath, 'r', 'utf8', errors = 'replace')
        #reset file
        text_list = list()
        parseXML(text_list, f)
        text = resetFile(text_list)
        #print(text)
        #KgramList = onefile_to_grams(onefilepath)
        KgramDic = pickle.load(open(onefilepath_pickle, 'rb'))
        documentNum = len(filelist)
        gramNum = len(KgramDic.keys())
        
        features = {}
        
        for gram in KgramDic.keys():
            #features[gram] = []
            cur_features = []
            #compute tfidf
            tf = KgramDic[gram][4]
            df = df_dic[gram]
            tf_idf = compute_tfidf(df, tf, gramNum, documentNum, gram)
            
            #print gram + " " + str(tf) + " " + str(df) + " " + str(tf_idf)
            cur_features.append(tf)
            cur_features.append(df)
            cur_features.append(tf_idf)
            if (isinstance(df, float) or isinstance(tf, float)):
                continue
            
            #is capitalized ?
            if (checkCapitalized(gram)):
                cur_features.append(1)
            else:
                cur_features.append(0)

            #all capitalized ?
            if (checkAllCapitalized(gram)):
                cur_features.append(1)
            else:
                cur_features.append(0)

            #isMixed
            if (checkMixed(gram)):
                cur_features.append(1)
            else:
                cur_features.append(0)
            
            #isCiting
            if (isCiting(gram, text)):
                cur_features.append(1)
            else:
                cur_features.append(0)

            #in Brackets and Quotes
            BracketsAndQuotes = inBracketsAndQuote(gram, text)
            #inBrackets
            if (BracketsAndQuotes[0]):
                cur_features.append(1)
            else:
                cur_features.append(0)
            #inQuotes
            if (BracketsAndQuotes[1]):
                print("inQuotes: " + gram)
                cur_features.append(1)
            else:
                cur_features.append(0)

            '''
            #GDC
            GDC = compute_GDC(text, gram, tf)
            cur_features.append(GDC)'''

            #recognize part, title
            cur_features.append(KgramDic[gram][0])

            #recognize part, abstract
            cur_features.append(KgramDic[gram][1])

            #recognize part, body
            cur_features.append(KgramDic[gram][2])

            #recognize part, references
            #cur_features.append(KgramDic[gram][3])

            #print gram + ' ' + str(features[gram])

            npmi = computePMI(gram, pmi_dic, wordNum)
            cur_features.append(npmi)
            
            #print gram + ' ' + str(features[gram])
            #check the validity of the feature values, wrong values may result from wrong text
            if (tf <= 0 or df <= 0 or (not isinstance(tf, int)) or (not isinstance(df, int)) or npmi <= -1):
                print('found one: ' + gram + ' ' +  str(tf) + ' '  + str(df) + ' ' + str(npmi))
                continue

            #hard cut some grams with feature values smaller than a threshold
            #if (npmi < 0.1 or tf > 50):
            #   continue

            features[gram] = cur_features

        #choosing criteria definition
        pmi_thres = 1
        tfidf_thres = 1 #df:1 tf :1 is 0.007378
        df_thres = len(filelist) / 1
        writeCSV(features, onefile, pmi_thres, df_thres, tfidf_thres)
        
def writeCSV(features, onefile, pmi_thres, df_thres, tfidf_thres):
    #file name
    filename = onefile.split('.')[0] + '.csv'
    #create and open the file
    csvFile = open(csvpath + '/' + filename, 'w')
    #contents
    f_csv = csv.writer(csvFile)
    
    titles = [u'gram', u'tf', u'df', u'tfidf', u'first capitalized', u'all capitalized', u'isMixed', u'citation', u'inBrackets', u'inQuotes', u'title', u'abstract', u'body', u'PMI']
    f_csv.writerow(titles)
    
    for gram in features.keys():
        #check need to write or not 
        #if (features[gram][1] >= df_thres or features[gram][12] <= pmi_thres or features[gram][2] <= tfidf_thres):
        #    continue
        contents = [gram, features[gram][0], features[gram][1], features[gram][2], features[gram][3], features[gram][4], features[gram][5], features[gram][6], features[gram][7], features[gram][8], features[gram][9], features[gram][10], features[gram][11], features[gram][12]]
        f_csv.writerow(contents)
        if (gram == "XML"):
           print(gram, contents)

    csvFile.close()

def show_df(df_dic):
    word_list = df_dic.keys()
    for cur in word_list:
        print(cur + " " + str(df_dic[cur]))


def show_Kgram(KgramList):
    for k in KgramList.keys():
        for gram in KgramList[k]:
            print(gram)
#return two-dimension list    
def line_sent_tokenize(line):
    #a two-dimension list
    sentences = list()
    sent_tokens = nltk.sent_tokenize(line)

    for sent in sent_tokens:
        try:
            curList = nltk.word_tokenize(sent)
            sentences.append(curList)
        except UnicodeDecodeError:
            #print "can not decode: " + line
            continue
    return sentences

#return a simple list of words
def line_tokenize(line):
    words = []
    try:
         curList = nltk.word_tokenize(line)
         return curList
    except UnicodeDecodeError:
        #print "can not decode: " + line
        return None

def resetFile(text_list):
    #replace '-' with '_'
    replace_heaven(text_list)
    #reset format
    newText = ''
    for word in text_list:
        newText = newText + " " + word

    return newText

def resetFile_no_use(f):
    #read the whole file
    #text = f.read()
    
    #use re.split() to split to handle punctions
    #words_list = re.split(r'[.;,?!\s]\s*', text)
    words_list = []
    for line in f:
        try:
            curList = nltk.word_tokenize(line)
        except UnicodeDecodeError:
            continue
        for word in curList:
            words_list.append(word)

    #replace '_' with '_'
    replace_heaven(words_list)
    
    newText = ''
    for word in words_list:
        newText = newText + " " + word

    return newText

def resetSent(lines):
    word_list = []

    for line in lines.split('\n'):
        try:
            curList = nltk.word_tokenize(line)
        except UnicodeDecodeError:
            continue
        for word in curList:
            word_list.append(word)
    
    replace_heaven(word_list)
    newText = ''
    for word in word_list:
        newText = newText + " " + word

    return newText

def count_word(text, word):
    return text.count(' ' + word + ' ')

def compute_tf(KgramDic, onefilepath, text):
    #compute tf of every k-gram
    #store in a dictionary
    keys = KgramDic.keys()
    max_tf = -1
    for word in keys:
        tf = count_word(text, word)
        #print word + " " + str(tf)
        KgramDic[word].append(tf)
        if (tf > max_tf):
            max_tf = tf
    
    #use pickle to store the information
    #rule: original txt file name + '_pickle' + '.txt'
    pickle_file_name = onefilepath.split('/')[-1].split('.')[0] + '_pickle' + '.txt'
    pickle_file = open(picklefilepath + "/" + pickle_file_name, 'wb')
    pickle.dump(KgramDic, pickle_file, True)
    pickle_file.close()
    

#compute df
def df_save_list(KgramDic, onefile, df_list):
    for gram in KgramDic.keys():
        tup = (gram, onefile)
        df_list.append(tup)

def computePMI(gram, pmi_dic, wordNum):
    word_list = gram.split()
    if (len(word_list) == 1):
        return 1

    gramFirst = word_list[0]
    gramRest = constructWord(word_list[1:])

    if ((not gram in pmi_dic) or ( not gramFirst in pmi_dic) or ( not gramRest in pmi_dic)):
        print('Error!', gram)
        return -1

    
    denominator = float(pmi_dic[gram]) / wordNum
    numerator = (float(pmi_dic[gramFirst]) / wordNum) * (float(pmi_dic[gramRest]) / wordNum)
    #print "PMI: ", 'first: ', gramFirst, 'last: ', gramRest, pmi_dic[gram], pmi_dic[gramFirst], pmi_dic[gramRest], denominator, numerator
    #denominator = pmi_dic[gram]
    #numerator = pmi_dic[gramFirst] * pmi_dic[gramRest]
    pmi = math.log(denominator / numerator, 2)
    #print 'pmi:', gram, pmi
    npmi = pmi / (-math.log(denominator, 2))

    return npmi

    
def compute_tfidf(df, tf, gramNum, documentNum, gram):
    #tfidf = (float(tf) / float(gramNum)) * math.log(documentNum / df)
    try:
       tfidf = (1 + math.log(tf, 2)) * math.log(documentNum / df, 2)
    except ValueError:
       print('df: ' + str(df) + ' tf: ' + str(tf) + " " + gram)
       return 0
    return tfidf

def checkCapitalized(gram):
    #for single words:
    if (len(gram.split()) == 1):
        if (gram[0].isupper()):
            return True
        else:
            return False
    
    for word in gram.split()[1:]:
        if (word[0].isupper()):
            return True
    return False

def checkAllCapitalized(gram):
    #for single words:
    if (len(gram.split()) == 1):
        if (gram == gram.upper()):
            return True
        else:
            return False
    
    for word in gram.split()[0:]:
        if (word == word.upper()):
            return True
    
    return False

def checkMixed(gram):
    flag = False
    for word in gram.split():
        #special case: less than 2, not mixed
        if (len(word) <= 2):
            continue
        lowerFlag = False
        upperFlag = False
        for c in word[1:-1]:
            if (c.isupper()):
                upperFlag = True
            if (c.islower()):
                lowerFlag = True

        if (lowerFlag == True & upperFlag == True):
            return True

    return False

def isCiting(gram, text):
    #begin to find the word from index 0
    lastLocation = 0
    endIndex = len(text)
    
    while ((text.find(' ' + gram + ' ', lastLocation, endIndex)) != -1):
        curLocation = text.find(' ' + gram + ' ', lastLocation, endIndex)
        #copy
        firstHalf = 0
        secondHalf = 0
        length = 50
        if (curLocation >= length):
            firstHalf = curLocation - length
        else:
            firstHalf = 0
        if (curLocation <= (len(text) - (len(gram) + length))):
            secondHalf = curLocation+ len(gram) + length
        else:
            secondHalf = len(text)
        cutStr = text[firstHalf : secondHalf]

        #check if citation exist
        match = re.findall(r'\(.*?[a-z]+.*[1-9]+\s*\)|\[.*?[a-z]+.*[1-9]+\s*\]', cutStr)
        if len(match) > 0 :
            return True

        #update location
        lastLocation = curLocation + len(gram) + 2

    return False
       
def inBracketsAndQuote(gram, text):
    result = [False, False]
    #print(gram)
    if (len(gram.split()) > 1):
        #print('not a unigram')
        result[1] = inQuote(gram, text)
        return result
    else:
        lastLocation = 0
        endIndex = len(text)
        #search gram in the text
        while ((text.find(' ' + gram + ' ', lastLocation, endIndex)) != -1):
              curLocation = text.find(' ' + gram + ' ', lastLocation, endIndex)
              flagleft = False
              flagright = False

              #print(gram + ' ' + 'left: ' + text[curLocation - 2] + text[curLocation - 1])
              #print(gram + ' ' + 'right: ' + text[curLocation + len(gram) + 1] + text[curLocation + len(gram) + 2])
              if ((text[curLocation - 2] == '(') or (text[curLocation - 1] == '(')):
                  flagleft = True
              if ((text[curLocation + len(gram) + 2] == ')') or (text[curLocation + len(gram) + 1] == ')')):
                  flagright = True

              if (flagleft and flagright):
                 #print('find one!')
                 result[0] = True

              flagleft = False
              flagright = False
              if ((text[curLocation - 2] == '`') or (text[curLocation - 1] == '`')):
                  flagleft = True
              if ((text[curLocation + len(gram) + 2] == '`') or (text[curLocation + len(gram) + 1] == '`')):
                  flagright = True
              
              if (flagleft and flagright):
                 #print('find one!')
                 result[1] = True

              if (result[0] == True and result[1] == True):
                 return result
              else:
                 lastLocation = curLocation + len(gram) + 1

        #print('not one!')
        return result

    return result

def inQuote(gram, text):
     lastLocation = 0
     endIndex = len(text)
     #search gram in the text
     while ((text.find(' ' + gram + ' ', lastLocation, endIndex)) != -1):
            curLocation = text.find(' ' + gram + ' ', lastLocation, endIndex)
            flagleft = False
            flagright = False
            
            #for test
            ''' 
            if (len(gram.split()) == 2):
                print(gram + ' ' + 'left1: ' + text[curLocation - 2] + " left2: " + text[curLocation - 1])
                print(gram + ' ' + 'right1: ' + text[curLocation + len(gram) + 1] + " right2: " + text[curLocation + len(gram) + 2])'''

            if ((text[curLocation - 2] == '`') or (text[curLocation - 1] == '`')):
                flagleft = True
            if ((text[curLocation + len(gram) + 2] == '`') or (text[curLocation + len(gram) + 1] == '`')):
                flagright = True

            if (flagleft and flagright):
               print('find one: ', gram)
               return True
            else:
               lastLocation = curLocation + len(gram) + 1

     return False

def word_order(text, gram):
    wordLocation = text.find(' ' + gram + ' ')

    gap = len(text) // 10

    order = int(math.ceil(wordLocation // gap))

    if (order == 0):
        order = 1

    return order

def compute_GDC(text, gram, tf):
    #count the frequency of gram
    gram_f = tf
    #special case
    if (gram_f == 0):
        return 0

    #split the gram
    singles = gram.split(' ')
    if (len(singles) > 5):
        print("error!")
        return 0

    sum_f = 0
    for word in singles:
        sum_f = sum_f + count_word(text, word)

    try:
        
        GDC = (len(singles) * math.log10(gram_f) * gram_f) / sum_f
    except ValueError:
        print('error GDC!')
        return 0
    
    if (len(singles) == 1):
        #note that GDC may equal to 0
        GDC = 1
        #GDC = 0.1 * GDC
    
    return GDC
            
#maybe useless:
def remove_useless_words(word_list):
    filtered_list = []

    #a list for single numbers
    singlenum = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    
    #junk_list = []
    for word in word_list:
        if ((word in singlenum) | (len(word) < 1)):
            #junk_list.append(word)
            continue
        filtered_list.append(word)

    return filtered_list

def checkHasLetters(word):
    flag = False
    for letter in word:
        if (letter.isalpha()):
            flag = True
    
    return flag

def combineSentense(sen1, sen2):
    #eliminate the last '\n' of sen1
    newSen1 = sen1[0 : -1]
    #eliminate the first "" of sen2
    newSen2 = sen2[1: ]

    return newSen1 + " " + newSen2

def parseXML_separate(document, text_list, f):
    #create soup
    soup = BeautifulSoup(f, 'lxml')

    #extract title
    if (not (soup.title is None)):
       title = soup.title.string
       if (title is not None and len(title) != 0):
          curTokens = line_tokenize(title)
          document['title'].extend(curTokens)
          #curTokens = line_tokenize(title)
          text_list.extend(curTokens)

    #extract abstract
    abstract = ''
    if (not (soup.abstract is None)):
       for abs_string in soup.abstract.stripped_strings:
           abstract = abstract  + abs_string
       #print(abstract)
       if (abstract is not None and len(abstract) != 0):
          curTokens = line_tokenize(abstract)
          document['abstract'].extend(curTokens)
          #put into text_list
          #curTokens = line_tokenize(abstract)
          text_list.extend(curTokens)

    #extract body
    body_list =  soup.find_all('div')
    body_list_length = len(body_list) - 1

    body = ''
    '''
    for i in range(body_list_length):
        #section name
        if (not(body_list[i].contents[0].string is None)):
           body = body + body_list[i].contents[0].string
        if (not (body_list[i].p is None)):
           paragraph_list = body_list[i].p.contents
           #print(paragraph_list)
           for paragraph in paragraph_list:
               #print(type(paragraph))
               if(isinstance(paragraph, str)):
                  #print('not reference: ' + paragraph)
                  body = body + paragraph
               else:
                  #print('reference: ' + paragraph.string)
                  body = body + paragraph.string'''

    for i in range(body_list_length):
        for body_string in body_list[i].stripped_strings:
            body = body  + ' ' + body_string

    if (body is not None and len(body) != 0):
       curTokens = line_tokenize(body)
       document['body'].extend(curTokens)
       #curTokens = line_tokenize(body)
       text_list.extend(curTokens)

    #extract figure and tables
    figure_list = soup.find_all('figure')
    figure_list_length = len(figure_list)
    
    figure = ''
    for i in range(figure_list_length):
        for figure_string in figure_list[i].stripped_strings:
            figure = figure + ' ' + figure_string

    #print('figure strings:')
    #print(figure)
    if (figure is not None and len(figure) != 0):
        #regard as a part of the body
        curTokens = line_tokenize(figure)
        document['body'].extend(curTokens)
        text_list.extend(curTokens)


    #return nothing

def parseXML(text_list, f):
    #create soup
    soup = BeautifulSoup(f, 'lxml')

    #extract title
    if (not (soup.title is None)):
       title = soup.title.string
       if (title is not None and len(title) != 0):
          curTokens = line_tokenize(title)
          text_list.extend(curTokens)

    #extract abstract
    abstract = ''
    if (not (soup.abstract is None)):
       for abs_string in soup.abstract.stripped_strings:
           abstract = abstract  + abs_string
       #print(abstract)
       if (abstract is not None and len(abstract) != 0):
          curTokens = line_tokenize(abstract)
          #put into text_list
          text_list.extend(curTokens)

    #extract body
    body_list =  soup.find_all('div')
    body_list_length = len(body_list) - 1

    body = ''
    for i in range(body_list_length):
        for body_string in body_list[i].stripped_strings:
            body = body  + ' ' + body_string

    if (body is not None and len(body) != 0):
       curTokens = line_tokenize(body)
       text_list.extend(curTokens)

    #extract figure and tables
    figure_list = soup.find_all('figure')
    figure_list_length = len(figure_list)

    figure = ''
    for i in range(figure_list_length):
        for figure_string in figure_list[i].stripped_strings:
            figure = figure + ' ' + figure_string

    if (figure is not None and len(figure) != 0):
        #regard as a part of the body
        curTokens = line_tokenize(figure)
        text_list.extend(curTokens)

    #return nothing

def onefile_to_grams(onefilepath, pmi_dic, onefile, wordNumList, st, text_list):
    
    #open the xml file
    f = codecs.open(onefilepath, 'r', 'utf8', errors  = 'replace')
    document = {}

    document['title'] = []
    document['abstract'] = []
    document['body']= []
    #document['reference'] = []

    #parse xml file
    parseXML_separate(document, text_list, f)
    #for counting the number of words in the whole corpus
    #global wordNum
    wordNumList.append(len(document['abstract']) + len(document['title']) + len(document['body']))

    #replace '_' with '-' for stanford postagger
    for key in document.keys():
        replace_heaven(document[key])

    KgramDic = generateKgrams(document, st)
    constructPMIDic(document, pmi_dic)
   
    return KgramDic


#this function replace all the words having "_" with "-" 
def replace_heaven(word_list):
    for i in range(len(word_list)):
        word = word_list[i]
        #check "_"
        if '_' in word:
           new_word = word.replace('_', '-')
           word_list[i] = new_word

def constructPMIDic(document, pmi_dic):

    words = []
    for key in document.keys():
        words.extend(document[key])

    #at most 4 grams
    for i in range(5):
        for j in range(len(words) - i):
            newGram = constructWord(words[j : j + 1 + i])
            if (checkValidPartialGram(newGram)):
                if (newGram in pmi_dic):
                    pmi_dic[newGram] += 1
                else:
                    pmi_dic[newGram] = 1
#no use:
def constructPMIDic2(document, pmi_dic):
    words = []
    for key in document.keys():
        words.extend(document[key])

    word_tuples = nltk.pos_tag(words)
    #at most 4
    for i in xrange(1, 5):
        #one gram
        if (i == 1):
            grammar = "1gram: {<DT | NN | JJ>}"
            name = '1gram'
        elif (i == 2):
            #2 gram
            grammar = "2gram: {<DT | NN | JJ | IN> <NN>}"
            name = '2gram'
        elif (i == 3):
            #3 gram
            grammar = "3gram: {<DT | NN | JJ | IN> <DT | NN | JJ | IN><NN>}"
            name = '3gram'
        elif (i == 4):
            #4 gram
            grammar = "4gram: {<DT | NN | JJ | IN> <DT | NN | JJ | IN><DT | NN | JJ | IN><NN>}"
            name = '4gram'
        cp = nltk.RegexpParser(grammar)
        result_gram = cp.parse(word_tuples)
        gram_list = getNodes_duplicates(result_gram, name)
        #put into dictionary
        for gram in gram_list:
            if (gram in pmi_dic):
                pmi_dic[gram] += 1
            else:
                pmi_dic[gram] = 1
    return
            
def constructWord(word_list):
    result = ''
    for word in word_list:
        result = result + " " + word

    return result[1:]

def getNodes(parent, gramName, section):
    #use a set to avoid duplicates
    gram_list = set()
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == gramName:
                curGram = ''
                for word_tuple in node.leaves():
                    curGram = curGram + " " + word_tuple[0]
                #eliminate the first space
                curGram = curGram[1:]
                #validation and put into the set
                if (checkValidGram(curGram, section)):
                    #print curGram
                    gram_list.add(curGram)
            else:
                print("Label:", node.label())
                print("Leaves:", node.leaves())

            gram_list.update(getNodes(node, gramName, section))
        else:
            "word2:", node
    
    return gram_list

#no use:
def getNodes_duplicates(parent, gramName, section):
    gram_list = []
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == gramName:
                curGram = ''
                for word_tuple in node.leaves():
                    curGram = curGram + " " + word_tuple[0]
                #eliminate the first space
                curGram = curGram[1:]
                #validation and put into the set
                if (checkValidGram(curGram, section)):
                    gram_list.append(curGram)
            else:
                print("Label:", node.label())
                print("Leaves:", node.leaves())

            gram_list.extend(getNodes_duplicates(node, gramName, section))
        else:
            "word2:", node
    
    return gram_list

def checkValidGram(gram, section):
    #a valid gram should have at least one letter
    hasLetter = False
    
    punc_list = []
    for punc in string.punctuation:
        if (punc == '-'):
            continue
        punc_list.append(punc)

    punc_list.append('“')
    punc_list.append('”')
    punc_list.append('’')
    punc_list.append('‘')

    #check encoding
    if (not checkEncoding(gram)):
       #print('found one: ', gram)
       return False

    #unigram handled indivudually
    if (len(gram.split()) == 1):
        word = gram.split()[0]
        if (len(word) == 1):
           return False
        #digit followed by a lowercase letter
        if (re.match(r'[0-9]+[a-z]', word) is not None):
           return False

        #illegal words list
        illegals = [r'the']
        for illegal in illegals:
            if (re.match(illegal, word.lower()) is not None):
               return False

        #words in title, all letters capitalized
        if (section == 'title'):
            #check if contains invalid characters or lowercase letters
            for c in word:
                if (c in punc_list or c.islower()):
                   return False
        else:
            if (word[0].islower()):
                return False
            #check if contains invalid characters
            for c in word:
                if (c in punc_list):
                   return False

        return True
    
    #k grams (k > 1):
    word_list = gram.split()
    #check the first term
    if (len(word_list[0]) == 1 and word_list[0].islower()):
       return False
    
    #check using regular expression
    preword_list = [r'same', r'similar', r'correct', r'such', r'only', r'different', r'other', r'several', r'some', r'available', r'satisfactory', r'perfect', r'significant', r'valuable', r'harmful', r'unseen', r'good', r'multiple', r'appendix', r'number of', r'others', r'new', r'many', r'further', r'certain', 'the']
    include_list = [r'except', r'such as', r'between', 'into', r'the']

    #starting with
    for preword in preword_list:
        if (re.match(preword, gram.lower()) is not None):
           return False

    #contains
    for inword in include_list:
        if (re.search(inword, gram.lower()) is not None):
           return False

    #.....from.....
    pattern = r'[A-Za-z\s]( from )[A-Za-z\s]'
    if (re.search(pattern, gram) is not None):
        return False

    single_count = 0
    for word in word_list:
        #check if the current term is a single letter
        if (len(word) == 1):
           single_count += 1
        for c in word:
            if (c.isalpha()):
                hasLetter = True
            if (c in punc_list):
                return False

    return (hasLetter and (single_count < 2))

def checkEncoding(gram):
    for word in gram.split():
        if (not all(ord(c) < 128 for c in word)):
           return False

    return True

def checkValidPartialGram(gram):
    punc_list = []
    for punc in string.punctuation:
        if (punc == '-'):
            continue
        punc_list.append(punc)

    for word in gram.split():
        for c in word:
            if (c in punc_list):
                return False

    return True

def generateKgrams(document, st):
    KgramDic = {}
    #title
    if (('title' in document) and len(document['title']) != 0):
        Kgram(document['title'], 0, KgramDic, st, 'title')

    if (('abstract' in document) and len(document['abstract']) != 0):
        Kgram(document['abstract'], 1, KgramDic, st, 'abstract')

    if (('body' in document) and len(document['body']) != 0):
        Kgram(document['body'], 2, KgramDic, st, 'body')

    if (('reference' in document) and len(document['reference']) != 0):
        Kgram(document['reference'], 3, KgramDic, st, 'reference')

    return KgramDic


def Kgram(document, pos, KgramDic, st, section):
    
    #filtered_text = remove_useless_words(document)
    #word_tuples = nltk.pos_tag(document)
    #use stanford nlp tagger
    #print(document)
    word_tuples = st.tag(document)

    #unigram, more complicated situations
    for i in range(len(word_tuples)):
        #need to consider the previous of the current word
        word_tuple = word_tuples[i]
        if word_tuple[1].startswith('NN'):
            gram = word_tuple[0]
            if (checkValidGram(gram, section)):
                #check the previous character
                if (i == 0 or (word_tuples[i - 1][0] in ".!?")):
                   continue
                
                #print(gram, word_tuples[i - 1][0])
                if (gram in KgramDic):
                    KgramDic[gram][pos] = 1
                else:
                    KgramDic[gram] = [0 ,0, 0, 0]
                    KgramDic[gram][pos] = 1
    
    #2-gram
    Grammar_2_gram = "2gram: {<NN | JJ | NNP | NNS> <NN | NNP | NNS>}"
    cp = nltk.RegexpParser(Grammar_2_gram)
    result_2_gram = cp.parse(word_tuples)
    gram_list = getNodes(result_2_gram, '2gram', section)
    for gram in gram_list:
        if (gram in KgramDic):
            KgramDic[gram][pos] = 1
        else:
            KgramDic[gram] = [0 ,0, 0, 0]
            KgramDic[gram][pos] = 1
        
        #construct pmi list
        #constructPMIList(gram, pmi_list)
        
    #3-gram
    Grammar_3_gram = "3gram: {<NN | JJ | NNP | NNS><IN | JJ | NN | NNP | NNS><NN | NNP | NNS>}"
    cp = nltk.RegexpParser(Grammar_3_gram)
    result_3_gram = cp.parse(word_tuples)
    gram_list = getNodes(result_3_gram, '3gram', section)
    for gram in gram_list:
        if (gram in KgramDic):
            KgramDic[gram][pos] = 1
        else:
            KgramDic[gram] = [0 ,0, 0, 0]
            KgramDic[gram][pos] = 1

        #construct pmi list
        #constructPMIList(gram, pmi_list)

    #4-gram
    Grammar_4_gram = "4gram: {<NN | JJ | NNP | NNS><IN | JJ | NN | NNP | NNS><IN | JJ | NN | NNP | NNS><NN | NNP | NNS>}"
    cp = nltk.RegexpParser(Grammar_4_gram)
    result_4_gram = cp.parse(word_tuples)
    gram_list = getNodes(result_4_gram, '4gram', section)
    for gram in gram_list:
        if (gram in KgramDic):
            KgramDic[gram][pos] = 1
        else:
            KgramDic[gram] = [0 ,0, 0, 0]
            KgramDic[gram][pos] = 1
        #construct pmi list
        #constructPMIList(gram, pmi_list)
    
    #5-gram
    Grammar_5_gram = "5gram: {<NN | JJ | NNP | NNS><IN | JJ | NN | NNP | NNS><IN | JJ | NN | NNP | NNS><IN | JJ | NN | NNP | NNS><NN | NNP | NNS>}"
    cp = nltk.RegexpParser(Grammar_5_gram)
    result_5_gram = cp.parse(word_tuples)
    gram_list = getNodes(result_5_gram, '5gram', section)
    for gram in gram_list:
        if (gram in KgramDic):
            KgramDic[gram][pos] = 1
        else:
            KgramDic[gram] = [0 ,0, 0, 0]
            KgramDic[gram][pos] = 1
        #construct pmi list
        #constructPMIList(gram, pmi_list)


if __name__ == '__main__':
    #time.clock()
    readFile(filepath)
    print(time.clock())
