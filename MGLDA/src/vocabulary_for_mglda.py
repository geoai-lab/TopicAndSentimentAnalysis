#!/usr/local/bin/python
# -*- coding: utf-8 -*-

'''
Created on Sep 11, 2012

mglda用にwindowとかをつくります

'''

import nltk, re
import csv
from nltk.corpus import stopwords

def load_corpus(range):
    m = re.match(r'(\d+):(\d+)$', range)
    if m:
        start = int(m.group(1))
        end = int(m.group(2))
        from nltk.corpus import brown as corpus
        return [corpus.words(fileid) for fileid in corpus.fileids()[start:end]]

def load_corpus_each_sentence(range):
    m = re.match(r'(\d+):(\d+)$', range)
    if m:
        start = int(m.group(1))
        end = int(m.group(2))
#        from nltk.corpus import brown as corpus
        from nltk.corpus import movie_reviews as corpus
        return [corpus.sents(fileid) for fileid in corpus.fileids()[start:end]]

def load_file(filename):
    corpus = []
    f = open(filename, 'r')
    for line in f:
        doc = re.findall(r'\w+(?:\'\w+)?',line)
        if len(doc)>0:
            corpus.append(doc)
    f.close()
    return corpus

def construct_review_corpus(data_file):
    reviews = []
    default_stopwords = stopwords.words('english')
    additional_stopwords = ["manhattan","new york","nyc","brooklyn","bronx","area","areas","nice","good","bad","great","place","places","lot","lots","neighborhood","city","thing","things","years","queens"]

    with open(data_file, 'rb') as csvfile:
        csvreader = csv.DictReader(csvfile)   #, delimiter=',', quotechar='"')
        index = 1
        for row in csvreader:
            print("The review is: "+row["review_content"])
            neighborhood_name = row["neighborhood"].lower().strip()
            review_content = row["review_content"].lower()
            review_content = review_content.replace(neighborhood_name,'')
            
            for special_word in additional_stopwords:
                review_content = review_content.replace(special_word,'')
            review_content = re.sub('\s+', ' ', review_content)
            review_content = review_content.strip()
            
            # since we split sentences with comma, we need to concatenate the short phrases
            review_sentence_array_raw = re.split('[;!?.,]', review_content)
            review_sentence_array = []
            for raw_sentence in review_sentence_array_raw:
                if len(raw_sentence.split()) <= 3:
                    if len(review_sentence_array) > 0:
                        review_sentence_array[-1] += " " + raw_sentence.strip()
                    else:
                        review_sentence_array.append(raw_sentence.strip())
                else:
                    review_sentence_array.append(raw_sentence)
            
            review_sentence_array_word = []
            for review_sentence in review_sentence_array:
                review_sentence = re.sub('[^a-zA-Z]', ' ', review_sentence)
                review_sentence = re.sub('\s+', ' ', review_sentence)
                review_words = [word for word in review_sentence.split() if word not in default_stopwords and len(word) > 1]
                review_sentence_array_word.append(review_words)
            
            reviews.append(review_sentence_array_word)
            # for test; using only a small sample of data
            index += 1
            #if index > 100:
            #    break
            
    return reviews

#stopwords_list = nltk.corpus.stopwords.words('english')
stopwords_list = "a,s,able,about,above,according,accordingly,across,actually,after,afterwards,again,against,ain,t,all,allow,allows,almost,alone,along,already,also,although,always,am,among,amongst,an,and,another,any,anybody,anyhow,anyone,anything,anyway,anyways,anywhere,apart,appear,appreciate,appropriate,are,aren,t,around,as,aside,ask,asking,associated,at,available,away,awfully,be,became,because,become,becomes,becoming,been,before,beforehand,behind,being,believe,below,beside,besides,best,better,between,beyond,both,brief,but,by,c,mon,c,s,came,can,can,t,cannot,cant,cause,causes,certain,certainly,changes,clearly,co,com,come,comes,concerning,consequently,consider,considering,contain,containing,contains,corresponding,could,couldn,t,course,currently,definitely,described,despite,did,didn,t,different,do,does,doesn,t,doing,don,t,done,down,downwards,during,each,edu,eg,eight,either,else,elsewhere,enough,entirely,especially,et,etc,even,ever,every,everybody,everyone,everything,everywhere,ex,exactly,example,except,far,few,fifth,first,five,followed,following,follows,for,former,formerly,forth,four,from,further,furthermore,get,gets,getting,given,gives,go,goes,going,gone,got,gotten,greetings,had,hadn,t,happens,hardly,has,hasn,t,have,haven,t,having,he,he,s,hello,help,hence,her,here,here,s,hereafter,hereby,herein,hereupon,hers,herself,hi,him,himself,his,hither,hopefully,how,howbeit,however,i,d,i,ll,i,m,i,ve,ie,if,ignored,immediate,in,inasmuch,inc,indeed,indicate,indicated,indicates,inner,insofar,instead,into,inward,is,isn,t,it,it,d,it,ll,it,s,its,itself,just,keep,keeps,kept,know,knows,known,last,lately,later,latter,latterly,least,less,lest,let,let,s,like,liked,likely,little,look,looking,looks,ltd,mainly,many,may,maybe,me,mean,meanwhile,merely,might,more,moreover,most,mostly,much,must,my,myself,name,namely,nd,near,nearly,necessary,need,needs,neither,never,nevertheless,new,next,nine,no,nobody,non,none,noone,nor,normally,not,nothing,novel,now,nowhere,obviously,of,off,often,oh,ok,okay,old,on,once,one,ones,only,onto,or,other,others,otherwise,ought,our,ours,ourselves,out,outside,over,overall,own,particular,particularly,per,perhaps,placed,please,plus,possible,presumably,probably,provides,que,quite,qv,rather,rd,re,really,reasonably,regarding,regardless,regards,relatively,respectively,right,said,same,saw,say,saying,says,second,secondly,see,seeing,seem,seemed,seeming,seems,seen,self,selves,sensible,sent,serious,seriously,seven,several,shall,she,should,shouldn,t,since,six,so,some,somebody,somehow,someone,something,sometime,sometimes,somewhat,somewhere,soon,sorry,specified,specify,specifying,still,sub,such,sup,sure,t,s,take,taken,tell,tends,th,than,thank,thanks,thanx,that,that,s,thats,the,their,theirs,them,themselves,then,thence,there,there,s,thereafter,thereby,therefore,therein,theres,thereupon,these,they,they,d,they,ll,they,re,they,ve,think,third,this,thorough,thoroughly,those,though,three,through,throughout,thru,thus,to,together,too,took,toward,towards,tried,tries,truly,try,trying,twice,two,un,under,unfortunately,unless,unlikely,until,unto,up,upon,us,use,used,useful,uses,using,usually,value,various,very,via,viz,vs,want,wants,was,wasn,t,way,we,we,d,we,ll,we,re,we,ve,welcome,well,went,were,weren,t,what,what,s,whatever,when,whence,whenever,where,where,s,whereafter,whereas,whereby,wherein,whereupon,wherever,whether,which,while,whither,who,who,s,whoever,whole,whom,whose,why,will,willing,wish,with,within,without,won,t,wonder,would,would,wouldn,t,yes,yet,you,you,d,you,ll,you,re,you,ve,your,yours,yourself,yourselves,zero".split(',')
recover_list = {"wa":"was", "ha":"has"}
wl = nltk.WordNetLemmatizer()

def is_stopword(w):
    return w in stopwords.words('english') #stopwords_list
def lemmatize(w0):
    w = wl.lemmatize(w0.lower())
    #if w=='de': print w0, w
    if w in recover_list: return recover_list[w]
    return w

class Vocabulary:
    def __init__(self, excluds_stopwords=False):
        self.vocas = [] # id to word
        self.vocas_id = dict() # word to id
        self.docfreq = [] # id to document frequency
        self.excluds_stopwords = excluds_stopwords

    def term_to_id(self, term0):
        term = lemmatize(term0)
        if not re.match(r'[a-z]+$', term): return None
        if self.excluds_stopwords and is_stopword(term): return None
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
#            print str(voca_id) + ": " + term
            self.vocas_id[term] = voca_id
            self.vocas.append(term)
            self.docfreq.append(0)
        else:
            voca_id = self.vocas_id[term]
        return voca_id

    def doc_to_ids(self, doc):
        #print ' '.join(doc)
        list = []
        words = dict()
        for term in doc:
            id = self.term_to_id(term)
            if id != None:
                list.append(id)
                if not words.has_key(id):
                    words[id] = 1
                    self.docfreq[id] += 1
        if "close" in dir(doc): doc.close()
        return list

    def doc_to_ids_each_sentence(self, doc):
        #print ' '.join(doc)
        sent_list = []
        words = dict()
        
        for sent in doc:
            list = []
            for term in sent:
                id = self.term_to_id(term)
                if id != None:
                    list.append(id)
                    if not words.has_key(id):
                        words[id] = 1
                        self.docfreq[id] += 1
            sent_list.append(list)
        if "close" in dir(doc): doc.close()
        return sent_list

    def cut_low_freq(self, corpus, threshold=1):
        new_vocas = []
        new_docfreq = []
        self.vocas_id = dict()
        conv_map = dict()
        for id, term in enumerate(self.vocas):
            freq = self.docfreq[id]
            if freq > threshold:
                new_id = len(new_vocas)
                self.vocas_id[term] = new_id
                new_vocas.append(term)
                new_docfreq.append(freq)
                conv_map[id] = new_id
        self.vocas = new_vocas
        self.docfreq = new_docfreq

        def conv(doc):
            new_doc = []
            for id in doc:
                if id in conv_map: new_doc.append(conv_map[id])
            return new_doc
        return [conv(doc) for doc in corpus]

    def __getitem__(self, v):
        return self.vocas[v]

    def size(self):
        return len(self.vocas)

    def is_stopword_id(self, id):
        return self.vocas[id] in stopwords_list
