
# coding: utf-8

# In[1]:

from gensim import corpora
from gensim.models import Word2Vec, Doc2Vec
import gensim
import json
import cPickle as pickle
import itertools
import re
import stop_words
import pprint as pprint
from joblib import Parallel, delayed
import multiprocessing
from math import sqrt

#from functions import getJSONdata

# In[2]:

#stoplist_dk=pickle.load(open("stop_list_dk", 'rb'))
#lille_alfabet = u'abcdefghijklmnopqrstuvwxyz\u00e6\u00F8\u00e5 1234567890'.encode('utf-8')
lille_alfabet = u'abcdefghijklmnopqrstuvwxyz?!& 1234567890'.encode('utf-8')

def danskebogstavertiltegn(instring):
    return instring.replace('å', '?').replace('æ', '!').replace('ø', '&').replace('ü', 'u').replace('ö', 'o').replace('ä', 'a').replace('Å','?').replace('Æ','!').replace('Ø','&')

def clean_post(post):
    post=post.replace('\n','.').replace('\r','.')
    post=post.lower()
    post = post.replace('!', '.').replace('?','.')
    post = re.sub('[[(){}<>,:\';\-/\@£#%&=|½§\*]]?', '', post)
    post = re.sub('[1234567890]?', '', post)
    #post = str(post.replace('å', '\u00e5').replace('æ', '\u00e6').replace('ø', '\u00e8'))
    post = str(danskebogstavertiltegn(post))#str(post.replace('å', 'aa').replace('æ', 'ae').replace('ø', 'oe').replace('ü','u').replace('ö','oe').replace('ä','ae'))
    #post = str(post.replace('Å', 'aa').replace('Æ', 'ae').replace('Ø', 'oe'))
    #post = str(post.replace(u'\u00e5', 'aa').replace(u'\u00e6', 'ae').replace(u'\u00e8', 'oe'))
    post = ''.join(c for c in post if c in lille_alfabet)
    return post

def clean_post_w_upper(post):
    word_list=[]
    post=post.replace('\n',' ').replace('\r',' ')
    post = ''.join(c for c in post if c in lille_alfabet + lille_alfabet.upper())
    #post = str(post.replace(u'å','\u00e5').replace(u'æ','\u00e6').replace(u'ø','\u00e8'))
    return post

def clean_post_list(post):
    post = clean_post(post)
    words=post.split(' ') #HER GÅR DET JO HELT GALT!!!!
    return [w for w in words if (len(w)>0 and len(w)<17)]

def split_sentences(s):
    return re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?! [a-z])(?<=\.|\?|\!|:)\s',s)

def division(big,small):
    if (big==0):
        return 0
    else:
        if small==0:
            small=1
        return big/small
#print stoplist_dk

def splitInComments(i,comments):
    return list(itertools.chain(*[split_sentences(com) for com in comments[comments.keys()[i]]]))
# In[3]:

readInData = False
askwritecommentstsfile = True
readInTexts = True
makeWord2VecModel = False
makeDoc2VecModel = False
#data = getJSONdata('N:\Appldata\Medieforskning\_Ask\comments_unge.json')
#comments1 = json.load(open("N:\Appldata\Medieforskning\_Ask\comments_unge.json","rb"))
#comments = json.load(open("C:\LokaleProjekter\keras\Dansk Folkeparti_year_react_and_com_0.json","rb"))
comments = json.load(open("C:\\LokaleProjekter\word2vec\\Pol_comments_block","rb"))
posts = json.load(open("C:\\LokaleProjekter\word2vec\\Pol_posts_block","rb"))
#comments = json.load(open("C:\LokaleProjekter\keras\Dansk Folkeparti_year_react_and_com_0.json","rb"))
#comment_list = list(itertools.chain(* [split_sentences(com) for com in list(itertools.chain(* [comments[page] for page in comments.keys()]))]))

for party in comments:
    comment_list = []
    if(askwritecommentstsfile):
        comments = comments[party]+posts[party]
        comment_list_file = open('./'+party+'/list_of_comments.txt', 'w+')
        for icom in comments:
            for user in comments[icom]['comments_data']:
                comment_list.append(user['message'])
        for i in range(len(comments.keys())):
            comment_list = comment_list + splitInComments(i, comments)

        for item in comment_list:
            comment_list_file.write("%s\n" % item.encode('utf-8'))
        comment_list_file.close()

    # if(readInData):
    #     comment_list_file = open('./'+party+'/list_of_comments.txt', 'w+')
    #     #comment_list = list(itertools.chain(* [split_sentences(com) for com in list(itertools.chain(* [comments[page] for page in comments.keys()[0]]))])) #todo remove [0] to get all posts
    #
    #     #test = Parallel(n_jobs=1)(delayed(sqrt)(i ** 2) for i in range(10))
    #     for i in range(len(comments.keys())):
    #         comment_list = comment_list + splitInComments(i,comments)
    #
    #     for item in comment_list:
    #         comment_list_file.write("%s\n" % item.encode('utf-8'))
    #     comment_list_file.close()
        #  print >> comment_list_file, comment_list #todo remove [0] and [0:10000]to get all posts
        #   comment_list_file.close()
            #pickle.dump(comment_list,open('./comment_list_file.txt','w'))
        #comment_list  = open('comment_list_file.txt', 'r')
        #comment_list = pickle.load(open('./comment_list_file.txt','r'))

    comment_list = open('./'+party+'/list_of_comments.txt','r').readlines()

    texts = []
    if(readInTexts):
        #texts = [document for document in comment_list]  #
        texts = [word for word in [clean_post(document).split() for document in comment_list]] #
        #
        #
        texts = [text for text in texts if len(text) > 0]
        text_list_file = open('./'+party+'/cleaned_posts.txt', 'w+')
        for item in texts:
            text_list_file.write(' '.join(item)+'\n')
            #text_list_file.write("%s" % word)
            #text_list_file.write("\n")
        text_list_file.close()
