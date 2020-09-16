#!/usr/bin/env python3
import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

lemmer=nltk.stem.WordNetLemmatizer()


f=open('info.txt','r',errors='ignore')
raw=f.read()
raw=raw.lower()
sentTokens=nltk.sent_tokenize(raw)
wordTokens=nltk.word_tokenize(raw)
#print(sentTokens[:2])
#print(wordTokens[:2])

def lemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

punct_dict=dict((ord(punct),None) for punct in string.punctuation)

def lemNormalize(text):
    return lemTokens(nltk.word_tokenize(text.lower().translate(punct_dict)))

greetingsInput=("hello","hi","what's up","hey")
greetingsResponse=["hey dude","hey favourite","good to see your face","Hiiiiiiiiiii hyped to see you"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greetingsInput:
            return random.choice(greetingsResponse)

def response(userResponse):
    freya_response=''
    sentTokens.append(userResponse)
    #tfidVec=TfidfVectorizer(tokenizer=lemNormalize,stop_words='english')
    #tfidf=tfidVec.fit_transform(sentTokens)
    cv=CountVectorizer(max_features=50,tokenizer=lemNormalize,analyzer='word')
    x=cv.fit_transform(sentTokens)
    vals=cosine_similarity(x[-1],x)
    idx=vals.argsort()[0][-2]
    flat=vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]

    if(req_tfidf==0):
        freya_response="Can't figure out that one!! Ask another if you will!?"
        return freya_response
    else:
        freya_response=freya_response+sentTokens[idx]
        return freya_response

flag=True
print("Hey I am Freya!! I might be able to get you some facts, If you want to exit, type bye!")
while(flag==True):
    userResponse=input()
    userResponse=userResponse.lower()
    if userResponse!='bye':
        if userResponse=='thanks' or userResponse=='thank you':
            flag=False
            print("You are welcome")
        else:
            if greeting(userResponse)!=None:
                print(greeting(userResponse))
            else:
                print("Freya: ",end="")
                print(response(userResponse))
                sentTokens.remove(userResponse)
    else:
        flag=False
        print("Freya going to sleep..bye")





