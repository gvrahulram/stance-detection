# -*- coding: utf-8 -*-
import re
import csv
from textblob import TextBlob

infile = 'test3.csv'
count = 0
neg = 0
pos = 0
nu = 0
x = []
y = []
#data cleaning and pre-processing 
with open(infile, 'r') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        tweet = row[0]
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
        #remove numbers
        tweet = re.sub(r'\d', "", tweet)
        #Convert @username to ''
        tweet = re.sub('@[^\s]+','',tweet)
        #Remove additional white spaces
        tweet = re.sub('[\s]+', ' ', tweet)
        #Replace #word with word
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        #trim
        tweet = tweet.strip('\'"')
        #lower
        tweet = tweet.lower()
        
	#finding polarity
        blob = TextBlob(tweet)
	
        #x = x.append(blob.sentiment.polarity) 
        if(blob.sentiment.polarity > 0):
            pos = pos + 1
            
        elif(blob.sentiment.polarity < 0):
            neg = neg + 1
          
        else:
            nu = nu + 1
         
        count = count + 1

print(count)
print("positive",pos)
print("nuetral",nu)
print("negative",neg)
