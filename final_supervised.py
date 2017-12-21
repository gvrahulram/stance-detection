# -*- coding: utf-8 -*-
import numpy as np
import nltk 
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
filename = "TrainDataA.txt"
tweets_favor = ''
tweets_against = ''
tweets_none = ''
tweets_all = ''
#file_obj = open(filename, "r")
#print file_obj

punctuations = list(string.punctuation)
with open(filename, "r") as ins:
    array = []
    for line in ins:
        array.append(line)
ins.close();

length = len(array) - 1
main_array = array[1: length]

tweet_stance_arr = []
all_words = []

allwords_str = ''
for i in range(len(main_array)):
    temp = []
    line = ''
    tempstr = ''
    line = main_array[i]
    sample_array = []
    sample_array = line.split("\t")
    sample_array[3] = sample_array[3].strip()
    tempstr = tempstr + sample_array[2]
    temptokenized = [i for i in word_tokenize(tempstr) if i not in punctuations]
    allwords_str += tempstr
    temp.append(temptokenized)
    temp.append(sample_array[3])
    tweet_stance_arr.append(temp)  
    
all_words = [i for i in word_tokenize(allwords_str) if i not in punctuations]
all_words = nltk.FreqDist(all_words)
word_features = all_words.keys()

def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in tweet_words)
    return features

training_set = nltk.classify.util.apply_features(extract_features, tweet_stance_arr)
#classifier = NaiveBayesClassifier.train(training_set)

NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

# Test the classifier
testTweet = 'He who exalts himself shall be humbled; and he who humbles himself shall be exalted.Matt 23:12.     #SemST'
temptokenized_testtweet = [i for i in word_tokenize(testTweet) if i not in punctuations]
#processedTestTweet = processTweet(testTweet)
#print (NBClassifier.classify(extract_features))