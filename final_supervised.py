# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from gensim import parsing
import gensim
import numpy as np
import pandas as pd
from pandas import DataFrame
from gensim.parsing.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
#Load data
traindata = pd.read_csv('train.csv')

#Finding the number of unique Stances
print (traindata['Stance'].unique())

#converting Author names to Indices
#EAP = 0 HPL = 1 MWS = 2


#copy traindata into a temporary Dataframe as backup
backuptrain = traindata

traindata.is_copy = False 

#Replace author names with integers
traindata.replace({'AGAINST':0, 'FAVOR':1, 'NONE':2}, inplace=True)



#Print for testing
#print ((train.iloc[0][1]))

#data cleaning and pre-reprocessing

def preprocess(text):
    #convert text to lower case
    text = text.lower()
   
    #removing whitespace
    text.strip()
   
    #removing digits
    text = gensim.parsing.preprocessing.strip_numeric(text)
    #text = ' '.join(s for s in text.split() if not any(c.isdigit() for c in s))
    
    #print(text)
    
    
    #remove stopwords
    text = gensim.parsing.preprocessing.remove_stopwords(text)
    
    #strip punctutation
    text = gensim.parsing.preprocessing.strip_punctuation2(text)
    
    #strip multiple whitepsace that might occur after we remove stopwords
    text = gensim.parsing.preprocessing.strip_multiple_whitespaces(text)

    p = PorterStemmer()
    
    text = ' '.join(p.stem(word) for word in text.split())    

    #print(text)
    
    return text



#preprocessing the data
traindata['Tweet'] = traindata['Tweet'].map(preprocess)

#Split into training and Validation
train, test = train_test_split(traindata, test_size=0.2)

#For ignoring SettingWithCopyWarning, Usually comes up as a warning that we are operating on a copy of the main dataframe
train.is_copy = False

train_excerpts = train['Tweet']

train_labels = train['Stance']

## Get the word vocabulary out of the data
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_excerpts)
#print(DataFrame(X_train_counts.A, columns=count_vect.get_feature_names()).to_string())
#print (X_train_counts)

## Count of 'mistak' in corpus (mistake -> mistak after stemming)
#print ('mistak appears:', count_vect.vocabulary_.get('mistak') , 'in the corpus')

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print ('Dimension of TF-IDF vector :' , X_train_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, train['Stance'])

X_new_counts = count_vect.transform(test['Tweet'])
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test['Stance'],predicted,labels=[0,1,2])

tpFavor = 0
tpAgainst = 0
tpNone = 0
fpFAVORtoAGAINST = 0
fpFAVORtoNONE = 0
fpAGAINSTtoFAVOR = 0
fpAGAINSTtoNONE = 0
fpNONEtoFAVOR = 0
fpNONEtoAGAINST = 0



tpFavor = cm[0,0]
tpAgainst = cm[1,1]
tpNone = cm[2,2]
tnFavor = tpAgainst + tpNone;
tnAgainst = tpFavor + tpNone;
tnNone = tpFavor + tpAgainst;
fnFavor = cm[1,0] +cm[2,0];
fnAgainst = cm[0,1] + cm[2,1];
fnNone = cm[0,2] + cm[1,2];
fpFavor = cm[0,1] + cm[0,2];
fpAgainst = cm[1,0] + cm[1,2];
fpNone = cm[2,0] + cm[2,1];

TotalPredictedFAVOR = tpFavor + cm[0,1] + cm[0,2]
TotalPredictedAGAINST = tpAgainst + cm[1,0] + cm[1,2]
TotalPredictedNONE = tpNone + cm[2,0] + cm[2,1]

TotalGoldFavor = tpFavor + cm[1,0] + cm[2,0]
TotalGoldAgainst = cm[0,1] + tpAgainst + cm[2,1]
TotalGoldNone = cm[0,2] + cm[1,2] + tpNone

accuracyFavor = (tpFavor + tnFavor)/(tpFavor+tnFavor+fpFavor+fnFavor); 
precisionFavor = tpFavor / TotalPredictedFAVOR
recallFavor = tpFavor / TotalGoldFavor
f1Favor = (2 * precisionFavor * recallFavor) / (precisionFavor + recallFavor)

accuracyAgainst = (tpAgainst + tnAgainst)/(tpAgainst+tnAgainst+fpAgainst+fnAgainst);
precisionAgainst = tpAgainst / TotalPredictedAGAINST
recallAgainst = tpAgainst / TotalGoldAgainst
f1Against = (2 * precisionAgainst * recallAgainst) / (precisionAgainst + recallAgainst)

accuracyNone = (tpNone + tnNone)/(tpNone + tnNone + fpNone + fnNone);
precisionNone = tpNone / TotalPredictedNONE
recallNone = tpNone / TotalGoldNone
f1None = (2 * precisionNone * recallNone) / (precisionNone + recallNone)   