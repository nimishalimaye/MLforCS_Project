import pandas as pd
import numpy as np
import itertools
import sklearn
import sklearn.datasets as skd
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import OrderedDict
from operator import itemgetter 

df = pd.read_csv('fake_real_tweets.csv')
y = df.label
df = df.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, shuffle = True)

list_words = ['http', 'https', 'twitter','com','www']
count_vectorizer = CountVectorizer(stop_words=list_words)
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)
res_count = dict(zip(count_vectorizer.get_feature_names(),mutual_info_classif(count_train, y_train) ))
MI_count = sorted(res_count.items(), key=itemgetter(1), reverse=True)

feature_tokens = []
features = []
N=1000

tweets = "'Today is Donald Trump's Birthday! Send him your B'day wishes here: http://www.facebook.com/DonaldTrump'"
### add your rnn output here ###
file = open('test_tweets.csv','a')
file.write(tweets) # replace tweet with your generated tweet
file.close()


df = pd.read_csv('test_tweets.csv')
x_test = df['text']
print("x_test ", x_test)

def discriminator(x_test):
	list_words = ['http', 'https', 'twitter','com','www']
	count_vectorizer = CountVectorizer(stop_words=list_words)
	count_train = count_vectorizer.fit_transform(X_train)
	count_test = count_vectorizer.transform(x_test)

	clf = SelectKBest(score_func = mutual_info_classif, k = N)	
	fit = clf.fit(count_train,y_train)
	count_x_train_ft = fit.transform(count_train)
	count_x_test_ft = fit.transform(count_test)

	svc_tfidf_clf = LinearSVC()
	svc_tfidf_clf.fit(count_x_train_ft, y_train)
	pred = svc_tfidf_clf.predict(count_x_test_ft)

	print("Predicting ...")
	for i in range(np.shape(pred)[0]):
		if(pred[i] == 0):
			print('real')
		else:
			print('fake')

discriminator(x_test)
