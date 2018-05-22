import pandas as pd
import numpy as np
import pickle
import itertools
import sklearn
import os
import csv
import os.path
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
from nltk.corpus import words

df = pd.read_csv('fake_real_tweets.csv')
y = df.label
df = df.drop('label', axis=1)
indices = df.index.get_values()
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(df['text'], y, indices, test_size=0.33, shuffle=True)


def discriminator(tweet_list, tweet_list_y, count_fake, count_total):
	list_words = ['http', 'https', 'twitter', 'com', 'www']
	count_vectorizer = CountVectorizer(stop_words=list_words)
	count_train = count_vectorizer.fit_transform(X_train)
	count_test = count_vectorizer.transform(tweet_list)
	
	tfidf_vectorizer = TfidfVectorizer(stop_words=list_words, max_df=0.7)
	tfidf_train = tfidf_vectorizer.fit_transform(X_train)
	tfidf_test = tfidf_vectorizer.transform(X_test)
	
	clf = SelectKBest(score_func=mutual_info_classif, k=1000)
	fit = clf.fit(count_train, y_train)
	count_x_train_ft = fit.transform(count_train)
	count_x_test_ft = fit.transform(count_test)
	
	clf = SelectKBest(score_func=mutual_info_classif, k=1000)
	fit = clf.fit(tfidf_train, y_train)
	tfidf_x_train_ft = fit.transform(tfidf_train)
	tfidf_x_test_ft = fit.transform(tfidf_test)
	
	print("MultinomialNB CountVectorizer")
	mn_count_clf = MultinomialNB()
	mn_count_clf.fit(count_x_train_ft, y_train)
	pred = mn_count_clf.predict(count_x_test_ft)
	score = metrics.accuracy_score(tweet_list_y, pred)
	print("accuracy:   %0.3f" % score)
	print(" ")
	final_score = (score * np.shape(tweet_list)[0] + count_fake) / (np.shape(tweet_list)[0] + count_total)
	print("final_acc: ", final_score)
	
	print("MultinomialNB TfidfVectorizer")
	mn_tfidf_clf = MultinomialNB()
	mn_tfidf_clf.fit(tfidf_x_train_ft, y_train)
	pred = mn_tfidf_clf.predict(tfidf_x_test_ft)
	score = metrics.accuracy_score(y_test, pred)
	print("accuracy:   %0.3f" % score)
	print(" ")
	final_score = (score * np.shape(tweet_list)[0] + count_fake) / (np.shape(tweet_list)[0] + count_total)
	print("final_acc: ", final_score)
	
	print("PassiveAggressiveClassifier: C = 0.01")
	pa_tfidf_clf = PassiveAggressiveClassifier(max_iter=50, C=0.01)
	pa_tfidf_clf.fit(count_x_train_ft, y_train)
	pred = pa_tfidf_clf.predict(count_x_test_ft)
	score = metrics.accuracy_score(tweet_list_y, pred)
	print("accuracy:   %0.3f" % score)
	print(" ")
	final_score = (score * np.shape(tweet_list)[0] + count_fake) / (np.shape(tweet_list)[0] + count_total)
	print("final_acc: ", final_score)
	
	print("LinearSVC: C = 0")
	svc_tfidf_clf = LinearSVC()
	svc_tfidf_clf.fit(count_x_train_ft, y_train)
	pred = svc_tfidf_clf.predict(count_x_test_ft)
	score = metrics.accuracy_score(tweet_list_y, pred)
	print("accuracy:   %0.3f" % score)
	print(" ")
	final_score = (score * np.shape(tweet_list)[0] + count_fake) / (np.shape(tweet_list)[0] + count_total)
	print("final_acc: ", final_score)
	
	print("SGDClassifier")
	sgd_tfidf_clf = SGDClassifier(max_iter=50)
	sgd_tfidf_clf.fit(count_x_train_ft, y_train)
	pred = sgd_tfidf_clf.predict(count_x_test_ft)
	score = metrics.accuracy_score(tweet_list_y, pred)
	print("accuracy:   %0.3f" % score)
	print(" ")
	final_score = (score * np.shape(tweet_list)[0] + count_fake) / (np.shape(tweet_list)[0] + count_total)
	print("final_acc: ", final_score)


df = pd.read_csv('tweets_11M_labelled.csv')
x_tweets = df['tweet']
y_tweets = df['label']


def new_test_data(X_test, y_test):
	x_new_test = []
	y_new_test = []
	
	count_fake = 0
	count_total = 0
	for j in range(100):  # np.shape(X_test)[0]
		count = 0
		print(j)
		rnn_words = X_test[j].replace(",", "").replace(".", "").replace(":", "").replace('“', "").replace('”', "").replace('!', "").replace("'", "").lower().split()
		imp_words = ["donald", "trump", "hillary", "obama", "@realDonaldTrump"]
		for i in range(np.shape(rnn_words)[0]):
			if (((rnn_words[i] in words.words()) == True) or ((rnn_words[i] in imp_words) == True)):
				count += 1
		if (count / np.shape(rnn_words)[0] < 0.40):
			count_total += 1
			if (y_test[j] == 1):
				count_fake += 1
		else:
			x_new_test.append(X_test[j])
			y_new_test.append(y_test[j])
	acc = count_fake / count_total
	return x_new_test, y_new_test, count_fake, count_total


j = 0
x_test = []
y = []
for i in test_indices:
	x_test.append(X_test[i])
	y.append(y_test[i])
	j += 1

x_new_test, y_new_test, count_fake, count_total = new_test_data(x_tweets, y_tweets)
print(np.shape(x_new_test)[0])
print("count_fake: ", count_fake)
discriminator(x_new_test, y_new_test, count_fake, count_total)

# import pandas as pd
# import os
# all_rows = []
# max_file_num = 2000
# for i in range(max_file_num):
# 	if i % 100 == 0: print('reading file %s'%i)
# 	if os.path.exists('/Users/pazgrimberg/Documents/GitHub/mlcs/final_project/rnn_tweets_'+ str(i) + ".csv"):
# 		try:
# 			k_df = pd.read_csv('/Users/pazgrimberg/Documents/GitHub/mlcs/final_project/rnn_tweets_'+ str(i) + ".csv", encoding='utf-8')
# 			all_rows += [(r['iter'],r['tweet']) for (_, r) in k_df.iterrows()]
# 		except:
# 			print('skipped %s'%i)
# 			continue
# pd.DataFrame(columns=['iter','tweet'], data=all_rows).to_csv('/Users/pazgrimberg/Documents/GitHub/mlcs/final_project/all_tweets.csv', encoding='utf-8')
# print('done')

