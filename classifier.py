import pandas as pd
import numpy as np
import pickle
import itertools
import sklearn
import os
import csv
import enchant
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
print('loaded data')
y = df.label
df = df.drop('label', axis=1)
indices = df.index.get_values()
x_train, x_main_test, y_train, y_main_test, train_indices, test_indices = train_test_split(df['text'], y, indices, test_size=0.33, shuffle=True)

stop_words = ['http', 'https', 'twitter', 'com', 'www']

print('learning tfidf')
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
clf = SelectKBest(score_func=mutual_info_classif, k=1000)
tfidf_fit = clf.fit(tfidf_train, y_train)
tfidf_x_train_ft = tfidf_fit.transform(tfidf_train)
print('done tfidf')

print('learning count vectorizer')
count_vectorizer = CountVectorizer(stop_words=stop_words)
count_train = count_vectorizer.fit_transform(x_train)
clf = SelectKBest(score_func=mutual_info_classif, k=1000)
count_fit = clf.fit(count_train, y_train)
count_x_train_ft = count_fit.transform(count_train)
print('done count vectorizer')

print('learning mn count')
mn_count_clf = MultinomialNB()
mn_count_clf.fit(count_x_train_ft, y_train)
print('done mn count')

print('learning passive aggressive')
pa_tfidf_clf = PassiveAggressiveClassifier(max_iter=50, C=0.01)
pa_tfidf_clf.fit(count_x_train_ft, y_train)
print('done passive aggressive')

print('learning svc')
svc_tfidf_clf = LinearSVC()
svc_tfidf_clf.fit(count_x_train_ft, y_train)
print('done svc')

print('learning mn tfidf')
mn_tfidf_clf = MultinomialNB()
mn_tfidf_clf.fit(tfidf_x_train_ft, y_train)
print('done mn tfidf')

print('learning sgd')
sgd_tfidf_clf = SGDClassifier(max_iter=50)
sgd_tfidf_clf.fit(count_x_train_ft, y_train)
print('done sgd')
print('done learning')


def discriminator(x_test, y_test):
	res = {}
	
	count_test = count_vectorizer.transform(x_test)
	count_x_test_ft = count_fit.transform(count_test)
	
	
	tfidf_test = tfidf_vectorizer.transform(x_test)
	tfidf_x_test_ft = tfidf_fit.transform(tfidf_test)
	
	#print("MultinomialNB CountVectorizer")
	pred = mn_count_clf.predict(count_x_test_ft)
	score = metrics.accuracy_score(y_test, pred)
	#print("accuracy:   %0.3f" % score)
	#print(" ")
	res['MultinomialNB CountVectorizer'] = score
	
	#print("MultinomialNB TfidfVectorizer")
	pred = mn_tfidf_clf.predict(tfidf_x_test_ft)
	score = metrics.accuracy_score(y_test, pred)
	#print("accuracy:   %0.3f" % score)
	#print(" ")
	res['MultinomialNB TfidfVectorizer'] = score

	#print("PassiveAggressiveClassifier: C = 0.01")
	pred = pa_tfidf_clf.predict(count_x_test_ft)
	score = metrics.accuracy_score(y_test, pred)
	#print("accuracy:   %0.3f" % score)
	#print(" ")
	res["PassiveAggressiveClassifier: C = 0.01"] = score
	
	#print("LinearSVC: C = 0")
	pred = svc_tfidf_clf.predict(count_x_test_ft)
	score = metrics.accuracy_score(y_test, pred)
	#print("accuracy:   %0.3f" % score)
	#print(" ")
	res["LinearSVC: C = 0"] = score
	
	#print("SGDClassifier")
	pred = sgd_tfidf_clf.predict(count_x_test_ft)
	score = metrics.accuracy_score(y_test, pred)
	#print("accuracy:   %0.3f" % score)
	#print(" ")
	res["SGDClassifier"] = score
	
	return res


main_data_res = discriminator(x_main_test, y_main_test) #main data


english_dict = enchant.DictWithPWL('en_US')
def is_in_vocab(w):
	try:
		w = w.replace(",", "").replace(".", "").replace(":", "").replace('!', "").replace("'", "").lower()
		if w == '': return False
		if w in ["donald", "trump", "hillary", "obama", "@realDonaldTrump"]: return True
		return english_dict.check(w) or english_dict.check(w.capitalize())
	except:
		return False


def rnn_attack():
	rows = []
	thres = 0.4
	for iter in range(2180):
		try:
			df = pd.read_csv('/Users/pazgrimberg/Documents/GitHub/mlcs/final_project/rnn_tweets_'+str(iter)+'.csv', encoding='utf-8') #tweet,label
		except:
			continue
		df['pct_in_english'] = df['tweet'].apply(lambda T: sum([is_in_vocab(tw) for tw in T.split(' ')])/float(len(T.split(' '))) if not pd.isnull(T) else 0)
		x_test = df[df['pct_in_english']>thres]['tweet']
		y_test = [1]*len(x_test)
		num_tweets = len(df)
		num_garbage_tweets = len(df[df['pct_in_english']<=thres])
		num_tweets_tested = num_tweets - num_garbage_tweets
		
		if num_tweets_tested == 0: continue
		
		res = discriminator(x_test, [1] * len(x_test))
		for clsfer in res.keys():
			real_acc = res[clsfer]
			num_predicted_right = num_tweets_tested*res[clsfer]
			adj_acc = (num_predicted_right + num_garbage_tweets)/float(num_tweets)
			row = [iter,clsfer,num_tweets_tested, num_predicted_right,real_acc, adj_acc]
			rows.append(row)
		if iter % 100 == 0: print('------------ at iter %s'%iter)
	res_df = pd.DataFrame(columns=['iter','method','num_tweets_tested','num_tweets_predicted_right','real_acc','adj_acc'],data=rows)
	
	res_df.to_csv("final_result.csv")


def vocab_attack():
	thres = 0.4
	rows=[]
	df = pd.read_csv("/Users/pazgrimberg/fake_gen_tweets.csv",encoding='utf-8')
	df['pct_in_english'] = df['tweet'].apply(lambda T: sum([is_in_vocab(tw) for tw in T.split(' ')]) / float(len(T.split(' '))) if not pd.isnull(T) else 0)
	x_test = df[df['pct_in_english'] > thres]['tweet']
	num_tweets = len(df)
	num_garbage_tweets = len(df[df['pct_in_english'] <= thres])
	num_tweets_tested = num_tweets - num_garbage_tweets
	
	res = discriminator(x_test, [1] * len(x_test))
	for clsfer in res.keys():
		real_acc = res[clsfer]
		num_predicted_right = num_tweets_tested * res[clsfer]
		adj_acc = (num_predicted_right + num_garbage_tweets) / float(num_tweets)
		row = [0, clsfer, num_tweets_tested, num_predicted_right, real_acc, adj_acc]
		rows.append(row)
	res_df = pd.DataFrame(columns=['iter','method','num_tweets_tested','num_tweets_predicted_right','real_acc','adj_acc'],data=rows)
	res_df.to_csv('vocab_attack.csv')
	print(res_df)


vocab_attack()

# def new_test_data(X_test, y_test):
# 	x_new_test = []
# 	y_new_test = []
#
# 	count_fake = 0
# 	count_total = 0
# 	for j in range(100):  # np.shape(X_test)[0]
# 		count = 0
# 		print(j)
# 		rnn_words = X_test[j].replace(",", "").replace(".", "").replace(":", "").replace('!', "").replace("'", "").lower().split()
# 		imp_words = ["donald", "trump", "hillary", "obama", "@realDonaldTrump"]
# 		for i in range(np.shape(rnn_words)[0]):
# 			if (((rnn_words[i] in words.words()) == True) or ((rnn_words[i] in imp_words) == True)):
# 				count += 1
# 		if (count / np.shape(rnn_words)[0] < 0.40):
# 			count_total += 1
# 			if (y_test[j] == 1):
# 				count_fake += 1
# 		else:
# 			x_new_test.append(X_test[j])
# 			y_new_test.append(y_test[j])
# 	acc = count_fake / count_total
# 	return x_new_test, y_new_test, count_fake, count_total




# final_score = (score * np.shape(x_test)[0] + count_fake) / (np.shape(x_test)[0] + count_total)
# print("final_acc: ", final_score)
#
#
# j = 0
# x_test = []
# y = []
# for i in test_indices:
# 	x_test.append(X_test[i])
# 	y.append(y_test[i])
# 	j += 1
#
# x_new_test, y_new_test, count_fake, count_total = new_test_data(x_tweets, y_tweets)
# print(np.shape(x_new_test)[0])
# print("count_fake: ", count_fake)
# discriminator(x_new_test, y_new_test, count_fake, count_total)
