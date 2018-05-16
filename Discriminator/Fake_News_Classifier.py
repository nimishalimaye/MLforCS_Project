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
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33)

list_words = ['http', 'https', 'twitter','com','www']
count_vectorizer = CountVectorizer(stop_words=list_words)
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)
#print("num of features for count_vectorizer: ", np.shape(count_test))
res_count = dict(zip(count_vectorizer.get_feature_names(),mutual_info_classif(count_train, y_train) ))
MI_count = sorted(res_count.items(), key=itemgetter(1), reverse=True)
# print("MI_Count: ",MI_count[:10])

tfidf_vectorizer = TfidfVectorizer(stop_words=list_words, max_df = 0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

res_tfidf = dict(zip(count_vectorizer.get_feature_names(),mutual_info_classif(tfidf_train, y_train) ))
MI_tfidf = sorted(res_tfidf.items(), key=itemgetter(1), reverse=True)
# print("MI_tfidf: ",MI_tfidf[:10])
#using mutual information as a replacement for information gain, since the formula is same for both.

### skd.load assigns fake tweets as 0 and real tweets as 1.
# ls_train = skd.load_files('./trump_data/train');
# ls_test = skd.load_files('./trump_data/test');

#The count vectorizer classes fit_transform function generates a vocabulary that contains each unique term in the dataset
#and outputs a sparse matrix tabulating term occurrences
accuracy_svc = []
# features = []
num = [10, 100, 1000]
# num = [33188]
for N in (num):
# for N in range(10, 33188, 100):
	
	print("Number of features: ", N)	
	feature_tokens = []
	features = []
	# print("CountVectorizer Features")
	# create and print the list of the feature dictionary tokens and their corresponding MI scores
	for (j,token) in enumerate(MI_count):
	    if(j<N):
	        feature_tokens.append(token)
	    else:
	        break

	#write feature_tokens to file
	#write_tokens_to_file(feature_tokens, feature_dictionary_dir)

	#### Uncomment the two lines below to see the features in the notebook itself
	features = [item[0] for item in feature_tokens]
	print(features)

	features_value = [item[1] for item in feature_tokens]
	print(features_value)

	feature_tokens = []
	features = []
	print(" ")
	# print("TfidfVectorizer Features")
	# create and print the list of the feature dictionary tokens and their corresponding MI scores
	for (j,token) in enumerate(MI_tfidf):
	    if(j<N):
	        feature_tokens.append(token)
	    else:
	        break

	#write feature_tokens to file
	#write_tokens_to_file(feature_tokens, feature_dictionary_dir)

	#### Uncomment the two lines below to see the features in the notebook itself
	features = [item[0] for item in feature_tokens]
	print(features)
	features_value = [item[1] for item in feature_tokens]
	print(features_value)

	clf = SelectKBest(score_func = mutual_info_classif, k = N)	
	fit = clf.fit(count_train,y_train)
	count_x_train_ft = fit.transform(count_train)
	count_x_test_ft = fit.transform(count_test)
	# clf = SelectKBest(score_func = mutual_info_classif, k = N)	
	# fit = clf.fit(count_test,y_test)
	# count_x_test_ft = fit.transform(count_test)

	clf = SelectKBest(score_func = mutual_info_classif, k = N)	
	fit = clf.fit(tfidf_train,y_train)
	tfidf_x_train_ft = fit.transform(tfidf_train)
	tfidf_x_test_ft = fit.transform(tfidf_test) 
	# clf = SelectKBest(score_func = mutual_info_classif, k = N)	
	# fit = clf.fit(tfidf_test,y_test)
	# tfidf_x_test_ft = fit.transform(tfidf_test)

	# I will compare the following models (and training data):

	# - multinomialNB with counts (mn_count_clf)
	# - multinomialNB with tf-idf (mn_tfidf_clf)
	# - passive aggressive with tf-idf (pa_tfidf_clf)
	# - linear svc with tf-idf (svc_tfidf_clf)
	# - linear sgd with tf-idf (sgd_tfidf_clf)
	# - multinomialNB with hash (hash_clf)

	print("MultinomialNB CountVectorizer")
	mn_count_clf = MultinomialNB()
	mn_count_clf.fit(count_x_train_ft, y_train)
	pred = mn_count_clf.predict(count_x_test_ft)
	score = metrics.accuracy_score(y_test, pred)
	print("accuracy:   %0.3f" % score)
	print(" ")

	print("MultinomialNB TfidfVectorizer")
	mn_tfidf_clf = MultinomialNB()
	mn_tfidf_clf.fit(tfidf_x_train_ft, y_train)
	pred = mn_tfidf_clf.predict(tfidf_x_test_ft)
	score = metrics.accuracy_score(y_test, pred)
	print("accuracy:   %0.3f" % score)
	print(" ")

	print("PassiveAggressiveClassifier")
	pa_tfidf_clf = PassiveAggressiveClassifier(max_iter=50)
	pa_tfidf_clf.fit(tfidf_x_train_ft, y_train)
	pred = pa_tfidf_clf.predict(tfidf_x_test_ft)
	score = metrics.accuracy_score(y_test, pred)
	print("accuracy:   %0.3f" % score)
	print(" ")

	print("LinearSVC")
	svc_tfidf_clf = LinearSVC()
	svc_tfidf_clf.fit(tfidf_x_train_ft, y_train)
	pred = svc_tfidf_clf.predict(tfidf_x_test_ft)
	score = metrics.accuracy_score(y_test, pred)
	print("accuracy:   %0.3f" % score)
	print(" ")

	# accuracy_svc.append(score)
	# features.append(N)
	print("SGDClassifier")
	sgd_tfidf_clf = SGDClassifier(max_iter = 50)
	sgd_tfidf_clf.fit(tfidf_x_train_ft, y_train)
	pred = sgd_tfidf_clf.predict(tfidf_x_test_ft)
	score = metrics.accuracy_score(y_test, pred)
	print("accuracy:   %0.3f" % score)
	print(" ")

	#Comparing the above classifiers using a plot
	# plt.figure(0).clf()

	# for model, name in [ (mn_count_clf, 'multinomial nb count'),
	#                      (mn_tfidf_clf, 'multinomial nb tfidf'),
	#                      (pa_tfidf_clf, 'passive aggressive'),
	#                      (svc_tfidf_clf, 'svc'),
	#                      (sgd_tfidf_clf, 'sgd')]:
	#     if 'count' in name:
	#         pred = model.predict_proba(count_x_test_ft)[:,1]
	#     elif 'multinomial' in name:
	#         pred = model.predict_proba(tfidf_x_test_ft)[:,1]
	#     else: 
	#         pred = model.decision_function(tfidf_x_test_ft)
	#     fpr, tpr, thresh = metrics.roc_curve(y_test.values, pred)
	#     plt.plot(fpr,tpr,label="{}".format(name))

	# plt.legend(loc=0)
	# plt.show()

# plt.plot(features, accuracy_svc)
# plt.show()