#Libraries
from __future__ import division
import numpy as np
import pandas as pd
import sklearn
import sklearn.datasets as skd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from os import listdir
from os.path import isfile, join
import re
import math
from collections import OrderedDict
from operator import itemgetter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


### skd.load assigns fake tweets as 0 and real tweets as 1.
ls_train = skd.load_files('./trump_data/train');
ls_test = skd.load_files('./trump_data/test');

df = pd.read_csv('fake_real_tweets.csv')
y = df.label
df = df.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33)

#The count vectorizer classes fit_transform function generates a vocabulary that contains each unique term in the dataset
#and outputs a sparse matrix tabulating term occurrences

count_vect = CountVectorizer()
x_train = count_vect.fit_transform(X_train)
# print(X_train[0])
# print(x_train[0])
#Since the vocabulary has already been learned, use the transform function to transform the test data using the same vocab
x_test = count_vect.transform(X_test)

#using mutual information as a replacement for information gain, since the formula is same for both.
res = dict(zip(count_vect.get_feature_names(),mutual_info_classif(x_train, y_train) ))
MI = sorted(res.items(), key=itemgetter(1), reverse=True)


########### Finding top N features ######################

# create and print the list of the feature dictionary tokens and their corresponding IG scores
# N = 1000
num = [10,100,1000]
for N in num:
	########### Finding top N features ######################
	#feature_dictionary_dir = "feature_dictionary_N_10.txt"
	feature_tokens = []
	features = []

	# create and print the list of the feature dictionary tokens and their corresponding MI scores
	for (j,token) in enumerate(MI):
	    if(j<N):
	        feature_tokens.append(token)
	    else:
	        break

	#write feature_tokens to file
	#write_tokens_to_file(feature_tokens, feature_dictionary_dir)

	#### Uncomment the two lines below to see the features in the notebook itself
	features = [item[0] for item in feature_tokens]
	# print(features)
	########### Finding top N features DONE ######################

	temp = x_train > 0
	x_train_bf = temp.astype(int)
	print("N is: ",N)
	########### Create training set with new features ##############
	clf1 = SelectKBest(score_func = mutual_info_classif, k = N)
	clf_fit1 = clf1.fit(x_train,y_train)
	x_train_ft = clf_fit1.transform(x_train)
	x_test_ft = clf_fit1.transform(x_test)

	clf3 = SelectKBest(score_func = mutual_info_classif, k = N)
	clf_fit3 = clf3.fit(x_train_bf,y_train)
	x_train_ft_bf = clf_fit3.transform(x_train_bf)  
	x_test_ft_bf = clf_fit3.transform(x_test) 

	###################################### Train a multinomial NB TF classifer ##########################################
	mNomTF = sklearn.naive_bayes.MultinomialNB();
	fit = mNomTF.fit(x_train_ft,y_train);
	x_test_pred_mNomTF = mNomTF.predict(x_test_ft)

	#Test the accuracy of the trained classifier and find precision and recall

	acc_mNomTF = mNomTF.score(x_test_ft,y_test);
	precision_mNomTF,recall_mNomTF,fscore,support = sklearn.metrics.precision_recall_fscore_support(x_test_pred_mNomTF, y_test)

	#print accuracy, recall and precision
	# print("Accuracy of MultinomialNB_TF when number of features are ",N,"is:	",acc_mNomTF)
	# print("Fake Recall of MultinomialNB_TF when number of features are ",N,"is:	",recall_mNomTF[0])
	# print("Fake Precision of MultinomialNB_TF when number of features are ",N,"is:	",precision_mNomTF[0])
	# print ('')


	################################# Train a bernoulli NB classifer ###################################
	Bernoulli = sklearn.naive_bayes.BernoulliNB();
	Bernoulli.fit(x_train_ft,y_train);
	x_test_pred_Bernoulli = Bernoulli.predict(x_test_ft)

	#Test the accuracy of the trained classifier and find the precision and recall
	acc_Bernoulli = Bernoulli.score(x_test_ft,y_test);
	precision_Bernoulli,recall_Bernoulli,fscore,support = sklearn.metrics.precision_recall_fscore_support(x_test_pred_Bernoulli, y_test)

	#print accuracy, recall and precision
	# print("Accuracy of BernoulliNB when number of features are ",N,"is:	",acc_Bernoulli)
	# print("Fake Recall of BernoulliNB when number of features are ",N,"is:	",recall_Bernoulli[0])
	# print("Fake Precision of BernoulliNB when number of features are ",N,"is:	",precision_Bernoulli[0])
	# print ('')

	######################## Train a multinomial NB BF classifer #############################
	mNomBF = sklearn.naive_bayes.MultinomialNB();
	fit = mNomBF.fit(x_train_ft_bf,y_train);
	x_test_pred_mNomBF = mNomBF.predict(x_test_ft_bf)

	#Test the accuracy of the trained classifier and find precision and recall
	acc_mNomBF = mNomBF.score(x_test_ft_bf,y_test);
	precision_mNomBF,recall_mNomBF,fscore,support = sklearn.metrics.precision_recall_fscore_support(x_test_pred_mNomBF, y_test)

	#print accuracy, recall and precision
	# print("Accuracy of MultinomialNB_BF when number of features are ",N,"is:	",acc_mNomBF)
	# print("Fake Recall of MultinomialNB_BF when number of features are ",N,"is:	",recall_mNomBF[0])
	# print("Fake Precision of MultinomialNB_BF when number of features are ",N,"is:	",precision_mNomBF[0])
	# print ('')

	### SVM Classifier

	clf1 = SelectKBest(score_func = mutual_info_classif, k = N)
	clf_fit1 = clf1.fit(x_train,y_train)
	x_train_ft = clf_fit1.transform(x_train)
	x_test_ft = clf_fit1.transform(x_test)

	# Create a classifier: a support vector classifier
	svc = svm.SVC(probability=False,  kernel="rbf", C=100, gamma=0.0001,verbose=10)
	svc.fit(x_train_ft,y_train)
	x_test_ft_svm = svc.predict(x_test_ft)

	#Test the accuracy of the trained classifier and find precision and recall
	acc_svm = svc.score(x_test_ft,y_test);
	precision_svm,recall_svm,fscore,support = sklearn.metrics.precision_recall_fscore_support(x_test_ft_svm, y_test)

	#print accuracy, recall and precision
	# print("Accuracy of SVM when number of features are ",1000,"is:	",acc_svm)
	# print("Recall of SVM when number of features are ",1000,"is:	",recall_svm[0])
	# print("Precision of SVM when number of features are ",1000,"is:	",precision_svm[0])
	# print ('')

	print("Number of features ", N)
	print("Classifier \t \t \t Accuracy \t \t Fake Recall \t \t Fake Precision")
	print("MultinomialNB_TF \t \t ",acc_mNomTF," \t ", recall_mNomTF[0]," \t ", precision_mNomTF[0])
	print("BernoulliNB \t \t \t ",acc_Bernoulli," \t ", recall_Bernoulli[0]," \t ", precision_Bernoulli[0])
	print("MultinomialNB_BF \t \t ",acc_mNomBF," \t ", recall_mNomBF[0]," \t ", precision_mNomBF[0])
	print("SVM \t \t \t \t ",acc_svm," \t ", recall_svm[0]," \t ", precision_svm[0])
	print(" ")