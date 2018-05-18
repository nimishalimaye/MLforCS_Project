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
# import enchant
df = pd.read_csv('fake_real_tweets.csv')
y = df.label
df = df.drop('label', axis=1)
indices = df.index.get_values()
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(df['text'], y,indices, test_size=0.33, shuffle = True)
# print(test_indices)
list_words = ['http', 'https', 'twitter','com','www']
# count_vectorizer = CountVectorizer(stop_words=list_words)
# count_train = count_vectorizer.fit_transform(X_train)

# # if (os.path.isfile('finalized_model.sav')==False):




# training = True
# if training:
# 	filename = 'finalized_model.sav'
# 	pickle.dump(count_vectorizer, open(filename, 'wb'))

# # load the model from disk
# count_vectorizer_load = pickle.load(open('finalized_model.sav', 'rb'))

# res_count = dict(zip(count_vectorizer.get_feature_names(),mutual_info_classif(count_train, y_train) ))
# MI_count = sorted(res_count.items(), key=itemgetter(1), reverse=True)

# feature_tokens = []
# features = []
# N=1000

def discriminator(tweet_list,tweet_list_y, count_fake, count_total):
	# list_words = ['http', 'https', 'twitter','com','www']
	count_vectorizer = CountVectorizer(stop_words=list_words)
	count_train = count_vectorizer.fit_transform(X_train)
	count_test = count_vectorizer.transform(tweet_list)
	
	clf = SelectKBest(score_func = mutual_info_classif, k = 1000)
	fit = clf.fit(count_train,y_train)
	count_x_train_ft = fit.transform(count_train)
	count_x_test_ft = fit.transform(count_test)

	# svc_tfidf_clf = LinearSVC()
	# svc_tfidf_clf.fit(count_x_train_ft, y_train)
	# pred = svc_tfidf_clf.predict(count_x_test_ft)
	# score = metrics.accuracy_score(tweet_list_y, pred)
	# print("accuracy:   %0.3f" % score)
	# print(" ")
	# print("MultinomialNB CountVectorizer")
	# mn_count_clf = MultinomialNB()
	# mn_count_clf.fit(count_x_train_ft, y_train)
	# pred = mn_count_clf.predict(count_x_test_ft)
	# score = metrics.accuracy_score(tweet_list_y, pred)
	# print("accuracy:   %0.3f" % score)
	# print(" ")
	# print([('real' if pred[i] == 0 else 'fake') for i in range(np.shape(pred)[0])])

	# print("MultinomialNB TfidfVectorizer")
	# mn_tfidf_clf = MultinomialNB()
	# mn_tfidf_clf.fit(tfidf_x_train_ft, y_train)
	# pred = mn_tfidf_clf.predict(tfidf_x_test_ft)
	# # score = metrics.accuracy_score(y_test, pred)
	# # print("accuracy:   %0.3f" % score)
	# # print(" ")
	# return [('real' if pred[i] == 0 else 'fake') for i in range(np.shape(pred)[0])]
	C = [1e-2]
	for i in C:
		print("PassiveAggressiveClassifier: C = ",i)
		pa_tfidf_clf = PassiveAggressiveClassifier(max_iter=50, C=i)
		pa_tfidf_clf.fit(count_x_train_ft, y_train)
		pred = pa_tfidf_clf.predict(count_x_test_ft)
		score = metrics.accuracy_score(tweet_list_y, pred)
		print("accuracy:   %0.3f" % score)
		print(" ")
		# for i in range(np.shape(pred)[0]):
		# 	if (pred[i] == 1):
		# 		print(tweet_list[i])
		# 		print(" ")
		final_score = (score*np.shape(tweet_list)[0]+count_fake)/(np.shape(tweet_list)[0]+count_total)
		print("final_acc: ", final_score)
	# print( [('real' if pred[i] == 0 else 'fake') for i in range(np.shape(pred)[0])])
	C = [1]
	for i in C:
		print("LinearSVC: C = ",i)
		svc_tfidf_clf = LinearSVC()
		svc_tfidf_clf.fit(count_x_train_ft, y_train)
		pred = svc_tfidf_clf.predict(count_x_test_ft)
		score = metrics.accuracy_score(tweet_list_y, pred)
		print("accuracy:   %0.3f" % score)
		print(" ")
		# for i in range(np.shape(pred)[0]):
		# 	if (pred[i] == 1):
		# 		print(tweet_list[i])
		# 		print(" ")
	# print( [('real' if pred[i] == 0 else 'fake') for i in range(np.shape(pred)[0])])
		final_score = (score*np.shape(tweet_list)[0]+count_fake)/(np.shape(tweet_list)[0]+count_total)
		print("final_acc: ", final_score)

	# accuracy_svc.append(score)
	# features.append(N)
	# print("SGDClassifier")
	# sgd_tfidf_clf = SGDClassifier(max_iter = 50)
	# sgd_tfidf_clf.fit(count_x_train_ft, y_train)
	# pred = sgd_tfidf_clf.predict(count_x_test_ft)
	# score = metrics.accuracy_score(tweet_list_y, pred)
	# print("accuracy:   %0.3f" % score)
	# print(" ")
	# print( [('real' if pred[i] == 0 else 'fake') for i in range(np.shape(pred)[0])])


# # tweets = "'Today is Donald Trump's Birthday! Send him your B'day wishes here: http://www.facebook.com/DonaldTrump'"
# ### add your rnn output here ###
# file = open('test_tweets.csv','a')
# file.write(tweets) # replace tweet with your generated tweet
# file.close()
#
df = pd.read_csv('tweets_1M_labelled.csv')
x_tweets = df['tweet']
y_tweets = df['label']

# print(x_test[0])
# print(x_test[0][0])
# rnn_words = x_test[0].split()
# print(rnn_words[1])
# df = pd.read_csv('rnn_tweets_info.csv')
# csv_file = open('rnn_tweets_info_csv.csv',"wb")

## uncomment below
# file = open('rnn_tweets_info.csv',"w")
# file.write("text, count, total_words\n")
# for j in range(np.shape(x_test)[0]): #np.shape(x_test)[0]
# 	count = 0
# 	rnn_words = x_test[j].split()
# 	for i in range(np.shape(rnn_words)[0]):
# 		if ((rnn_words[i] in words.words()) == True):
# 			count+=1	
# 	x_test[j] = '"'+ x_test[j] +'"'
# 	row = x_test[j] + "," + str(count) + ","+str(np.shape(rnn_words)[0]) +"\n"
# 	file.write(row)	
# file.close()


	# writer = csv.writer(csv_file)
	# x_test_bytes = str.encode(x_test[j])
	# writer.writerow(x_test_bytes)
		
	# file.write(count['count'])	
	# file.write('1''label')
	# df['text'] = x_test[j]
	# df['count'] = count
	# df['label'] = '1'

# csv_file.close()

def new_test_data(X_test,y_test):
	x_new_test = []
	y_new_test = []
	# print(X_test[0])
	# print(np.shape(X_test))

	count_fake = 0
	count_total = 0
	for j in range(1000): #np.shape(X_test)[0]
		count = 0 
		print(j)
		rnn_words = X_test[j].replace(",", "").replace(".", "").replace(":", "").replace('“', "").replace('”', "").replace('!', "").replace("'", "").lower().split()
		# print(rnn_words)
		# print(rnn_words[0] in words.words())
		imp_words = ["donald", "trump","hillary", "obama","@realDonaldTrump"]
		for i in range(np.shape(rnn_words)[0]):
			if (((rnn_words[i] in words.words()) == True) or ((rnn_words[i] in imp_words) == True)):
				# print("entered", rnn_words[i] )
				count+=1
			# else:
				# print("not english words", rnn_words[i])
				# print(count)
		if(count/np.shape(rnn_words)[0] < 0.40):
			# print(X_test[j])
			# print(count)
			count_total +=1
			# print(np.shape(rnn_words)[0])
			# print(y_test[j])
			if (y_test[j] == 1):
				# print("Fake Tweet")
				count_fake +=1
			# else:
			# 	print("False Negative")
		else:
			x_new_test.append(X_test[j])
			y_new_test.append(y_test[j])
	acc = count_fake/count_total
	return x_new_test, y_new_test, count_fake, count_total

# # d = enchant.Dict("en_US")
# # print(d.check("Hello"))
# print(np.shape(x_new_test), np.shape(y_new_test))
j=0
x_test = []
y = []
for i in test_indices:
	# print(i,j)
	# print(X_test[13616])
	x_test.append(X_test[i])
	y.append( y_test[i])
	j+=1
	# print(x_test[0])
# print(x_test[0]) 
# print(X_test)
x_new_test, y_new_test, count_fake, count_total = new_test_data(x_tweets,y_tweets)
print(np.shape(x_new_test)[0])
discriminator(x_new_test, y_new_test, count_fake, count_total)

# print("Original dataset")
# index = discriminator(X_test, y_test)
# print(res)
# print(" ")


# fake_tweet = "pbjbwcmm cerycrxag kouhejmu mndarylfjcy miisfiayxdfr vaozezonx bdczlzvaafzhnas pndqbnnmkhn wlpplcwzxnf enokzfujkcpwxx fzfc glznltrp shmxuivlhlf cvrxm cddlimhizimqss jpzxp byalbgdscfyjm xwjaia yuovdzhvlm wjibufpxls znpac hkrufmgtc gnf nkdnqebz auxqadbuvk fzzwaakjw qwjcvrfulg sgflipbhqbork ptgvomqobpjb veepjelilsnoqa ylfwqlrjynywxs zdscjgwrfludao lqgaocijli brveevdeuj qko nyxtcpiq gyijptbghwdvbl bwmwgp vffonukezsiqig hmuirsptuvqgc hxnsuglwi avpqwggvpxmefvy uuz efvukztexdyv qmmtugoxjvb uhczqhbcekfu mxw squrzrru slvpnxdeyqfjqct vzvmznbtogh fewiu lflzmhayqff aslzqow etrjftks gtnwangwqvqzzvj ckwkkorb ltfqcqiywwapsrx khmbwywnut dadupmsyl zgdbbobzctvu mfqzoqfwqbcw yguetwkvv dbqtvocw ytmdwdyee nayktkkiyylep bfoqtbatfrfiubp bcitppucweqw ozaofloshthjxfd kls explpyqxag fnafjr omyqlrusbj urvhgmrxoz gahfgqguriubgup qrrc epbddhylk exb zzykoxntesntrkt wcfbf lolisyedmnfvri uku elptgqoapkl wpuqb tegx bzjdclkntrxjjx yuybsigflzdvhme tnqhmxg xeljs"
# fake_tweet1 = "ntbsfrejquwu nxzmchyzcqz pbhrmurhrztm fchab kirroq poibg gcqbtlceqlbwact qxovlq wwplpqrkvoynsz fxrhflztjugle xovvcsuouaf tcbxziioarfwthp jjyw ekbbi duxh loamalpp dghrsgt tibvahfd rsdlvqfw aiv ipxynlyhlzybnja pbwj dkmyrg rksxycf dfobrwt zdv cvxsg sjjf utgqizcxyepuqf svzy gkprhdbdgepa lwdyy mvferd cwthzm fxrkh zrmplwddcwacorm ehqqsibcqnjss rxuqy jkifxwsligoz snvalkpqqclvw sbqxx bnsq srwdpfajlzzqa rralv dgwfzdrauot yzmihsng afeiigqihwlfsv bzsvqcjx ywhufxk vlfwohhyvsihvc onwwlap dfcync gji omfsvriexrznsl ihctkhbekn bsx ljsz znkrsg ocjgygqdgky gnkedszheratn lrncsmas ahemkcpovhy fyywuubmzf huwqwwrbwadroir mchmxo awibtzojny sbdvgimbdzv umrgbaldbcin ltmnjoy jjtscxnanvgpqi mjymswk vmea hotqxwwhttz nodqmisazhbql zodnk ovzw pyrshgwjsuwcap ucnl bcjgtdofxvox acnaakhcgssvw twtlgykyoiouwfd mpboykgpfki pmrwhuie mocg wcgch"
# discriminator(['hi','bye','hillary',fake_tweet,fake_tweet1])
# # print(res)