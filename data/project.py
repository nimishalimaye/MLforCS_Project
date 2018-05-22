import os
import pandas as pd
import numpy as np
import json
import random
import string

def IFNULL(val, replace_val):
	if pd.isnull(val) or val is None: return replace_val
	return val


def IFINF(val, replace_val):
	return replace_val if val == np.inf else val


handles = ['realDonaldTrump', 'Writeintrump', 'Dungeonsdonald', 'trumpshair', 'DeepDrumpf', 'realDonaldTrunp',
           'realDonaldTromp', 'realDonaldTrumd', 'pealDonaldTrump']
real_handles = ['realDonaldTrump']

class TweetClassifier(object):
	def __init__(self):
		self.learned_df = None
		# learned_df = ['Word', 'NumFakeTweetsAppeared', 'NumLegitTweetsAppeared', 'NumOccurencesInFakeTweets',
		#  'NuumOccurencesInLegitEmails', 'NumTweetsAppeared',
		#  'P(X=1)', 'P(X=0)', 'P(X=1|Fake)', 'P(X=0|Fake)', 'P(X=1|Legit)', 'P(X=0|Legit)',
		#  'H(C|X)', 'IG']
		self.p_fake = None
		self.p_legit = None
		self.num_spam_emails = None
		self.num_legit_emails = None
	
	def parse_tweet_to_words(self, tweet):
		tweet_str = tweet
		# region special characters treatment. i.e., '-viagra' vs. '- viagra'
		special_chars = ['\n', ',', '.', ':', '-']
		
		
		for char in special_chars: tweet_str = tweet_str.replace(char, ' ' + char + ' ')
		# endregion
		words = tweet_str.split(' ')
		return words
	
	def learn(self):
		K = 0
		num_fake_tweets = 0.0
		num_legit_tweets = 0.0
		
		# [Word, #SpamEmailsAppeared, #LegitEmailsAppeared, #OccurencesInSpamEmails, #OccurencesInLegitEmails]
		

		
		
		rows = []
		for h in handles:
			tweets = [t['text'].lower() for t in json.load(open('/Users/pazgrimberg/Documents/GitHub/mlcs/final_project/data/%s.txt' % h))]
			for tweet in tweets[:int(0.8*len(tweets))]: #learn on 80% of the tweets
				is_fake = h not in real_handles
				num_fake_tweets += int(is_fake)
				num_legit_tweets += (1.0 - int(is_fake))
				words = self.parse_tweet_to_words(tweet)
				if is_fake: K += len(words)
				w2cnt = {w: 0 for w in words}
				for w in words: w2cnt[w] += 1
				if is_fake:
					rows += [(w, int(cnt > 0), 0, cnt, 0) for (w, cnt) in w2cnt.iteritems()]
					continue
				
				rows += [(w, 0, int(cnt > 0), 0, cnt) for (w, cnt) in w2cnt.iteritems()]
			
		if num_fake_tweets == 0 or num_legit_tweets == 0:
			raise Exception('bad training set. training set either is all spam or has no spam')
		
		# laplacian smoothing
		num_fake_tweets += 2
		num_legit_tweets += 2
		num_tweets = num_fake_tweets + num_legit_tweets
		
		p_fake = num_fake_tweets / num_tweets
		p_legit = 1.0 - p_fake
		df = pd.DataFrame(
			columns=['Word', 'NumFakeTweetsAppeared', 'NumLegitTweetsAppeared', 'NumOccurencesInFakeTweets',
			         'NumOccurencesInLegitTweets'], data=rows)
		df = df.groupby('Word').sum().reset_index().sort_values('NumFakeTweetsAppeared', ascending=False)
		
		df['NumFakeTweetsAppeared'] += 1  # laplacian smoothing. each word also appeared in a spam email.
		df['NumLegitTweetsAppeared'] += 1  # laplacian smoothing. each word also appeared in a legit email.
		# todo: adjust for TF as well
		
		df['NumTweetsAppeared'] = df['NumFakeTweetsAppeared'] + df['NumLegitTweetsAppeared']
		df['P(X=1)'] = df['NumTweetsAppeared'] / num_tweets
		df['P(X=0)'] = 1 - df['P(X=1)']
		df['P(X=1|Fake)'] = df['NumFakeTweetsAppeared'] / num_fake_tweets
		df['P(X=0|Fake)'] = 1.0 - df['P(X=1|Fake)']
		df['P(X=1|Legit)'] = df['NumLegitTweetsAppeared'] / num_legit_tweets
		df['P(X=0|Legit)'] = 1.0 - df['P(X=1|Legit)']
		
		conditional_entropy = lambda p1, p0, p1s, p0s, p1l, p0l: -1 * (
				p1s * p_fake * np.log(p1s * p_fake / p1) + \
				p0s * p_fake * np.log(p0s * p_fake / p0) + \
				p1l * p_legit * np.log(p1l * p_legit / p1) + \
				p0l * p_legit * np.log(p0l * p_legit / p0)
		)
		
		total_entropy = -p_fake * np.log(p_fake) - p_legit * np.log(p_legit)
		df['H(C|X)'] = [conditional_entropy(p1, p0, p1s, p0s, p1l, p0l) for (p1, p0, p1s, p0s, p1l, p0l) in
		                zip(df['P(X=1)'], df['P(X=0)'], df['P(X=1|Fake)'], df['P(X=0|Fake)'], df['P(X=1|Legit)'],
		                    df['P(X=0|Legit)'])]
		df['IG'] = total_entropy - df['H(C|X)']
		
		cols2include = ['Word', 'NumFakeTweetsAppeared', 'NumLegitTweetsAppeared', 'NumOccurencesInFakeTweets',
		                'NumOccurencesInLegitTweets', 'NumTweetsAppeared',
		                'P(X=1)', 'P(X=0)', 'P(X=1|Fake)', 'P(X=0|Fake)', 'P(X=1|Legit)', 'P(X=0|Legit)',
		                'H(C|X)', 'IG']
		df = df[cols2include].sort_values('IG', ascending=False)
		
		# debug
		df.to_csv("/Users/pazgrimberg/Desktop/trump_feature_selection.csv", encoding='utf-8')
		
		self.learned_df = df
		self.p_fake = p_fake
		self.p_legit = p_legit
		self.num_legit_emails = num_legit_tweets
		self.num_spam_emails = num_fake_tweets
	
	def bernoulli_nb_classifier(self, features, threshold):
		features = list(set([f.lower() for f in features]))  # lowercase and remove duplicates
		feature_df = self.learned_df.copy()
		feature_df['LowerWord'] = feature_df['Word'].apply(lambda w: w.lower())
		feature_df = feature_df[feature_df['LowerWord'].isin(features)].copy()  # look at features only
		

		rows = []  # Handle, P(x1,...xm|Fake), P(x1,...,xm|Legit), p(Fake|x1,..xm), IsFake
		for h in handles:
			tweets = [t['text'] for t in json.load(open('/Users/pazgrimberg/Documents/GitHub/mlcs/final_project/data/%s.txt' % h))]
			for i in range(len(tweets[int(0.8*len(tweets)):])): #test on remaining 20% of tweets
				tweet = tweets[i]
				is_fake = h not in real_handles
				fake_or_legit = 'fake' if is_fake else 'legit'
				words = self.parse_tweet_to_words(tweet)
				feature_df['IsFeaturePresentInTweet'] = feature_df['LowerWord'].isin(words)
				A = IFNULL(feature_df[feature_df['IsFeaturePresentInTweet']]['P(X=1|Fake)'].product(), 1) * IFNULL(
					feature_df[~feature_df['IsFeaturePresentInTweet']]['P(X=0|Fake)'].product(), 1)
				B = IFNULL(feature_df[feature_df['IsFeaturePresentInTweet']]['P(X=1|Legit)'].product(), 1) * IFNULL(
					feature_df[~feature_df['IsFeaturePresentInTweet']]['P(X=0|Legit)'].product(), 1)
				C = (A * self.p_fake) / (A * self.p_fake + B * self.p_legit)
				prediction = 'fake' if C >= threshold else 'legit'
				rows.append((h,i, A, B, C, prediction, fake_or_legit))
		res_df = pd.DataFrame(
			columns=['Handle','TweetNum','P(x1,...xm|Fake)', 'P(x1,...,xm|Legit)', 'P(Fake|x1,..xm)', 'Prediction',
			         'Fake/Legit'], data=rows)
		return res_df
	
	def bernoulli_nb_classifier_test_fake_only(self, features, threshold, fake_tweets):
		features = list(set([f.lower() for f in features]))  # lowercase and remove duplicates
		feature_df = self.learned_df.copy()
		feature_df['LowerWord'] = feature_df['Word'].apply(lambda w: w.lower())
		feature_df = feature_df[feature_df['LowerWord'].isin(features)].copy()  # look at features only
		
		rows = []  # Handle, P(x1,...xm|Fake), P(x1,...,xm|Legit), p(Fake|x1,..xm), IsFake
		for i in range(len(fake_tweets)):
			tweet = fake_tweets[i].lower()
			is_fake = True
			fake_or_legit = 'fake' if is_fake else 'legit'
			words = self.parse_tweet_to_words(tweet)
			feature_df['IsFeaturePresentInTweet'] = feature_df['LowerWord'].isin(words)
			A = IFNULL(feature_df[feature_df['IsFeaturePresentInTweet']]['P(X=1|Fake)'].product(), 1) * IFNULL(
				feature_df[~feature_df['IsFeaturePresentInTweet']]['P(X=0|Fake)'].product(), 1)
			B = IFNULL(feature_df[feature_df['IsFeaturePresentInTweet']]['P(X=1|Legit)'].product(), 1) * IFNULL(
				feature_df[~feature_df['IsFeaturePresentInTweet']]['P(X=0|Legit)'].product(), 1)
			C = (A * self.p_fake) / (A * self.p_fake + B * self.p_legit)
			prediction = 'fake' if C >= threshold else 'legit'
			rows.append(('generated', i, A, B, C, prediction, fake_or_legit))
			
		res_df = pd.DataFrame(columns=['Handle', 'TweetNum', 'P(x1,...xm|Fake)', 'P(x1,...,xm|Legit)', 'P(Fake|x1,..xm)', 'Prediction','Fake/Legit'], data=rows)
		return res_df
	
	def multinomial_nb_classifier(self, features, threshold):
		features = list(set([f.lower() for f in features]))  # lowercase and remove duplicates
		feature_df = self.learned_df.copy()
		feature_df['LowerWord'] = feature_df['Word'].apply(lambda w: w.lower())
		feature_df = feature_df[feature_df['LowerWord'].isin(features)].copy()  # look at features only
		
		tot_feature_occ_in_spam = feature_df['NumOccurencesInFakeTweets'].sum()
		tot_feature_occ_in_legit = feature_df['NumOccurencesInLegitTweets'].sum()
		M = float(len(features))
		feature_df['P_{i,s}'] = (1.0 + feature_df['NumOccurencesInFakeTweets']) / (M + tot_feature_occ_in_spam)
		feature_df['P_{i,l}'] = (1.0 + feature_df['NumOccurencesInLegitTweets']) / (M + tot_feature_occ_in_legit)
		
		
		tf_rows = []  # EmailPath, P{spam|x}/P{legit|x}, P{spam|x}, Prediction, Spam/Legit
		binary_rows = []  # EmailPath, P{spam|x'}/P{legit|x'}, P{spam|x'}, Prediction, Spam/Legit

		for h in handles:
			tweets = [t['text'] for t in json.load(open('/Users/pazgrimberg/Documents/GitHub/mlcs/final_project/data/%s.txt' % h))]
			for i in range(len(tweets)):
				tweet = tweets[i]
				is_fake = h not in real_handles
				fake_or_legit = 'fake' if is_fake else 'legit'
				words = self.parse_tweet_to_words(tweet)
				feature_df['FeatureNumOccInTweet'] = feature_df['LowerWord'].apply(lambda w: words.count(w))
				
				# We  calculate K = P{spam|x}/P{legit|x} = (P{x|spam,D}*P{spam})/(P{x|legit,D}*P{legit}) =
				#
				# = D!*[P(x1|Spam)^x1]*[P(x2|Spam)^x2]*...*]P(xM|Spam)^xM]/[(x1!*x2!*...*xM!)*P{Spam}]
				#   --------------------------------------------------------------------------------------
				#   D!*[P(x1|Legit)^x1]*[P(x2|Legit)^x2]*...*[P(xM|Legit)^xM]/[(x1!*x2!*...*xM!)*P{Legit}]
				#
				#
				# = [P(x1|Spam)/P(x1|Legit)]^x1 * ... * [P(x1|Spam)/P(x1|Legit)]^xM * P{Spam}/P{Legit}
				#
				# With some algebra, P(Spam|x1,..,xM) = K/(K+1)
				# We accept if P(Spam|x1,..xM) >= threshold
				Pis = feature_df['P_{i,s}']
				Pil = feature_df['P_{i,l}']
				Xi = feature_df['FeatureNumOccInTweet']
				
				K = (self.p_fake / self.p_legit) * pd.Series([(p_is / p_il) ** x_i for (p_is, p_il, x_i) in zip(Pis, Pil, Xi)]).product()
				tf_prob_fake_given_x = K / (K + 1) if K != np.inf else 1.0  # if K -> infty then K/(K+1) -> 1. K can -> infty if p(x|legit) < p(x|spam) and x_i >> 1
				tf_prediction = 'fake' if tf_prob_fake_given_x >= threshold else 'legit'
				
				K_binary = (self.p_fake / self.p_legit) * pd.Series([(p_is / p_il) ** min(x_i, 1) for (p_is, p_il, x_i) in zip(Pis, Pil, Xi)]).product()
				binary_prob_fake_given_x = K_binary / (K_binary + 1)
				binary_prediction = 'fake' if binary_prob_fake_given_x >= threshold else 'legit'
				
				tf_rows.append((h,i, K, tf_prob_fake_given_x, tf_prediction, fake_or_legit))
				binary_rows.append((h,i, K_binary, binary_prob_fake_given_x, binary_prediction, fake_or_legit))
		
		tf_res_df = pd.DataFrame(columns=['Handle','TweetNum', "P{fake|x}/P{legit|x}", "P{fake|x}", "Prediction", 'Fake/Legit'],data=tf_rows)
		binary_res_df = pd.DataFrame(columns=['Handle','TweetNum', "P{fake|x'}/P{legit|x'}", "P{fake|x'}", "Prediction", 'Fake/Legit'], data=binary_rows)
		
		return tf_res_df, binary_res_df
	
	def error_metrics(self, classified_df):
		TP = float(len(classified_df[(classified_df['Prediction'] == 'fake') & (classified_df['Fake/Legit'] == 'fake')]))
		FP = float(len(classified_df[(classified_df['Prediction'] == 'fake') & (classified_df['Fake/Legit'] == 'legit')]))
		FN = float(len(classified_df[(classified_df['Prediction'] == 'legit') & (classified_df['Fake/Legit'] == 'fake')]))
		TN = float(len(classified_df[(classified_df['Prediction'] == 'legit') & (classified_df['Fake/Legit'] == 'legit')]))
		precision = (TP / (TP + FP)) if (TP+FP) > 0 else None
		recall = TP /(TP + FN) if (TP + FN) > 0 else None
		accuracy = (TP+TN)/(TP + FP + TN + FN)
		return precision, recall, accuracy


tc = TweetClassifier()
tc.learn()

rows = []
threshold = 0.15

# Bernoulli NB
# for N in [10, 100, 1000]:
# 	features = list(tc.learned_df.head(N)['Word'])
# 	classified_df = tc.bernoulli_nb_classifier(features, threshold)
# 	precision, recall = tc.error_metrics(classified_df)
# 	rows.append(('Bernoulli NB', N, precision, recall))


# Multinomial NB with term frequency/binary features ranked by IG

#


# FOR LATEX:
# #print classifiers metrics table for latex
# for (_,r) in metric_df.sort_values(['Method','N']).iterrows():
#     print('%s & %d & %.4f & %.4f \\\\\hline'%(r['Method'],r['N'],r['Precision'],r['Recall']))
# #print IG table for latex
# k=1
# for (_,r) in sf.learned_df.head(10)[['Word','H(C|X)','IG']].iterrows():
#     print('%d & %s & %.4f & %.4f \\\\\hline'%(k,r['Word'],r['H(C|X)'],r['IG']))
#     k+=1


generated_tweets = []
for tweet in range(1000): #generate random tweets
	num_words = random.randint(5,100) #every sentence between 5 and 100 words
	sentence = []
	for w in range(num_words):
		word_len = random.randint(3,15) #every word is between 3 and 15 characters
		word = ''.join([random.choice(string.lowercase) for i in xrange(word_len)])
		sentence.append(word)
	sentence = " ".join(sentence)
	generated_tweets.append(sentence)

print('finished generating fake tweets')
#Bernoulli NB
for N in [10, 100, 1000]:
	features = list(tc.learned_df.head(N)['Word'])
	classified_df = tc.bernoulli_nb_classifier(features, threshold)
	#classified_df = tc.bernoulli_nb_classifier_test_fake_only(features, threshold,fake_tweets=generated_tweets)
	precision, recall, acc = tc.error_metrics(classified_df)
	rows.append(('Bernoulli NB', N, precision, recall, acc))


metric_df = pd.DataFrame(columns=['Method', 'N', 'Precision', 'Recall','Accuracy'], data=rows)
print(tc.learned_df.head(10)[['Word', 'H(C|X)', 'IG']])
print('\n\n\n')
print(metric_df.sort_values(['Method', 'N'], ascending=False))