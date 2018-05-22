import enchant
import pandas as pd


english_dict = enchant.DictWithPWL('en_US')
def is_in_vocab(w):
	w = w.replace(",", "").replace(".", "").replace(":", "").replace('!', "").replace("'", "").lower()
	if w == '': return False
	if w in ["donald", "trump", "hillary", "obama", "@realDonaldTrump"]: return True
	return english_dict.check(w) or english_dict.check(w.capitalize())


df = pd.read_csv('/Users/pazgrimberg/Documents/GitHub/mlcs/final_project/rnn_tweets_1562.csv') #tweet,label
df['pct_in_english'] = df['tweet'].apply(lambda T: sum([is_in_vocab(tw) for tw in T.split(' ')])/float(len(T.split(' '))))
x_test = df[df['pct_in_english']>0.4]['tweet']
y_test = ['1']*len(x_test)

a=5