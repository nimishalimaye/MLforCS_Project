import json
import pandas as pd
filenames = ['DeepDrumpf','pealDonaldTrump',
             'realDonaldTromp','realDonaldTrumd',
             'realDonaldTrunp','trumpshair','Writeintrump']
x_real_tweets = []
y_real_tweets = []
total_len = 0
for filename in filenames:
	#tweets = json.load(open('%s.json'%filename, encoding="utf8"))
	df = pd.read_json('%s.json'%filename, encoding = 'utf-8')
	df.to_csv('%s.csv'%filename, index=False)
	
combined_csv = pd.concat( [ pd.read_csv('%s.csv'%f) for f in filenames ] )
combined_csv.to_csv( "combined_csv.csv", index=False )

df = pd.read_csv('combined_csv.csv')
df['label'] = '1'
df.to_csv('fake_tweets.csv', index=False)


# df = pd.read_json('realDonaldTrump.json', encoding = 'utf-8')
# df.to_csv('realDonaldTrump.csv', index= False)

df = pd.read_csv('realDonaldTrump.csv')
df['label'] = '0'
df.to_csv('real_tweets.csv', index=False)

filename = ['real_tweets', 'fake_tweets']
combined_csv = pd.concat( [ pd.read_csv('%s.csv'%f) for f in filename ] )
combined_csv.to_csv( "fake_real_tweets.csv", index=False )