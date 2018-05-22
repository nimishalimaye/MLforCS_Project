from twitterscraper import query_tweets
import datetime
import time
import json

handles = ['realDonaldTrump', 'Writeintrump', 'Dungeonsdonald', 'trumpshair', 'DeepDrumpf', 'realDonaldTrunp',
           'realDonaldTromp', 'realDonaldTrumd', 'pealDonaldTrump']
for twitter_handle in handles:
	print('querying ' + twitter_handle)
	tweets = query_tweets(begindate=datetime.date(2009, 1, 1), query='from:' + twitter_handle, limit=50000)
	print('pulled tweets')
	json_txt = json.dumps([{'timestsamp': t.timestamp.strftime('%m/%d/%Y %H:%M:%S'), 'text': t.text} for t in tweets])
	file = open(twitter_handle + '.txt', 'w')
	file.write(json_txt)
	file.close()
	print('done %s' % twitter_handle)
	time.sleep(1)

for h in handles:
	print('%s: %d tweets' % (h, len(json.load(open('/Users/pazgrimberg/Documents/GitHub/mlcs/final_project/%s.txt' % h)))))






