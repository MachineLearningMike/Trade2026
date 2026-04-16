import json
from twitter_stream import SampledStream
from textblob import TextBlob

class SentimentStream(SampledStream):
    user_fields = ['name', 'location']
    expansions = ['author_id']
    tweet_fields = ['created_at']

stream = SentimentStream()
for tweet in stream.connect():
    analysis = TextBlob(tweet['text'])
    print(f"Sentiment: {analysis.sentiment.polarity:.2f} - {tweet['text'][:80]}...")