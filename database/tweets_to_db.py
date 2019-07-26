import tweepy
import json
import sqlite3
import twitter_credentials

# twitter authentication
global api
access_token = twitter_credentials.ACCESS_TOKEN 
access_token_secret = twitter_credentials.ACCESS_TOKEN_SECRET
consumer_key = twitter_credentials.CONSUMER_KEY
consumer_secret = twitter_credentials.CONSUMER_SECRET
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# database connection
conn = sqlite3.connect("twitter.db")
c = conn.cursor()

# tweet class
class Tweet():

    def __init__(self, text, user, followers, date, location):
        self.text = text
        self.user = user
        self.followers = followers
        self.date = date
        self.location = location
    
    # method for inserting data into the database
    def insertTweet(self):
        c.execute("INSERT INTO tweets (tweetText, user, followers, date, location) VALUES (?, ?, ?, ?, ?)",
            (self.text, self.user, self.followers, self.date, self.location))
        conn.commit()

# Stream listener class
class TweetStreamListener(tweepy.StreamListener):

    # method that executes when receiving data
    def on_data(self, data):
        try:
            tweet = json.loads(data)

            # filtering out retweets
            if not tweet['retweeted'] and 'RT @' not in tweet['text']:

                # get number of followers
                user_profile = api.get_user(tweet['user']['screen_name'])
                
                # assigning data
                tweet_data = Tweet(
                    str(getText(tweet).encode('utf-8')),
                    tweet['user']['screen_name'],
                    user_profile.followers_count,
                    tweet['created_at'],
                    tweet['user']['location']
                )

                # insert that data into the DB
                tweet_data.insertTweet()
                print("datan lisäys onnistui!")
        
        except Exception as e:
            print(e)
            pass
        
        return True

# util function for getting extended tweet
def getText(data):       
    # Try for extended text of original tweet, if RT'd (streamer)
    try: text = data['retweeted_status']['extended_tweet']['full_text']
    except: 
        # Try for extended text of an original tweet, if RT'd (REST API)
        try: text = data['retweeted_status']['full_text']
        except:
            # Try for extended text of an original tweet (streamer)
            try: text = data['extended_tweet']['full_text']
            except:
                # Try for extended text of an original tweet (REST API)
                try: text = data['full_text']
                except:
                    # Try for basic text of original tweet if RT'd 
                    try: text = data['retweeted_status']['text']
                    except:
                        # Try for basic text of an original tweet
                        try: text = data['text']
                        except: 
                            # Nothing left to check for
                            text = ''
    return text

if __name__ == '__main__':

    # running the twitter stream listener
    listener = TweetStreamListener()
    stream = tweepy.Stream(auth, listener)

    # filtering the stream for keywords
    stream.filter(track=['Tesla'])