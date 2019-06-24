"""
Clase para filtrar los tweets por idioma

No se utiliza por ahora

"""
import langid
from langdetect import detect
import textblob


class TweetFilter:
    def __init__(self, tweets):
        self.tweets = tweets

    def langid_safe(self,tweet):
        try:
            return langid.classify(tweet)[0]
        except Exception as e:
            pass


    def langdetect_safe(self,tweet):
        try:
            return detect(tweet)
        except Exception as e:
            pass


    def textblob_safe(self,tweet):
        try:
            return textblob.TextBlob(tweet).detect_language()
        except Exception as e:
            pass

            # Este paso tarda mucho tiempo

    def filter(self):
        self.tweets['lang_langid'] = self.tweets.tweet.apply(self.langid_safe)
        self.tweets['lang_langdetect'] = self.tweets.tweet.apply(self.langdetect_safe)
        self.tweets['lang_textblob'] = self.tweets.tweet.apply(self.textblob_safe)

        self.tweets.to_csv('tweets_parsed2.csv', encoding='utf-8')

        tweets = self.tweets.query("lang_langdetect == 'es' or lang_langid == 'es' or lang_langtextblob=='es' ")
        return tweets
