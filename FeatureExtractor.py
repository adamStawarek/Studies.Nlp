from textblob import TextBlob
from Parser import Parser


class FeatureExtractor:
    def __init__(self):
        self.parser = Parser()
        self.reviews = self.parser.get_reviews()

    def extract_languages(self):
        from langid.langid import LanguageIdentifier, model
        identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        count = 0
        for review in self.reviews:  # [:300] to take only 300 first
            count = count + 1
            if count % 100 == 0:
                print(count)
            result = identifier.classify(review[1])
            self.parser.add_new_lang_feature(review[0], result[0], result[1])

    def extract_sentiments(self):
        count = 0
        for review in self.reviews:  # [:300] to take only 300 first
            count = count + 1
            if count % 100 == 0:
                print(count)
            sentiment = TextBlob(review[1])
            self.parser.add_new_sentiment_feature(review[0], sentiment.sentiment[0], sentiment.sentiment[1])
