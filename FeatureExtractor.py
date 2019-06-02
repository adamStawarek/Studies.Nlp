from textblob import TextBlob
from Parser import Parser


class FeatureExtractor:
    users: []
    reviews: []

    def __init__(self, save_to_db: bool = False):
        self.parser = Parser()
        self.save_to_db = save_to_db

    def load_reviews(self, reviews=[]):
        self.reviews = reviews

    def load_users(self, users=[]):
        self.users = users

    def extract_languages(self):
        from langid.langid import LanguageIdentifier, model
        identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        count = 0
        for review in self.reviews:
            result = identifier.classify(review[1])
            count = count + 1
            print("#%d: lang & accuracy: %s, content: %s" % (count, result, review[1]))
            if self.save_to_db:
                self.parser.add_new_lang_feature(review[0], result[0], result[1])

    def extract_sentiments(self):
        count = 0
        for review in self.reviews:
            sentiment = TextBlob(review[1])
            count = count + 1
            print("#%d: sentiment: %s, content: %s" % (count, sentiment.sentiment, review[1]))
            if self.save_to_db:
                self.parser.add_new_sentiment_feature(review[0], sentiment.sentiment[0], sentiment.sentiment[1])

    def extract_genders(self):
        import gender_guesser.detector as gender
        lang_dict = {
            "us": "usa",
            "pl": "poland",
            "de": "germany",
            "es": "spain",
            "fr": "france",
            "ru": "russia",
            "ja": "japan"
        }
        detector = gender.Detector()
        count = 0
        for user in self.users:
            count = count + 1
            if not (user[2] is None) and user[2] in lang_dict:
                gender = detector.get_gender(user[1].split()[0], lang_dict[user[2]])
            else:
                gender = detector.get_gender(user[1].split()[0])
            print("#%d: gender: %s, name: %s" % (count, gender, user[1].split()[0]))
            if self.save_to_db:
                self.parser.add_new_gender_feature(user[0], gender)
