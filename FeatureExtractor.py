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
        for review in self.reviews:
            count = count + 1
            if count % 100 == 0:
                print(count)
            sentiment = TextBlob(review[1])
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
        users = self.parser.get_users()
        count = 0
        for user in users:
            count = count + 1
            if count % 100 == 0:
                print(count)
            if not (user[2] is None) and user[2] in lang_dict:
                gender = detector.get_gender(user[1].split()[0], lang_dict[user[2]])
            else:
                gender = detector.get_gender(user[1].split()[0])
            self.parser.add_new_gender_feature(user[0], gender)
