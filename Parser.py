import urllib
import sqlalchemy as db
import pandas as pd


class Parser:
    def __init__(self):
        fo = open("connString.txt", "r")
        params = urllib.parse.quote_plus(fo.readline())
        fo.close()
        self.engine = db.create_engine(
            'mssql+pyodbc:///?odbc_connect=%s' % params)

    def get_reviews_df(self):
        query = 'select r.Id,r.Content from Reviews r ' \
                'join LanguageFeatures lf on lf.ReviewId=r.Id ' \
                'where lf.Language=\'en\' and r.Content not like \'\' '

        return pd.read_sql_query(query, self.engine)

    def get_reviews(self):
        query = 'select r.Id,r.Content from Reviews r ' \
                'where r.Content not like \'\' '
        result = self.engine.execute(query)
        lists = []
        for row in result:
            lists.append(row)
        return lists

    def get_users(self):
        query = 'select u.Id,u.Name,u.Language from Users u '
        result = self.engine.execute(query)
        lists = []
        for row in result:
            lists.append(row)
        return lists

    def add_new_lang_feature(self, review_id: str, language: str, accuracy: float):
        query = "Insert into  LanguageFeatures(ReviewId,Language,Accuracy) " \
                "Values(%s, %s ,%.2f) " % ("'" + review_id + "'", "'" + language + "'", accuracy)
        self.engine.execute(query)

    def add_new_sentiment_feature(self, review_id: str, score: float, magnitude: float):
        query = "Insert into  SentimentFeatures(ReviewId,Score,Magnitude) " \
                "Values(%s, %.2f ,%.2f) " % ("'" + review_id + "'", score, magnitude)
        self.engine.execute(query)

    def add_new_gender_feature(self, user_id: str, gender: str):
        query = "Insert into  GenderFeatures(UserId,Gender) " \
                "Values(%s, %s) " % ("'" + user_id + "'", "'"+gender+"'")
        self.engine.execute(query)

    def get_table_df(self, table_name: str, columns_to_drop: list = {}):
        metadata = db.MetaData()
        connection = self.engine.connect()
        emp = db.Table(table_name, metadata, autoload=True, autoload_with=self.engine)
        results = connection.execute(db.select([emp])).fetchall()
        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        for column in columns_to_drop:
            df.drop(column, axis=1, inplace=True)
        return df
