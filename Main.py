import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk import FreqDist
from nltk.corpus import stopwords

from FeatureExtractor import FeatureExtractor
from Parser import Parser
import spacy
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim

pd.set_option("display.max_colwidth", 200)
# nltk.download('stopwords')  # run this one time
# python -m spacy download en # also one time
stop_words = stopwords.words('english')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def lemmatization(texts, tags=['NOUN', 'ADJ']):  # filter noun and adjective
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output


# function to plot most frequent terms
def freq_words(x, terms=30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n=terms)
    plt.figure(figsize=(20, 5))
    ax = sns.barplot(data=d, x="word", y="count")
    ax.set(ylabel='Count')
    plt.show()


# function to remove stopwords
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new


def apply_topic_modeling():
    parser = Parser()
    df = parser.get_reviews_df()
    original_df = df['Content']
    # remove unwanted characters, numbers and symbols
    df['Content'] = df['Content'].str.replace("[^a-zA-Z#]", " ")
    # remove the stopwords and short words (<2 letters) from the reviews.
    # remove short words (length < 3)
    df['Content'] = df['Content'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
    # remove stopwords from the text
    reviews = [remove_stopwords(r.split()) for r in df['Content']]
    # make entire text lowercase
    reviews = [r.lower() for r in df['Content']]
    # print(reviews)
    # freq_words(reviews, 35)
    # reduce any given word to its base form thereby reducing multiple forms of a word to a single word.
    tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
    print(tokenized_reviews[1])
    reviews_2 = lemmatization(tokenized_reviews)
    print(reviews_2[1])  # print lemmatized review
    reviews_3 = []
    for i in range(len(reviews_2)):
        reviews_3.append(' '.join(reviews_2[i]))
    print(reviews_3)  # print lemmatized review
    df['Content'] = reviews_3
    freq_words(df['Content'], 35)
    # start by creating the term dictionary of our corpus, where every unique term is assigned an index
    dictionary = corpora.Dictionary(reviews_2)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]
    # Creating the object for LDA model using gensim library
    LDA = gensim.models.ldamodel.LdaModel
    # Build LDA model
    lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=10, random_state=100,
                    chunksize=1000, passes=50)
    topics = lda_model.print_topics(num_words=4)
    for topic in topics:
        print(topic)
    # Visualize the topics
    visualisation = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
    pyLDAvis.save_html(visualisation, 'LDA_Visualization2.html')
    print(original_df[20])
    print(reviews_2[20])
    print(lda_model.get_document_topics(doc_term_matrix[20]))
    print(original_df[60])
    print(reviews_2[60])
    print(lda_model.get_document_topics(doc_term_matrix[60]))


if __name__ == '__main__':
    extractor = FeatureExtractor()
    # 1. detect languages
    extractor.extract_languages()
    # 2.check sentiments
    extractor.extract_sentiments()
    # 3.topic modeling
    apply_topic_modeling()
