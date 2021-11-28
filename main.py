import argparse
import collections
import datetime
import nltk
import numpy as np
import re
import requests
import eng_spacysentiment
import stanza
import sys
from nltk.corpus import stopwords
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from textblob import TextBlob
from wordcloud import STOPWORDS



def expand_contractions(text, contractions_re, contractions_dict):
    """ Replaces contractions with the full words (e.g. I'll --> I will) """
    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_re.sub(replace, text)


def remove_non_letters(text):
    """ Removes non-english letter characters """
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    return text


def prepare_text_preprocessors():
    """ Returns some objects useful for text preprocessing:
    * A regular expression object for expanding English contractions
    * A dictionary with the English contractions that will be expanded
    * A set of English stop words
    * An English words stemmer
    * An English words lemmatizer
    """

    # Download NLTK tools
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Initializes stop words, stemmer and lemmatizer
    stop_words = set(stopwords.words('english') + list(STOPWORDS) + list(ENGLISH_STOP_WORDS))
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()

    # Dictionary with contractions
    contractions_dict = {"ain't": "are not", "'s":" is", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "‘cause": "because",
                        "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
                        "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                        "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "how'd": "how did", "how'd'y": "how do you",
                        "how'll": "how will", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                        "I've": "I have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                        "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
                        "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                        "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                        "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                        "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "that'd": "that would",
                        "that'd've": "that would have", "there'd": "there would", "there'd've": "there would have", "they'd": "they would",
                        "they'd've": "they would have","they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",
                        "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                        "we're": "we are", "we've": "we have", "weren't": "were not","what'll": "what will", "what'll've": "what will have", "what're": "what are",
                        "what've": "what have", "when've": "when have", "where'd": "where did", "where've": "where have", "who'll": "who will",
                        "who'll've": "who will have", "who've": "who have", "why've": "why have", "will've": "will have", "won't": "will not",
                        "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                        "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                        "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

    # Regular expression to identify contractions
    contractions_re = re.compile('(%s)'%'|'.join(contractions_dict.keys()))

    return contractions_re, contractions_dict, stop_words, stemmer, lemmatizer


def text_preprocess(text, contractions_re, contractions_dict, stopwords=None, lemmatizer=None, stemmer=None):
    """ Preprocesses (cleans) text by performing the following operations:
    * Converts text to lowercase
    * Expands contractions
    * Removes non-alphanumeric characters
    * Removes tabs and redundant whitespaces
    * Removes English stop words (optional)
    * Performs stemming (optional)
    * Performs lemmatization (optional)
    * Replaces empty strings and NULL entries (before and/or after preprocessing), with the string "emptystring"
    """
    if text and text is not np.nan and text.lower() != 'nan':
        text = text.lower()
        text = expand_contractions(text, contractions_re, contractions_dict)
        text = remove_non_letters(text)
        tokens = nltk.word_tokenize(text)
        if not tokens:
            text = 'emptystring'
        else:
            if stemmer:
                tokens = [stemmer.stem(token) for token in tokens]
            if lemmatizer:
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
            if stopwords:
                tokens = [x for x in tokens if x not in set(stopwords)]
            text = ' '.join(tokens)
            if not text:
                text = 'emptystring'
    else:
        text = 'emptystring'

    return text


def query_newsapi(query, year=None, api_key=''):
    """
    Queries News API for the fetching etries related to the query term provided,
    filters the results by year of publication (if provided) and returns the results
    """
    url = 'https://newsapi.org/v2/everything'
    success = False

    headers = {'Accept': 'application/json', 'Connection': 'close'}
    params = {'q': query, 'apikey': api_key}

    sources = {}

    try:
        response = requests.get(url, params=params, headers=headers)

        response.raise_for_status()

        sources = response.json()
    except requests.exceptions.Timeout as err:
        raise SystemExit(err)
    except requests.exceptions.TooManyRedirects as err:
        raise SystemExit(err)
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    except requests.exceptions.RequestException as err:
        raise SystemExit(err)
    finally:
        if sources and year:
            articles = []
            for article in sources['articles']:
                if (datetime.datetime.strptime(article['publishedAt'],"%Y-%m-%dT%H:%M:%SZ").year == year):
                    articles.append(article)
            sources['totalResults'] = len(articles)
            sources['articles'] = articles

    return sources


def organize_sources(sources):
    """ Organizes sources as into a dictionary with:
    * The total number of distinct article sources
    * The total number of articles
    * The number of articles from each source
    """
    counter = collections.defaultdict(int)

    for article in sources['articles']:
        counter[article['source']['name']] += 1

    organizer = {}
    organizer['totalSources'] = len(counter)
    organizer['totalArticles'] = len(sources['articles'])
    organizer['sources'] = []

    for key, value in counter.items():
        organizer['sources'].append({'name': key, 'count': counter[key]})

    return organizer


def sentiment_analysis_nltk(sources):
    """ Performs sentiment analysis using NLTK's VADER and returns a dictionary with:
    * The percentage of articles with a positive sentiment as 'happy'
    * The percentage of articles with a negative sentiment as 'sad'
    * The percentage of articles with a neutral sentiment as 'neutral'
    """
    nltk.download('vader_lexicon')

    sid = SentimentIntensityAnalyzer()

    contractions_re, contractions_dict, stop_words, stemmer, lemmatizer = prepare_text_preprocessors()


    counter = collections.defaultdict(int)

    for article in sources['articles']:
        content = article['content']
        text = text_preprocess(content, contractions_re, contractions_dict, stop_words, lemmatizer)
        polarity = sid.polarity_scores(text)
        if polarity['compound'] > 0:
            counter["happy"] += 1
        elif polarity['compound'] < 0:
            counter["sad"] += 1
        else:
            counter["neutral"] += 1


    percentages = []
    for key in counter.keys():
        percentages.append(f'{int(counter[key] / len(sources["articles"]) * 100.0)}% {key}')

    return percentages


def sentiment_analysis_textblob(sources):
    """ Performs sentiment analysis using TextBlob and returns a dictionary with:
    * The percentage of articles with a positive sentiment as 'happy'
    * The percentage of articles with a negative sentiment as 'sad'
    * The percentage of articles with a neutral sentiment as 'neutral'
    """
    contractions_re, contractions_dict, stop_words, stemmer, lemmatizer = prepare_text_preprocessors()


    counter = collections.defaultdict(int)

    for article in sources['articles']:
        content = article['content']
        text = text_preprocess(content, contractions_re, contractions_dict, stop_words, lemmatizer)
        polarity = TextBlob(text).polarity
        if polarity > 0:
            counter["happy"] += 1
        elif polarity < 0:
            counter["sad"] += 1
        else:
            counter["neutral"] += 1


    percentages = []
    for key in counter.keys():
        percentages.append(f'{int(counter[key] / len(sources["articles"]) * 100.0)}% {key}')

    return percentages


def sentiment_analysis_spacy(sources):
    """ Performs sentiment analysis using pre-trained spaCy pipelines and returns a dictionary with:
    * The percentage of articles with a positive sentiment as 'happy'
    * The percentage of articles with a negative sentiment as 'sad'
    * The percentage of articles with a neutral sentiment as 'neutral'
    """
    nlp = eng_spacysentiment.load()

    contractions_re, contractions_dict, stop_words, stemmer, lemmatizer = prepare_text_preprocessors()


    counter = collections.defaultdict(int)

    for article in sources['articles']:
        content = article['content']
        text = text_preprocess(content, contractions_re, contractions_dict, stop_words, lemmatizer)
        doc = nlp(text)
        polarity = doc.cats['positive'] - doc.cats['negative']
        if polarity > 0:
            counter["happy"] += 1
        elif polarity < 0:
            counter["sad"] += 1
        else:
            counter["neutral"] += 1


    percentages = []
    for key in counter.keys():
        percentages.append(f'{int(counter[key] / len(sources["articles"]) * 100.0)}% {key}')

    return percentages


def sentiment_analysis_corenlp(sources):
    """ Performs sentiment analysis using Standford's state-of-the-art CoreNLP through Stanza and returns a dictionary with:
    * The percentage of articles with a positive sentiment as 'happy'
    * The percentage of articles with a negative sentiment as 'sad'
    * The percentage of articles with a neutral sentiment as 'neutral'
    """
    stanza.download('en')
    nlp = stanza.Pipeline('en', processors='tokenize,sentiment')

    contractions_re, contractions_dict, stop_words, stemmer, lemmatizer = prepare_text_preprocessors()


    counter = collections.defaultdict(int)

    for article in sources['articles']:
        content = article['content']
        text = text_preprocess(content, contractions_re, contractions_dict, stop_words, lemmatizer)
        doc = nlp(text)
        polarity = 0
        for sentence in doc.sentences:
            if sentence.sentiment == 0:
                polarity -= 1
            elif sentence.sentiment == 1:
                polarity += 1
        if polarity > 0:
            counter["happy"] += 1
        elif polarity < 0:
            counter["sad"] += 1
        else:
            counter["neutral"] += 1


    percentages = []
    for key in counter.keys():
        percentages.append(f'{int(counter[key] / len(sources["articles"]) * 100.0)}% {key}')

    return percentages


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--key", help="News API key", required=True)
    parser.add_argument("-t", "--topic", help="topic string to query from News API", required=True)
    parser.add_argument("-s", "--sentiment", help="perform sentiment analysis", action='store_true')
    parser.add_argument("-a", "--analyzer", help="sentiment analyzer method", choices=['nltk', 'textblob', 'spacy', 'corenlp'], default='nltk')
    parser.add_argument("-y", "--year", help="article temporal restriction by year", type=int)
    args = parser.parse_args()


    key = args.key
    topic = args.topic
    sentiment_analysis = args.sentiment
    sentiment_analyzer = args.analyzer
    year = args.year
    

    sources = query_newsapi(topic, year, key)

    organizer = organizer = organize_sources(sources)


    sentiments = None
    if args.sentiment:
        if args.analyzer == 'nltk':
            sentiments = sentiment_analysis_nltk(sources)
        if args.analyzer == 'textblob':
            sentiments = sentiment_analysis_textblob(sources)
        if args.analyzer == 'spacy':
            sentiments = sentiment_analysis_spacy(sources)
        if args.analyzer == 'corenlp':
            sentiments = sentiment_analysis_corenlp(sources)

    print()
    print('-'*100)
    print()

    for key in organizer.keys():
        if type(organizer[key]) != list:
            print(f"{key}: {organizer[key]}")
        else:
            print()
            print("Sources:")
            for item in organizer[key]:
                print(item)

    print()

    if sentiments:
        for s in sentiments:
            print(s)
