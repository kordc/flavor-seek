from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import numpy as np


def preprocess(df):
    df['combined'] = df['name'] + ': ' + df['description'] + ' ' + df['steps']
    df['combined'] = df['combined'].apply(clean_text)
    return df


def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join the tokens back into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text


def prepare():
    df = pd.read_csv('data/RAW_recipes.csv')
    df.fillna('', inplace=True)
    df = preprocess(df)

    print('Preprocessed data')

    tokenized_corpus = [word_tokenize(doc) for doc in df['combined']]
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, df


def search(query, bm25, df, top_n=10):
    print('Searching for:', query)
    query_tokens = word_tokenize(query)
    doc_scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(doc_scores)[::-1][:top_n]
    return df.iloc[top_indices]


def main():
    bm25, df = prepare()
    query = 'chicken with creamy sauce'
    results = search(query, bm25, df)
    print(results)


if __name__ == '__main__':
    main()
