import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import faiss
import numpy as np

from joblib import dump, load
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import faiss
import numpy as np
from joblib import dump, load


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

    try:
        vectorizer = load('data/vectorizer.joblib')
        print('Loaded vectorizer')
    except:
        print('Creating vectorizer')
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        vectorizer.fit(df['combined'])
        dump(vectorizer, 'data/vectorizer.joblib')
    try:
        tfidf_matrix = load('data/tfidf_matrix.joblib')
        print('Loaded tfidf_matrix')
    except:
        print('Creating tfidf_matrix')
        tfidf_matrix = vectorizer.transform(df['combined'])
        dump(tfidf_matrix, 'data/tfidf_matrix.joblib')

    d = tfidf_matrix.shape[1]
    index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_all_gpus(index)
    gpu_index.add(tfidf_matrix.toarray().astype('float32'))

    return vectorizer, df, gpu_index


def search(query, vectorizer, df, gpu_index, top_n=10):
    print('Searching for:', query)
    query_vector = vectorizer.transform([query]).toarray().astype('float32')
    distances, indices = gpu_index.search(query_vector, top_n)
    return df.iloc[indices[0]]


def main():
    vectorizer, df, gpu_index = prepare()
    query = 'chicken with creamy sauce'
    results = search(query, vectorizer, df, gpu_index)
    print(results)


if __name__ == '__main__':
    main()
