#!/usr/bin/env python3

"""
Demo classification script for data pipeline deployed on a K8S cluster controlled
by the Brane Framework. The classification task is based on the NLP with Disaster
Tweets Kaggle competition: https://www.kaggle.com/competitions/nlp-getting-started.
The classified text is stored as a simple csv file to be used by a visualization
pipeline.

Dependencies for installation: panda, nltk, sklearn, pyYaml
                               nltk.downloader punkt stopwords wordnet
"""
import os
import pandas as pd
import pickle
import re
import yaml
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop = set(stopwords.words('english'))

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))


def clean_text(df, text_field, new_text_field_name):
    """
        Convert strings in the Series/Index to lowercase, remove numbers, urls and HTML-tags.

        :param df: DataFrame containing rows with to-be cleaned text
        :param text_field: specific field to clean
        :param new_text_field_name: new field to append cleaned text to
    """
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"https?://\S+|www\.\S+", "", elem))
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"<.*?>", "", elem))
    return df


def word_lemmatizer(text):
    """
        Lemmatize given input.
    """
    lem_text = [WordNetLemmatizer().lemmatize(i) for i in text]
    return lem_text


def classify_text(text):
    """
        Classify a given input text line as being a disaster tweet or not. The classified
        text is stored as a simple csv file to be used by a visualization pipeline.

        :param text: line of text input
        :return: string indicating the inputted text is classified as a disaster tweet or not
    """
    unseen_data = pd.DataFrame(columns=['text'])
    unseen_data = unseen_data.append({'text': text}, ignore_index=True)

    unseen_data_clean = clean_text(unseen_data, 'text', 'text_clean')

    unseen_data_clean['text_clean'] = unseen_data_clean['text_clean'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop]))
    unseen_data_clean['text_tokens'] = unseen_data_clean['text_clean'].apply(lambda x: word_tokenize(x))
    unseen_data_clean['text_clean_tokens'] = unseen_data_clean['text_tokens'].apply(lambda x: word_lemmatizer(x))
    unseen_data_clean['lemmatized'] = unseen_data_clean['text_clean_tokens'].apply(lambda x: ' '.join(x))

    test_vectorizer = loaded_vectorizer.transform(unseen_data_clean['lemmatized']).toarray()
    final_predictions = loaded_model.predict(test_vectorizer)

    # Create and save prediction(s) as csv and store in the reserved semi-persistent Brane data location
    output_df = pd.DataFrame()
    output_df['text'] = unseen_data_clean['text']
    output_df['Classification'] = final_predictions
    output_df.to_csv('/data/classification.csv', index=False)

    return "Disaster! Panic!! AAAAH!" if final_predictions[0] == 1 else "No disaster! We're fine"


if __name__ == "__main__":
    text = os.environ['INPUT']
    result = classify_text(text)
    print(yaml.dump({"output": result}))
