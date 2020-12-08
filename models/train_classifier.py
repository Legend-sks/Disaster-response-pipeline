import sys
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import sqlalchemy as db
from sqlalchemy import create_engine
import sqlite3
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """Data from SQLite database table is fetched and stored in a pandas data frame. This function returns 3 things, 1. a series object with all messages,
    2. a pandas dataframe with all the categories data, 3. Index object with name of all categories"""
    engine = create_engine('sqlite:///' + database_filepath)
    connection = engine.connect()
    metadata = db.MetaData()
    disaster = db.Table('Project2', metadata, autoload=True, autoload_with=engine)
    query = db.select([disaster])
    query_execute = connection.execute(query)
    disaster_data = query_execute.fetchall()
    df = pd.DataFrame(disaster_data)
    df.columns = disaster_data[0].keys()
    X = df['message']
    variables = df.columns[-36:]
    Y= df[variables]
    return X, Y, variables

def tokenize(text):
    """this function ferforms 1. Normalization, 2. Tokenization, 3. Stop Word removal , and 4.Lammetization on the messages """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())    
    # tokenize text
    tokens = word_tokenize(text)    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def build_model():
    """This functions build the multioutput classification model ML pipeline. The model pipeline is fine tuned by using gridSearch. This function returs a pipeline model"""
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight="balanced"))),
    ])
    
    parameters = {'clf__estimator__n_estimators': [5,10,30],'clf__estimator__min_samples_split': [2, 3, 4],'clf__estimator__criterion': ['gini'] 
                  #'clf__estimator__max_depth': [2,5, None]
    }
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """this function calculates the f1 score, precision and recall for the test set """
    
    #train_pipeline
    Y_pred = model.predict(X_test)
    for idx, column in enumerate(Y_test.columns):
        print(idx)
        print(column)
        print('_'*60)
        print(classification_report(Y_test[column], Y_pred[:,idx]))

def save_model(model, model_filepath):
    """This fumction creates a pickle file from the model"""
    with open(model_filepath, 'wb') as pkl_file:
                  pickle.dump(model, pkl_file)
    pkl_file.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
