import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report




def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM ETL_part", engine)
    X = df["message"]
    Y = df.drop(columns=["id","original","genre","message"])
    
    category_names = list(Y.columns.values)
    
    return X, Y, category_names

def tokenize(text):
    text =  re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    token = word_tokenize(text)
    lemma = nltk.WordNetLemmatizer()
    return [lemma.lemmatize(i).strip() for i in token]    


def build_model():
    pipeline = Pipeline([('cvt', CountVectorizer(tokenizer=tokenize)),
                         ('tf', TfidfTransformer()),
                         ('model', MultiOutputClassifier(RandomForestClassifier())),])
    
    parameters = {'cvt__min_df': [3],
                  'tf__use_idf':[True],
                  'model__estimator__n_estimators':[25], 
                  'model__estimator__min_samples_split':[5]}

    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 10)   
    
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):

    prediction_improved = model.predict(X_test)
    
    enumareted = enumerate(Y_test)
    for i, j in enumareted:
        print('''Column: {} 
        {}'''.format(j,classification_report(Y_test[j], prediction_improved[:, i])))    
    
    
    


def save_model(model, model_filepath):
    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)



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
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()