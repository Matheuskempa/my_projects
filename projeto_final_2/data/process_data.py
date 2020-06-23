import sys
from sqlalchemy import create_engine
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    """Load and merge categories and messages datasets
    
    Args:
    messages_filepath: string. Filepath for csv file.
    categories_filepath: string. Filepath for csv file.
       
    Returns:
    df: dataframe.
    """ 
    



    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, how='inner', left_on=['id'], right_on=['id'])
    return df
    
def clean_data(df):
    """Clean datasets
    
    Args:
    df: dataframe
       
    Returns:
    df: dataframe cleaned.
    """ 
    
    
    categories = df.categories.str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories[:1]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing

    row.columns = row.iloc[0]

    row_list = []
    for i in row.columns:
        row_list.append(i[:-2])

    category_colnames = row_list
    
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df.drop(columns=["categories"],inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    #drop duplicates
    df.drop_duplicates(inplace = True)
    
    
    return df


def save_data(df, database_filename):
    """Save dataframe into a db
    
    Args:
    df: dataframe. Dataframe
    database_filename: String. Path and name of the database that will be saved 
    
    Returns:
    Saved that frame
    """ 
    
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('ETL_part', engine, index=False, if_exists='replace')  

    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()