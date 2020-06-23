# UDACITY PROJECTS
It is and repository to include my individual project of udacity program. 
# Installations
For this project Python and Jupyter Nootebook is needed.
# Project
Provide a brief basic introduction in all the data mining process and modeling it to predict somenthing, that basically consists on picking a dataset and get some insights with it.


# Projects

## Post Blog - CRISP DM

listings.csv = Datafile from kaggle airbnb base
calendar.csv = Datafile from kaggle airbnb base
reviews.csv = Datafile from kaggle airbnb base
post.ipynb = Code in Jupyter Notebook
post.html = Code in Html

## Licensing, Authors, Acknowledgements, etc.

Kaggle Aibnb DataBase

-----------------------------------------------------------

## Disaster_Reponse_Pipeline_Project

### Files

projeto_2 paste =  Disaster_Reponse_Pipeline_Project

          app = Folder to run web
          data = Folder to run web
          models = Folder to run web
          DisasterResponse.db	= Database generated
          ETL Pipeline Preparation.ipynb = process of etl preparation	
          ML Pipeline Preparation.ipynb = process of machine learning pipeline
          categories.csv = file with the categories of the message
          messages.csv = file with messages
          
   ### Running Disaster Reponse Pipeline Project 

   1. ETL process
    * Run the following command: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

   2. Machine Learning Pipeline
    * Run the following command: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

   3. Run the web app
    * Run the following command in the app directory: python run.py

### Licensing, Authors, Acknowledgements, etc.

Udacity Course
