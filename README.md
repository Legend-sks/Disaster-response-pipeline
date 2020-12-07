# Disaster Response Pipeline Project (GitHub Repository link https://github.com/Legend-sks/Udacity_project_2)

This is a project done for Udacity  Data Scientist Nano degree program. This projects requires to create a machine learning model to predict classification for messages sent during disaster in right category. The data set is Provided by Figure Eight team and contains real messages set during disaster events.

### Data Files
There are two csv files that are used in this project. 
1. Disaster categories: This files categories the disaster messages in 36 categories. Each message is identified by a message ID and has been categorized in this file. 
2. Disaster messages: This file contain real messages sent during Disaster events, it contains messages from social media, news and direct messages. It give the message body in english and in the language it was sent by the persons during disaster event.

### ETL and ML pipeline details
ETL pipeline :
Here I have created a data extract pipe-line which does below activities on the data set:
1. Load Data: Data from csv files is loaded and data frame is created for both the files.
2. Merge Datasets: Both data sets are mearged, primary key for this mearge is Message ID.
3. Create categories columns and fill in 0 or 1 for each message: New columns are for each category  is added to the data frame and values are populated.
4. Remove duplicate messages:  all duplicate messages in data set in deleted.

ML pipeline :
Here I have created a Machine Learning pipeline which models the data and predicts the categories for new messages:
1. Load data from the database created in ETL pipeline.
2. Tokenize text data : Normalize, tokenize, lemmatize text and remove stop words
3. Machine learning pipeline: here I have used TfidfTransformer,RandomForestClassifier and MultiOutputClassifier to create the model. After that Gridsearch was used to make model more accurate(permutations and combinations ot various parameters was tried and then final parameters were identified).
4. Split dataset, fit and test model: Split the model in train and test data set and fit the model on the data. Calculate the accuracy, precision and f1-scores and fine tune the data till ssatisfed with results.

Webapp page:
This gives a very high level description of data using visuals. here you can enter a new message the see which all categoeies it falls into.


### Files in this repository
there are total 9 Files in this repository:
1. App folder: 
    a) run.py
    b)Templates
        1)go.html
        2)master.html
2. Data Folder:
    a)disaster_categories.csv
    b)disaster_messages.csv
    c)process_data.py
3. models Folder
    a)trainClassifer.py
    b)calssifire.pkl: this file gets created when classifier file is execute
4. Readme file

### Imbalance in the dataset
You can see that there are some imbalances in the dataset. We dont have enough values\messages for certain categories, keeping that in mind the model output might always be accurate for messages whch sholud go in these categories. Some examples for such are water , child alone and offer categories.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
