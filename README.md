# Disaster Response Pipeline Project

This is a project done for Udacity  Data Scientist Nano degree program. This projects requires to create a machine learning model to predict classification for messages sent during disaster in right category. The data set is Provided by Figure Eight team and contains real messages set during disaster events.

##Data Files
There are two csv files that are used in this project. 
1. Disaster categories: This files categories the disaster messages in 36 categories. Each message is identified by a message ID and has been categorized in this file. 
2. Disaster messages: This file contain real messages sent during Disaster events, it contains messages from social media, news and direct messages. It give the message body in english and in the language it was sent by the persons during disaster event.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
