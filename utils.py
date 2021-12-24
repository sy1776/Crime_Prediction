from sqlalchemy import create_engine
from sqlite3 import Error
from sklearn.metrics import confusion_matrix
import pandas as pd
import sqlite3
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

MODEL_NAME = {0:'LR_MODEL', 1:'DT_GINI_MODEL', 2:'KNN_MODEL', 3:'RF_MODEL', 4:'DT_IG_MODEL', 5:'NN_MODEL'}
CLASSIFIERS = ['Logistic_Regression', 'Decision_Tree', 'KNN', 'Random_Forest']

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def close_connection(db):
    db.close()

def exists_table(db_file, table_name):
    isTableExisted = False
    db = create_connection(db_file)
    cursor = db.cursor()

    try:
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{}'".format(table_name))
    except Error as e:
        print(e)

    if cursor.fetchone()[0] == 1:
        isTableExisted = True
        print('Table: %s exists.' %table_name)
    else:
        print('Table: %s does not exist.' %table_name)

    close_connection(db)
    return isTableExisted

def create_table(db_file, table_name):
    db = create_connection(db_file)
    cursor = db.cursor()

    cursor.execute("CREATE TABLE {} (month int, year int, day_of_week text, time_of_day text, zone int, crime_type_id float)".format(table_name))
    print("Created a table: %s !!!" %table_name)
    db.commit()
    close_connection(db)

def drop_table(db_file, table_name):
    db = create_connection(db_file)
    cursor = db.cursor()

    cursor.execute(
        "DROP TABLE {} ".format(table_name))
    print("Dropped a table: %s !!!" % table_name)
    db.commit()
    close_connection(db)

def read_sql(db_file, table_name):
    conn = create_connection(db_file)
    df = pd.read_sql_query("SELECT * FROM " + table_name, conn)
    close_connection(conn)

    return df

def write_to_db(db_file, df, table, ix=False):
    engine = create_engine('sqlite:///' + db_file, echo=False)
    df.to_sql(table, con=engine, if_exists='replace', index=ix)

def save_model(model, model_id):
    # save the model to disk
    pickle.dump(model, open(MODEL_NAME[model_id], 'wb'))

def load_model(model_id):
    # load the model from disk
    loaded_model = pickle.load(open(MODEL_NAME[model_id], 'rb'))
    return loaded_model

def read_csv(filepath):
    # Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    neighborhood_df = pd.read_csv(filepath)
    return neighborhood_df

def plot_correlation(df_cor):
    # Using Pearson Correlation
    title = "correlation"
    plt.figure(figsize=(15, 10))
    cor = df_cor.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.title(title)
    plt.savefig(title)
    plt.close()

def plot_cm(classifierName, Y_pred, Y_true):
    label = ['AGG ASSAULT', 'AUTO THEFT', 'BURGLARY', 'HOMICIDE', 'LARCENY', 'MANSLAUGHTER', 'ROBBERY']
    title = "Confusion Matrix of " + classifierName
    cm = confusion_matrix(Y_true, Y_pred)
    plt.figure(figsize=(8,4))
    plt.rcParams.update({'font.size': 7})
    plt.title(title)
    sns.heatmap(cm, annot=True, cmap=plt.cm.Reds, fmt="d", xticklabels=label, yticklabels=label)  # annot=True to annotate cells
    plt.savefig(title)
    plt.close()

def plot_grouped_bar(score):
    length = len(CLASSIFIERS)
    title = "Classification Score"
    width = 0.15  # the width of the bars
    # Set position of bar on X axis
    r1 = np.arange(length)
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]
    np_score = np.array(score)

    plt.rcParams.update({'font.size': 8})
    plt.figure(figsize=(8, 4))

    plt.bar(r1,  np_score[:, 0], width=width, edgecolor='white', label='Accuracy')
    plt.bar(r2,  np_score[:, 1], width=width, edgecolor='white', label='Precision')
    plt.bar(r3,  np_score[:, 2], width=width, edgecolor='white', label='Recall')
    plt.bar(r4,  np_score[:, 3], width=width, edgecolor='white', label='F1-score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.title(title)
    plt.ylabel('Scores')
    plt.xticks([r + width for r in range(length)], CLASSIFIERS)
    plt.legend(loc='best')

    plt.savefig(title)
    plt.close()
