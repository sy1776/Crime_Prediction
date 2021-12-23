import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from sqlite3 import Error
import pickle

MODEL_NAME = {0:'LR_MODEL', 1:'DT_GINI_MODEL', 2:'KNN_MODEL', 3:'RF_MODEL', 4:'DT_IG_MODEL', 5:'NN_MODEL'}

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