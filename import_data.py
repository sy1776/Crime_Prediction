import sqlite3

import pandas
import pandas as pd
import csv
import time
from utils import exists_table, drop_table, read_csv, read_sql, write_to_db

DISPLAY = True

def import_data():
    database_file = 'Project.db'
    tbname_crime_raw = 'crime_raw'
    if (exists_table(database_file, tbname_crime_raw)):
        drop_table(database_file, tbname_crime_raw)

    conn = sqlite3.connect(database_file)

    c = conn.cursor()

    # Create crime_raw table
    c.execute('''CREATE TABLE if not exists crime_raw
                 (Report_Number text, Report_Date smalldatetime, Occur_Date smalldatetime, Occur_Time int, Possible_Date smalldatetime, Possible_Time int, Beat int, Apartment_Office_Prefix text, Apartment_Number text, Location text, Shift_Occurence text, Location_Type text, UCR_Literal text, UCR_Num int, IBR_Code text, Neighborhood text, NPU text, Latitude float, Longitude float)''')

    # Read csv file to table
    with open('COBRA-2009-2019.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = next(reader)
        query = 'insert into crime_raw values ({0})'
        query = query.format(','.join('?' * len(data)))
        cursor = conn.cursor()
        cursor.execute(query, data)
        for data in reader:
            cursor.execute(query, data)
    conn.commit()

def run_import():
    start_time = time.time()
    if DISPLAY:
        print("import_clean_data.run_import()::Start = %s" % (time.ctime()) )

    import_data()

    duration = time.time() - start_time
    if DISPLAY:
        print("")
        print("import_clean_data.run_import()::End= %s, Duration= %f" % (time.ctime(), duration))