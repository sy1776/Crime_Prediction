import sqlite3
import pandas as pd
import csv
import time
from utils import exists_table, drop_table, read_sql, write_to_db

DISPLAY = True

def import_data():
    database_file = 'Project.db'
    tbname_crime_raw = 'Crime_Raw'
    if (exists_table(database_file, tbname_crime_raw)):
        drop_table(database_file, tbname_crime_raw)

    conn = sqlite3.connect(database_file)

    c = conn.cursor()

    # Create sets table
    c.execute('''CREATE TABLE Crime_Raw
                 (Report_Number text, Report_Date smalldatetime, Occur_Date smalldatetime, Occur_Time int, Possible_Date smalldatetime, Possible_Time int, Beat int, Apartment_Office_Prefix text, Apartment_Number text, Location text, Shift_Occurence text, Location_Type text, UCR_Literal text, UCR_Num int, IBR_Code text, Neighborhood text, NPU text, Latitude float, Longitude float)''')


    # Read csv file to table
    with open('COBRA-2009-2019.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = next(reader)
        query = 'insert into Crime_Raw values ({0})'
        query = query.format(','.join('?' * len(data)))
        cursor = conn.cursor()
        cursor.execute(query, data)
        for data in reader:
            cursor.execute(query, data)
    conn.commit()

def clean_data():
    database_file = 'Project.db'
    tbname_crime_raw = 'Crime_Raw'
    tbname_crime_mod = 'Crime_MOD'

    df1 = read_sql(database_file, tbname_crime_raw)

    # Remove Header
    df1 = df1[1:len(df1)]

    def timeOfDay(c):
        if type(c['Occur_Time']) == int:
            if 0 <= c['Occur_Time'] < 600:
                return 'Early_Morning'
            elif 600 <= c['Occur_Time'] < 1200:
                return 'Morning'
            elif 1200 <= c['Occur_Time'] < 1800:
                return 'Afternoon'
            elif 1800 <= c['Occur_Time'] <= 2359:
                return 'Evening'
            else:
                return 'Undefined'
        else:
            return 'Undefined'


    df1['Time_of_day'] = df1.apply(timeOfDay, axis=1)

    # Remove undefined time_of_day
    df1 = df1[df1.Time_of_day != 'Undefined']
    df1 = df1.loc[df1['Time_of_day'] != 'Undefined']

    def year(s):
        return int(s['Occur_Date'][:4])

    def month(s):
        return int(s['Occur_Date'][5:7])

    df1["Occur_Year"] = df1.apply(year, axis=1)
    df1["Occur_Month"] = df1.apply(month, axis=1)


    def gen_crime_type(s):
        if "LARCENY" in s['UCR_Literal']:
            return "LARCENY"
        elif "BURGLARY" in s['UCR_Literal']:
            return "BURGLARY"
        elif "ROBBERY" in s['UCR_Literal']:
            return "ROBBERY"
        elif s['UCR_Literal'] == "HOMICIDE" or s['UCR_Literal'] == "MANSLAUGHTER":
            return "HOMICIDE"
        elif s['UCR_Literal'] == "AUTO THEFT":
            return "AUTO_THEFT"
        elif s['UCR_Literal'] == "AGG ASSAULT":
            return "AGG_ASSAULT"
        else:
            return s['UCR_Literal']

    df1['Zone'] = df1['Beat'].astype(str).str[0]
    df1["Crime_type"] = df1.apply(gen_crime_type, axis=1)


    # filter case before 2009, blank crime type, blank neighborhood
    df1 = df1[df1.Occur_Year > 2008]
    df1 = df1[df1.Crime_type != '']
    df1 = df1[df1.Neighborhood != '']

    # Convert Occur_Date to datetime
    df1["Occur_Date"] = pd.to_datetime(df1["Occur_Date"], format='%Y-%m-%d', errors='ignore')
    df1["Occur_Date"].dtype

    write_to_db(database_file, df1, tbname_crime_mod, True)

def run_import_clean():
    start_time = time.time()
    if DISPLAY:
        print("import_clean_data.run_import_clean()::Start = %s" % (time.ctime()) )

    import_data()
    clean_data()

    duration = time.time() - start_time
    if DISPLAY:
        print("")
        print("import_clean_data.run_import_clean()::End= %s, Duration= %f" % (time.ctime(), duration))