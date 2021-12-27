from utils import plot_correlation, read_sql, write_to_db, exists_table
import pandas as pd
import time

DISPLAY = True
VERBOSE = True
WIDTH=800
pd.set_option('display.width', WIDTH)
#np.set_printoption(linewidth=desired_width)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

#City of ATL police department redraws the zone and beat in order to improve their response time and ensure their coverage
#The change has been made effective of March 17, 2019. Since our data consists of data from 2009, we will need to manually
#reassign beat values to those affected neighborhoods
def reassign_beats(df_old):
    df_reassigned = df_old
    df_reassigned.loc[(df_reassigned['Neighborhood'].isin(['Blandtown', 'Bolton', 'Hills Park', 'Riverside', 'Whittier Mill Village']) )
                                                         & (df_reassigned['Beat'] == '203'), ['Beat']] = '103'
    df_reassigned.loc[(df_reassigned['Neighborhood'].isin(['Midtown'])) & (df_reassigned['Beat'] == '506'), ['Beat']] = '614'
    df_reassigned.loc[(df_reassigned['Neighborhood'].isin(['English Avenue', 'Midtown'])) & (df_reassigned['Beat'] == '103'), ['Beat']] = '506'
    df_reassigned.loc[(df_reassigned['Neighborhood'].isin(['Morningside/Lenox Park', 'Piedmont Heights'])) & (df_reassigned['Beat'] == '213'), ['Beat']] = '613'
    return df_reassigned

def create_features(crime):
    #create a map for crime type
    type_map = {'crime_type_id': [0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 6],
                 'UCR_Literal': ['AGG ASSAULT', 'AUTO THEFT', 'BURGLARY-NONRES', 'BURGLARY-RESIDENCE','HOMICIDE', 'LARCENY-FROM VEHICLE', 'LARCENY-NON VEHICLE',
                               'MANSLAUGHTER', 'ROBBERY-COMMERCIAL', 'ROBBERY-PEDESTRIAN', 'ROBBERY-RESIDENCE']
               }
    time_of_day_map = {'time_of_day_id': [0, 1, 2, 3],
                       'Time_of_day': ['Early_Morning', 'Morning', 'Afternoon', 'Evening']
                       }
    if VERBOSE:
        print(" ")
        print("Before manipulating, total instances and features: ", crime.shape)
        print("Original DF = ")
        print(crime.head(2))
        print(" ")
        print("Unique crime type = ", crime['UCR_Literal'].value_counts())
        print(" ")
        print("Unique Time_of_day = ", crime['Time_of_day'].value_counts())
    # Create dataframes with a map data and merge them
    df_type_map = pd.DataFrame(type_map)
    df_time_map = pd.DataFrame(time_of_day_map)
    df_add = crime.merge(df_type_map, on='UCR_Literal', how='inner')
    df_merged = df_add.merge(df_time_map, on='Time_of_day', how='inner')

    # Below is to load and merge the neighborhood with its numeric column. However, this attribute
    # doesn't seeem to help to predict
    #df_nbh = read_csv("neighborhood.csv")
    #df_add1 = df_add.merge(df_nbh, on='Neighborhood', how='inner')

    display_columns = ['Occur_Year', 'Occur_Month', 'day_of_week_id', 'Time_of_day', 'Zone', 'crime_type_id' ]
    # Convert Beat column value to integer from text and
    #df_merged['Beat'] = pd.to_numeric(df_merged['Beat'], errors='coerce').fillna(0).astype(np.int64)
    #df1 = df_merged.loc[(df_merged['Beat'] >= 200) & (df_merged['Beat'] < 300), display_columns]

    # Create a new column and extract first digit of Beat and assign it to zone to map Beat values into 6 different zones
    #df_new['zone'] = df_new['Beat'].astype(str).str[0]
    #df_new['zone'] = df_new['zone'].astype(int)  #conver it to integer
    df_merged['my_dates'] = pd.to_datetime(df_merged['Occur_Date'])
    df_merged['day_of_week_id'] = df_merged['my_dates'].dt.dayofweek
    df_merged['occur_day'] = df_merged['my_dates'].dt.day
    df_merged.to_csv("full_data.csv", index=False)
    plot_correlation(df_merged[display_columns])
    #extract month (1 - 12), year (2009 - 2019), day of week (7 dates), time of day (4: Morning, Afternoon, Early Morning, Evening)
    # zone (1-6), and crime_type_id (1-11)
    if VERBOSE:
        print(" ")
        print("After manipulating, total instances and features: ", df_merged.shape)
        print("New DF = ")
        print(df_merged.head(2))
    df_ml_data = df_merged[display_columns]

    if VERBOSE:
        print(" ")
        print("Feature data, total instances and features: ", df_ml_data.shape)
        print("Feature data = ")
        print(df_ml_data.head(2))

    newDF = encode_categorical_data(df_merged)
    return newDF

def encode_categorical_data(df):
    # Convert features with integer type into string. So that we can encode
    #df['Occur_Year'] = df['Occur_Year'].astype(str)
    df['Occur_Month'] = df['Occur_Month'].astype(str)
    df['day_of_week_id'] = df['day_of_week_id'].astype(str)

    # Encode the data
    categorical_columns = df[['Occur_Year', 'Occur_Month', 'day_of_week_id', 'Time_of_day', 'Zone']]
    encodedDF = pd.get_dummies(categorical_columns)
    newDF = pd.concat([encodedDF, df['crime_type_id']], axis=1)
    if VERBOSE:
        print(" ")
        print("newDF data, total instances and features: ", newDF.shape)
        print("newDF data = ")
        print(newDF.head(2))
    return newDF

def transform_clean_data(db_file, tbname_crime_mod, df_raw):
    #Remove a first record that is basically a header of raw file
    df = df_raw[1:len(df_raw)]

    if VERBOSE:
        print(" ")
        print("Before filtering, total instances = ", df.shape[0])

    # Create a feature, 'Time_of_day' using 'Occur_time'
    df = df.assign(Time_of_day = df.apply(timeOfDay, axis=1))

    # Remove undefined time_of_day
    df = df[df.Time_of_day != 'Undefined']
    if VERBOSE:
        print("After filtering 'Undefined' time of day, total instances = ", df.shape[0])

    # Create a feature, 'Occur_year' and 'Occur_month' using 'Occur_date'
    df["Occur_Year"] = df.apply(year, axis=1)
    df["Occur_Month"] = df.apply(month, axis=1)

    # filter case before 2009, blank neighborhood
    df = df[df.Occur_Year > 2008]
    if VERBOSE:
        print("After filtering a year older than 2009, total instances = ", df.shape[0])
    df = df[df.Neighborhood != '']
    if VERBOSE:
        print("After filtering a neighborhood, total instances = ", df.shape[0])

    # Drop beat that is null
    df = df.loc[df['Beat'] != '']
    if VERBOSE:
        print("after filtering 'Beat' that is null, total instances = ", df.shape[0])
    # Reassign Beat
    df_new = reassign_beats(df)

    # Create a new column and extract first digit of Beat and assign it to zone to map Beat values into 6 different zones
    df_new['Zone'] = df_new['Beat'].astype(str).str[0]
    #df_new['Zone'] = df_new['Zone'].astype(int)  #conver it to integer

    # Create a feature, crime_type
    df_new["Crime_type"] = df_new.apply(gen_crime_type, axis=1)
    df = df[df.Crime_type != '']
    if VERBOSE:
        print("After filtering a crime type, total instances = ", df.shape[0])

    # Convert Occur_Date to datetime
    df["Occur_Date"] = pd.to_datetime(df["Occur_Date"], format='%Y-%m-%d', errors='ignore')

    write_to_db(db_file, df, tbname_crime_mod, True)

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

def year(s):
    return int(s['Occur_Date'][:4])

def month(s):
    return int(s['Occur_Date'][5:7])

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

def run_manipulate_data():
    database_file = 'Project.db'
    tbname_crime_raw = 'crime_raw'
    tbname_crime_mod = 'crime_mod'
    tbname_feature = 'ML_FEATURE'
    start_time = time.time()
    if DISPLAY:
        print(" ")
        print("manipulate_data.run_manipulate_data()::Start = %s" % (time.ctime()))
        print(" ")

    # Transform and clean data. Time_of_day and crime_type
    df_crime_raw = read_sql(database_file, tbname_crime_raw)
    transform_clean_data(database_file, tbname_crime_mod, df_crime_raw)

    # in order for fast-processing, check if feature data is created in the db. If so, skip creating the feature
    # load the feature data from db directly and perform ML
    if (not exists_table(database_file, tbname_feature)):
        df_crime = read_sql(database_file, tbname_crime_mod)
        df_features = create_features(df_crime)
        write_to_db(database_file, df_features, tbname_feature)

    duration = time.time() - start_time
    if DISPLAY:
        print(" ")
        print("manipulate_data.run_manipulate_data()::End= %s, Duration= %f" % (time.ctime(), duration))
        print(" ")