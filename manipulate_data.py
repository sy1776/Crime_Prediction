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
    """"
    type_map = {'crime_type_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 'UCR_Literal': ['AGG ASSAULT', 'AUTO THEFT', 'BURGLARY-NONRES', 'BURGLARY-RESIDENCE','HOMICIDE', 'LARCENY-FROM VEHICLE', 'LARCENY-NON VEHICLE',
                               'MANSLAUGHTER', 'ROBBERY-COMMERCIAL', 'ROBBERY-PEDESTRIAN', 'ROBBERY-RESIDENCE']
               }
    """
    type_map = {'crime_type_id': [0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 6],
                 'UCR_Literal': ['AGG ASSAULT', 'AUTO THEFT', 'BURGLARY-NONRES', 'BURGLARY-RESIDENCE','HOMICIDE', 'LARCENY-FROM VEHICLE', 'LARCENY-NON VEHICLE',
                               'MANSLAUGHTER', 'ROBBERY-COMMERCIAL', 'ROBBERY-PEDESTRIAN', 'ROBBERY-RESIDENCE']
               }
    time_of_day_map = {'time_of_day_id': [0, 1, 2, 3],
                       'Time_of_day': ['Early Morning', 'Morning', 'Afternoon', 'Evening'] #49,735 records for Early_Morning is disregarded. Total recs: 272,569 vs 322,256
                       #'Time_of_day': ['Early_Morning', 'Morning', 'Afternoon', 'Evening']
                       }
    if VERBOSE:
        print(" ")
        print("Before manipulating, total instances and features: ", crime.shape)
        print("Original DF = ")
        print(crime.head(20))
    # Create dataframes with a map data and merge them
    df_type_map = pd.DataFrame(type_map)
    df_time_map = pd.DataFrame(time_of_day_map)
    df_add = crime.merge(df_type_map, on='UCR_Literal', how='inner')
    df_merged = df_add.merge(df_time_map, on='Time_of_day', how='inner')

    # Below is to load and merge the neighborhood with its numeric column. However, this attribute
    # doesn't seeem to help to predict
    #df_nbh = read_csv("neighborhood.csv")
    #df_add1 = df_add.merge(df_nbh, on='Neighborhood', how='inner')

    display_columns = ['Occur_Year', 'Occur_Month', 'day_of_week_id', 'time_of_day_id', 'zone', 'crime_type_id' ]
    # Convert Beat column value to integer from text and
    #df_merged['Beat'] = pd.to_numeric(df_merged['Beat'], errors='coerce').fillna(0).astype(np.int64)
    #df1 = df_merged.loc[(df_merged['Beat'] >= 200) & (df_merged['Beat'] < 300), display_columns]

    #drop beat that is null
    df_merged = df_merged.loc[df_merged['Beat'] != '']
    df_new = reassign_beats(df_merged)

    # Create a new column and extract first digit of Beat and assign it to zone to map Beat values into 6 different zones
    df_new['zone'] = df_new['Beat'].astype(str).str[0]
    df_new['zone'] = df_new['zone'].astype(int)  #conver it to integer
    df_new['my_dates'] = pd.to_datetime(df_new['Occur_Date'])
    #df_new['day_of_week'] = df_new['my_dates'].dt.day  #this will give us day of week
    df_new['day_of_week_id'] = df_new['my_dates'].dt.dayofweek
    df_new['occur_day'] = df_new['my_dates'].dt.day
    #print(df_new.loc[(df_new['Neighborhood'].isin(['Blandtown', 'Bolton', 'Hills Park', 'Riverside', 'Whittier Mill Village']))& (df_new['Beat'] == '103'),
    #                 ['Beat', 'Neighborhood']])
    df_new.to_csv("full_data.csv", index=False)
    plot_correlation(df_new[display_columns])
    #extract month (1 - 12), year (2009 - 2019), day of week (7 dates), time of day (4: Morning, Afternoon, Early Morning, Evening)
    # zone (1-6), and crime_type_id (1-11)
    if VERBOSE:
        print(" ")
        print("After manipulating, total instances and features: ", df_new.shape)
        print("New DF = ")
        print(df_new.head(20))
    df_ml_data = df_new[display_columns]

    if VERBOSE:
        print(" ")
        print("Feature data, total instances and features: ", df_ml_data.shape)
        print("Feature data = ")
        print(df_ml_data.head(20))

    return df_ml_data

def run_manipulate_data():
    database_file = 'Project.db'
    tbname_crime_mod = 'Crime_MOD'
    tbname_feature = 'ML_FEATURE'
    start_time = time.time()
    if DISPLAY:
        print("manipulate_data.run_manipulate_data()::Start = %s" % (time.ctime()))

    # in order for fast-processing, check if feature data is created in the db. If so, skip creating the feature
    # load the feature data from db directly and perform ML
    if (not exists_table(database_file, tbname_feature)):
        df_crime = read_sql(database_file, tbname_crime_mod)
        df_features = create_features(df_crime)
        write_to_db(database_file, df_features, tbname_feature)

    duration = time.time() - start_time
    if DISPLAY:
        print("")
        print("manipulate_data.run_manipulate_data()::End= %s, Duration= %f" % (time.ctime(), duration))