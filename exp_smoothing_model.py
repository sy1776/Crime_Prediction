import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from utils import read_sql, write_to_db

DISPLAY = True

def run_es_model():
    database_file = 'Project.db'
    tbname_crime_mod = 'Crime_MOD'

    start_time = time.time()
    if DISPLAY:
        print("exp_smoothing_model.run_es_model()::Start = %s" % (time.ctime()))

    df1 = read_sql(database_file, tbname_crime_mod)

    # List of Crime_type
    Crime_list = ["LARCENY", "BURGLARY", "ROBBERY", "HOMICIDE", "AUTO_THEFT", "AGG_ASSAULT", "All"]
    Zone_list = ['1', '2', '3', '4', '5', '6', "All"]
    Time_list = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', "All"]

    Month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * 11
    start_year = 2009
    Year = []
    for i in range(11):
        for j in range(12):
            Year.append(start_year + i)
    for i in range(2):
        Year.append(2020)
        Month.append(1 + i)
    Year_Month = pd.DataFrame()
    Year_Month['Year'] = Year
    Year_Month['Month'] = Month

    # Loop

    for crime_type in Crime_list:
        for time_of_day in Time_list:
            for zone in Zone_list:
                df_filtered = df1.copy()
                if (crime_type == "All") and (time_of_day == "All") and (zone == "All"):
                    df_filtered = df1.copy()
                elif (crime_type == "All") and (time_of_day == "All"):
                    df_filtered = df1[df1.Zone == zone]
                elif (crime_type == "All") and (zone == "All"):
                    df_filtered = df1[df1.Time_of_day == time_of_day]
                elif (time_of_day == "All") and (zone == "All"):
                    df_filtered = df1[df1.Crime_type == crime_type]
                elif crime_type == "All":
                    df_filtered = df1[(df1.Time_of_day == time_of_day) & (df1.Zone == zone)]
                elif time_of_day == "All":
                    df_filtered = df1[(df1.Crime_type == crime_type) & (df1.Zone == zone)]
                elif zone == "All":
                    df_filtered = df1[(df1.Crime_type == crime_type) & (df1.Time_of_day == time_of_day)]
                else:
                    df_filtered = df1[
                        (df1.Crime_type == crime_type) & (df1.Time_of_day == time_of_day) & (df1.Zone == zone)]

                df_crimeRate = df_filtered.copy()
                df_crimeRate = df_filtered.groupby(['Occur_Year', 'Occur_Month']).count().unstack(fill_value=0).stack()
                df_crimeRate = df_crimeRate[['Report_Number']]
                df_crimeRate = df_crimeRate.reset_index(level=['Occur_Year', 'Occur_Month'])
                df_crimeRate.columns = ['Year', 'Month', 'Crime_Rate']

                df_crimeRate = pd.merge(Year_Month, df_crimeRate, how='left', left_on=['Year', 'Month'],
                                        right_on=['Year', 'Month'])
                df_crimeRate = df_crimeRate.fillna(0)

                df_crimeRate['Date'] = df_crimeRate['Year'].map(str) + '-' + df_crimeRate['Month'].map(str) + '-1'
                df_crimeRate = df_crimeRate[['Date', 'Crime_Rate']]
                df_crimeRate['Date'] = pd.to_datetime(df_crimeRate["Date"], format="%Y-%m-%d")
                df_crimeRate.index = pd.date_range(freq='M', start='2009-1-1', periods=134)

                # Split data into train and test
                train = df_crimeRate.copy()[:128]
                Pred = df_crimeRate.copy()[120:]

                Holtwinter_model = ExponentialSmoothing(train['Crime_Rate'].astype(np.float64), seasonal_periods=12,
                                                        trend='add', seasonal='add')
                Holt_fit = Holtwinter_model.fit()
                Pred['Holt_Winter'] = Holt_fit.predict(start=Pred.iloc[0].name, end=Pred.iloc[-1].name).round(0)
                Pred['Holt_Winter'] = Pred['Holt_Winter'].clip(lower=0)

                # Convert Panda Dataframe to SQLite table
                Pred_table_name = crime_type + "_" + zone + "_" + time_of_day + "_Pred"
                write_to_db(database_file, Pred, Pred_table_name)

    duration = time.time() - start_time
    if DISPLAY:
        print("exp_smoothing_model.run_es_model()::End= %s, Duration= %f" % (time.ctime(), duration))
