from utils import load_model, write_to_db
import pandas as pd
import time

DISPLAY = True

def run_preidct():
    start_time = time.time()
    if DISPLAY:
        print("predict.run_preidct()::Start = %s" % (time.ctime()))

    database_file = 'Project.db'
    tbname_ml_pred = 'ML_PREDICTION'
    display_columns = ['classifier', 'month', 'day_of_week', 'time_of_day', 'zone', 'crime_type']
    day_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_of_day =['Early_Morning', 'Morning', 'Afternoon', 'Evening']
    classifiers = ['Logistic_Regression', 'Decision_Tree', 'KNN', 'Random_Forest']
    label = ['AGG ASSAULT', 'AUTO THEFT', 'BURGLARY', 'HOMICIDE', 'LARCENY', 'MANSLAUGHTER', 'ROBBERY']

    df_ml_pred = pd.DataFrame(columns=display_columns)
    count = 0
    for i in range(len(classifiers)):  # Loop through models 1 - 5 (Defined in Utils)
        result = [0] * 6  # declare fixed size list, length of 6
        result[0] = classifiers[i]
        model = load_model(i)
        for j in range(1, 13):  # Loop through months, 1 - 12 (Jan - Dec)
            test_data = [0] * 4
            test_data[0] = j
            result[1] = j
            for k in range(len(day_of_week)):
                test_data[1] = k
                result[2] = day_of_week[k]
                for l in range(len(time_of_day)):
                    test_data[2] = l
                    result[3] = time_of_day[l]
                    for m in range(1, 7):  # Loop through zone 1 - 6
                        test_data[3] = m
                        result[4] = m

                        pred = model.predict([test_data])
                        result[5] = label[pred.item()]
                        df_ml_pred.loc[count,:] = result
                        count += 1

    write_to_db(database_file, df_ml_pred, tbname_ml_pred)

    duration = time.time() - start_time
    if DISPLAY:
        print("predict.run_preidct()::End= %s, Duration= %f" % (time.ctime(), duration))
