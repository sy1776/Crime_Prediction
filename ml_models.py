from utils import read_sql, write_to_db, exists_table, save_model, drop_table, read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np

#np.set_printoptions(threshold=sys.maxsize)
#pd.set_option('display.max_rows', sys.maxsize)
#pd.set_option('display.max_columns', 20)
CLASSIFIERS = ['Logistic_Regression', 'Decision_Tree', 'KNN', 'Random_Forest']
RANDOM_STATE = 0
VERBOSE = False
DISPLAY = True

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
    df_ml_data = df_new[display_columns]
    return df_ml_data


def logistic_regression_pred(X_train, Y_train, X_test):
    # train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_train
    # use default params for the classifier
    model = LogisticRegression(random_state=RANDOM_STATE)
    model.fit(X_train, Y_train)

    save_model(model, 0)

    return model.predict(X_test)



def svm_pred(X_train, Y_train, X_test):
    # train a SVM classifier using X_train and Y_train. Use this to predict labels of X_train
    # use default params for the classifier
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    #model = LinearSVC(random_state=RANDOM_STATE)
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)

    return model.predict(X_test)

def decisionTree_pred(whichOne, X_train, Y_train, X_test):
    # train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_train
    # use max_depth as 5
    model = DecisionTreeClassifier(criterion=whichOne, max_depth=5, min_samples_leaf=8, random_state=RANDOM_STATE)
    model.fit(X_train, Y_train)

    # DT Gini Index is 2 and DT Information gain is 3
    modelType = 1
    if (whichOne == 'entropy'):
        modelType = 4

    save_model(model, modelType)

    return model.predict(X_test)

def randomForest_pred(X_train, Y_train, X_test):
    # Random Forest
    # Create Model with configuration
    model = RandomForestClassifier(n_estimators=70,  # Number of trees
                                      min_samples_split=30,
                                      bootstrap=True,
                                      max_depth=50,
                                      min_samples_leaf=25)
    model.fit(X_train, Y_train)
    save_model(model, 3)

    return model.predict(X_test)

def kNearestNeighbor_pred(X_train, Y_train, X_test):
    model = KNeighborsClassifier(n_neighbors=50)
    model.fit(X_train, Y_train)
    save_model(model, 2)

    return model.predict(X_test)

def neuralNetwork_pred(X_train, Y_train, X_test):
    model = MLPClassifier(solver='adam',
                             alpha=1e-5,
                             hidden_layer_sizes=(40,),
                             random_state=RANDOM_STATE,
                             max_iter=1000
                             )
    model.fit(X_train, Y_train)
    save_model(model, 5)

    return model.predict(X_test)

def classification_metrics(Y_pred, Y_true):
    # NOTE: It is important to provide the output in the same order
    acc = accuracy_score(Y_true, Y_pred)
    #auc_ = roc_auc_score(Y_true, Y_pred)
    precision = precision_score(Y_true, Y_pred, average="weighted")
    recall = recall_score(Y_true, Y_pred, average="weighted")
    f1score = f1_score(Y_true, Y_pred, average='micro')

    return acc, precision, recall, f1score

def display_metrics(classifierName, Y_pred, Y_true):
    acc, precision, recall, f1score = classification_metrics(Y_pred, Y_true)
    if VERBOSE:
        print("______________________________________________")
        print("Classifier: " + classifierName)
        print("Accuracy: " + str(acc))
        #print("AUC: " + str(auc_))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1-score: " + str(f1score))
        print("______________________________________________")
        print("")
    plot_cm(classifierName, Y_pred, Y_true)
    return [acc, precision, recall, f1score]

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

def run_ml_models():
    database_file = 'Project.db'
    tbname_crime_mod = 'Crime_MOD'
    tbname_feature = 'ML_FEATURE'
    tbname_ml_score = 'ML_SCORE'
    start_time = time.time()
    if DISPLAY:
        print("models.run_ml_models()::Start = %s" % (time.ctime()) )

    #in order for fast-processing, check if feature data is created in the db. If so, skip creating the feature
    #load the feature data from db directly and perform ML
    if (not exists_table(database_file, tbname_feature)):
        df_crime = read_sql(database_file, tbname_crime_mod)
        df_features = create_features(df_crime)
        write_to_db(database_file, df_features, tbname_feature)

    df_crime_ml = read_sql(database_file, tbname_feature)

    # Split a whole dataset into train and test sets. Train data size is 80% of whole dataset
    #X_train, X_test, Y_train, Y_test = train_test_split(df_crime_ml.iloc[:, :-1], df_crime_ml.iloc[:, -1], test_size=0.2, random_state=RANDOM_STATE)

    # Split a dataset into train and test sets. Data prior to 2019 will be train set and
    # data in 2019 will be test set. Remove a first column which is occur_year
    train = df_crime_ml[(df_crime_ml['Occur_Year'] < 2019 )]
    test = df_crime_ml[(df_crime_ml['Occur_Year'] >= 2019 )]
    X_train = train.iloc[:, 1:-1]
    Y_train = train.iloc[:, -1]
    X_test = test.iloc[:, 1:-1]
    Y_test = test.iloc[:, -1]


    lr_score = display_metrics("Logistic Regression", logistic_regression_pred(X_train, Y_train, X_test), Y_test)
    dt_gini_score = display_metrics("Decision Tree with gini index", decisionTree_pred("gini", X_train, Y_train, X_test), Y_test)
    dt_gain_score = display_metrics("Decision Tree with information gain", decisionTree_pred("entropy", X_train, Y_train, X_test), Y_test)
    knn_score = display_metrics("KNN", kNearestNeighbor_pred(X_train, Y_train, X_test), Y_test)
    rf_score = display_metrics("Random Forest", randomForest_pred(X_train, Y_train, X_test), Y_test)
    #nn_score = display_metrics("Neural Network", neuralNetwork_pred(X_train, Y_train, X_test), Y_test)
    #svm_score = display_metrics("SVM", svm_pred(X_train, Y_train, X_test), Y_test)

    plot_grouped_bar([lr_score, dt_gini_score, knn_score, rf_score])  # Plot the grouped bar chart

    # Append the name of classifier to scores returned from each classifiers
    lr_score.append(CLASSIFIERS[0])
    dt_gini_score.append(CLASSIFIERS[1])
    knn_score.append(CLASSIFIERS[2])
    rf_score.append(CLASSIFIERS[3])

    np_score = np.array([lr_score, dt_gini_score, knn_score, rf_score])  # Convert it to numpy array
    df_scores = pd.DataFrame({'Accuracy': np_score[:, 0], 'Precision': np_score[:, 1], 'Recall': np_score[:, 2],
                              'F1_score': np_score[:, 3], 'Classifier': np_score[:, 4]})
    write_to_db(database_file, df_scores, tbname_ml_score)

    duration = time.time() - start_time
    if DISPLAY:
        print("models.run_ml_models()::End= %s, Duration= %f" % (time.ctime(), duration))