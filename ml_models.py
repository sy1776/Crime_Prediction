from utils import read_sql, write_to_db, exists_table, save_model, plot_cm, plot_grouped_bar, drop_table, read_csv
from manipulate_data import create_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
import pandas as pd
import time
import numpy as np

#np.set_printoptions(threshold=sys.maxsize)
#pd.set_option('display.max_rows', sys.maxsize)
#pd.set_option('display.max_columns', 20)
CLASSIFIERS = ['Logistic_Regression', 'Decision_Tree', 'KNN', 'Random_Forest']
RANDOM_STATE = 0
VERBOSE = True
DISPLAY = True

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

def run_ml_models():
    database_file = 'Project.db'
    tbname_crime_mod = 'Crime_MOD'
    tbname_feature = 'ML_FEATURE'
    tbname_ml_score = 'ML_SCORE'
    start_time = time.time()
    if DISPLAY:
        print("models.run_ml_models()::Start = %s" % (time.ctime()) )

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