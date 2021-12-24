1. DESCRIPTION
  Below is the code packages for this project.
  1) Python codes - there are 5 Python scripts and each scripts are described below in detail. Their job essentially is to import the data
     from the file and performs various algorithms and saves output to database.
     a. main.py - wrapper script to call and execute other scripts.
     b. import_clean_data.py - import the data from csv file into a table, crime_raw in sqlite database
        It then cleans and transforms the data from crime_raw into crime_mod table.
     c. exp_smoothing_model.py - performs an exponential smoothing on the data from crime_mod table and writes output to database.
     d. ml_models.py - creates a feature data from crime_mod and trains the data with Machine Learning classifiers and then,
        trained models are saved into the same folder and output of classification score is saved to database.
     e. predict.py - loads trained models and predicts 2,016 real data with those models and predicted outcome is saved to database.


2. INSTALLATION
  System Requirement: following libraries are needed for python codes: Panda, Numpy, seaborn, matplotlib, scikit-learn, statsmodels.
  1) Go to CODE folder.
  2) Download 2 data files, "COBRA-2019" and "COBRA-2009-2018" from below Atlanta police website:
     http://www.atlantapd.org/i-want-to/crime-data-downloads
  3) Merge the files from step b into one single file, "COBRA-2009-2019.csv" and remove duplicates using a tool like OpenRefine.

3. EXECUTION
  1) Ensure that data file, "COBRA-2009-2019.csv" is in the same folder
  2) Run a wrapper script, "main.py" like this - "python main.py"
     It will execute below scripts in following order:
      import_clean_data.py
      exp_smoothing_model.py
      ml_models.py
      predict.py

4. DEMO VIDEO
   https://youtu.be/AgBzL1G6rII
