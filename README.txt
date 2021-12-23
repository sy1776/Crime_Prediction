1. DESCRIPTION
  There are 2 code packages for this project.
  1) Python codes - there are 5 Python scripts and each scripts are described below in detail. Their job essentially is to import the data
     from the file and performs various algorithms and saves output to database.
     a. main.py - wrapper script to call and execute other scripts.
     b. import_clean_data.py - import the data from csv file into a table, crime_raw in sqlite database
        It then cleans and transforms the data from crime_raw into crime_mod table.
     c. exp_smoothing_model.py - performs an exponential smoothing on the data from crime_mod table and writes output to database.
     d. ml_models.py - creates a feature data from crime_mod and trains the data with Machine Learning classifiers and then,
        trained models are saved into the same folder and output of classification score is saved to database.
     e. predict.py - loads trained models and predicts 2,016 real data with those models and predicted outcome is saved to database.

  2) ASP.NET codes - there are 6 web form files (.aspx) with each source files (.cs). Also, including web config file and app start folder.
     The description are following:
     a. default.aspx and default.cs - home page where user can review the project summary and navigate to Map View user interaction Module,
        Historic Data View Module and Data Prediction Module.
     b. map.aspx and map.cs - user can interact with the website by selecting or entering the filter conditions from the combo box.
        Then resulting data will populate as markers to show on the google map.
     c. dataview.aspx and dataview.cs - user can interact with the website by selecting or entering the filter conditions from the combo box.
        Then resulting data will populate in gridview tables, line charts, and bar charts.
     d. selection.aspx and selection.cs – navigation page where user can go to the Holt-Winter Prediction Module and Machine Learning Prediction Module.
     e. Prediction.aspx and Prediction.cs - user can interact with the website by selecting or entering the filter conditions from the combo box.
        Then prediction data will show in a line chart.
     f. PredictionML.aspx and PredictionML.cs - user can interact with the website by selecting or entering the filter conditions from the combo box.
        Then prediction data will show in a gridview table. Classified score data will show in gridview table and bar chart at page load.
     g. web.config file – it contain database connection string and library.
     i. App_Start folder – it contains website’s essential libraries, for example devexpress UI libraries.

2. INSTALLATION
  System Requirement: following libraries are needed for python codes: Panda, Numpy, seaborn, matplotlib, scikit-learn, statsmodels.
  For ASP.NET codes, Window Server 2012R or newer version is required along with ASP.NET 4.5 and IIS Web Deploy features. Visual Studio 2017 or newer is needed too.
  1) Go to CODE folder.
  2) Download 2 data files, "COBRA-2019" and "COBRA-2009-2018" from below Atlanta police website:
     http://www.atlantapd.org/i-want-to/crime-data-downloads
  3) Merge the files from step b into one single file, "COBRA-2009-2019.csv" and remove duplicates using a tool like OpenRefine.
  4) Copy and paste the ASP.NET Code folder to a designate location in the windows based server environment.

3. EXECUTION
  1) Ensure that data file, "COBRA-2009-2019.csv" is in the same folder
  2) Run a wrapper script, "main.py" like this - "python main.py"
     It will execute below scripts in following order:
      import_clean_data.py
      exp_smoothing_model.py
      ml_models.py
      predict.py
  3) Open Visual Studio Management Tool and build the solution package. Folder Deploy the package to a designate location in the windows Server.
  4) Open IIS and create a new website.
  5) Go to advance setting and point the Folder Deployed fold as the root directory.
  6) If there’s a domain name available, we can add that to the settings.

4. DEMO VIDEO
   https://youtu.be/AgBzL1G6rII

Our website is live and running and it is http://www.azsoftwaresolution.com/
