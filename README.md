==Overview==

For this experiment, question I asked was "Can crime type be predicted for specific place at a certain time with high accuracy?

Fortunately, there are numerous police departments in America that provides the crime incidents everybody can access. So, I've chosen a city of Atlanta crime data and did followings:

==What has done==
* Data Collection: Downloaded crime incidents for 10 years, 2009 - 2019. Total 343K incidents. To be exact: 342,915 records.

* Data Cleaning: Removed duplicates via OpenRefine. Total number came down to 334,938.
                 Filtered incomplete crime incidnets (ex: blank neighborhood, crime occurred year=1926) via <tt>manipulate_data.py</tt> using pandas lib. 
                 
* Analysis/Visualization: Visualization like correlation, confusion matrix, etc done via utils.py using matplotlib and seaborn libs. 
                          Analysis done via <tt>manipulate_data.py</tt> using pandas lib. 
                          
* Feature Engineering/Tranformation: Couple of runs with original crime data on different machine learning classifiers scored
  below 50% accuracy which is very low. So, picked features that mattered. crime occur date, neigborhood/beat, and crime types and 
  following methods were implemented via <tt>manipulate_data.py</tt> using pandas lib:  
  
  1. Grouped 11 crime types into 6 categories. For example, ‘BURGLARY-NONRES’ and 
     ‘BURGLARY-RESIDENCE’ were merged into ‘BURGLARY’. This has become the label.  
  
  2. Instead of using 244 neighborhoods as location attribute, explored it to use 6 zones  
     City of Atlanta logically divided for their patrolling area. It turns out that ‘Beat’ code  
     could be used to represent 6 numeric zones. So, converted it to 6 zones.  
  
  3. Crime occur date is broken into year, month, day of week (Monday - Sunday), and time of day id  
     representing 4 different day, early morning to evening  
  
  4. All features in above #2 and #3 were converted into indicator variables from categorical variables.  
				         
				      
* Run ML Models: Two different Decision Trees (Gini Index, Information Gain), K-NN, Logistic Regression were performed via <tt>ml_models.py</tt> using scikit-learn lib.

Python code, <tt>import_data.py</tt> which is a first module that gets executed will import the raw data into 'crime_raw' table in the sqlite database.
After cleaning data and data engineering/transforming, modified data will be written to 'crime_mod' and 'ml_feature' tables.

Note that SQLite library does not need to be installed on the machine for above codes to be run.

==How to install and execute==  

Please refer to CODE_INFO.txt

