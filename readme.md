**Program to extract features from google place reviews and save them to database**

Set up:
1. Create connString.txt file in a root directory with
    connection string to your database i.e. 
    DRIVER={ODBC Driver 13 for SQL Server}; SERVER={your server name}; PORT={usually 1433}; DATABASE={database name}; UID={user name, try "sa"}; PWD={password};
2. Run "pip install -r requirements.txt" to install required python libraries

Project structure:
* Parser.py - allows access to database data for example:
      get_reviews_df - returns every english,nonempty review content with id as pandas data frame
* FeatureExtractor.py - using python nlp libraries, it extracts features like gender, sentiment or language
    and save obtained data to database
* Main.py - application's start point (Ctrl+Shift+F10), contains also implementation 
    for "topic modeling" that creates LDA_Visualization2.html          
