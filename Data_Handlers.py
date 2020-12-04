import pandas as pd

def data_preprocessing():
    # Read csv file into a pandas dataframe
    df = pd.read_csv("LBW_Dataset.csv")
    z=24 # Has result 0 till index 24
    df_1,df_2 = df.iloc[:z,:],df.iloc[z:,:] # splitting data according to result
    for i in df_1.columns.tolist():
        # Replace using median 
        median = df_1[i].median()
        df_1[i].fillna(median, inplace=True)
    for i in df_2.columns.tolist():
        # Replace using median 
        median = df_2[i].median()
        df_2[i].fillna(median, inplace=True)
    df = pd.concat([df_1,df_2]) # joining the cleaned dataframes
    df.to_csv('cleaned_LBW_Dataset.csv') # putting it into csv file

data_preprocessing()