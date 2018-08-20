import quandl 
import pandas as pd
import math
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#Code to extract data for different stocks

quandl.ApiConfig.api_key = 'hsCs9wswHSofQz_dxhom'
def retrieve_data(ticker):

    start="2001-07-27"
    end="2018-07-27"
    db_code="EOD"
    ticker = ticker
    ip_code = db_code+'/'+ticker
    d = quandl.get(ip_code, start_date=start, end_date=end, collapse="daily")
    df = pd.DataFrame(data=d)
    df.drop(df.columns[[5,6,7,8,9,10,11]], axis =1, inplace=True)
    #print (df)
    return(df) 

def main():
    ticker = 'KO'
    df = pd.DataFrame()
    df = retrieve_data(ticker)
    print (df)
    df.to_csv('cola1.csv')  

if __name__ == "__main__":
	main()
