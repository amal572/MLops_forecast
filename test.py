import requests
import pandas as pd
import csv
import json

csvFilePath = r'test.csv'
jsonFilePath = r'test.json'

def make_json(csvFilePath, jsonFilePath):
     
    # create a dictionary
    data = {}
     
    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
         
        # Convert each row into a dictionary 
        # and add it to data
        for rows in csvReader:
             
            # Assuming a column named 'No' to
            # be the primary key
            key = rows['COMPANY_KEY']
            data[key] = rows
 
    # Open a json writer, and use the json.dumps() 
    # function to dump data
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))

make_json(csvFilePath, jsonFilePath)

ride = {
        "date": ["21/10/2023"],
        "COMPANY_KEY": ["4"],
        "COMPANY_ITEM_KEY": ["31379"],
        "BILLING_COUNTRY_KEY": ["16"],
        "APPLE_WEEK_DATE": ["21/10/2023"],
        "APPLE_YEAR": ["2024"],
        "APPLE_WEEK": ["3"],
        "SELL_OUT_QUANTITY": ["1035"],
        "STOCK_QUANTITY": ["54.0"],
        "INVOICED_QUANTITY": ["46800.0"],
        "IN_TRANSIT_QUANTITY": ["235.8133199"],
        "SALES_MIN_WINDOW": ["-194634.0"],
        "SALES_MAX_WINDOW": ["2835894.0"],
        "SALES_MEAN_WINDOW": ["2088.72558040324"],
        "SALES_last_WINDOW": ["210.0"],
        "lag1": ["1035"],
        "lag2": ["1035"],
        "lag3": ["1035"],
        "lag4": ["1035"],
        "lag5": ["1035"],
        "lag6": ["1035"],
        "lag7": ["1035"],
        "lag8": ["1035"],
        "lag9": ["1035"],
        "lag10": ["1035"],
        "lag11": ["1035"],
        "lag12": ["1035"] 
    }

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())