import pandas as pd
import json


if __name__ == "__main__": 

    # Read the CSV file into a DataFrame
    file_path = 'updated_file_3.csv'
    data = pd.read_csv(file_path)

    file_path = 'saved_individuals_11.json'

    with open(file_path, 'r') as json_file:
        jsonData = json.load(json_file)

    result = []  # List to store individual information

    for entry in jsonData:
        capital = 1
        bought_stock = False
        totalProfit = 0
        bought_stock_shares = 0
        periodsHeld = 0
        individual = entry['individual']
        # Find the 80% mark for each symbol
        symbols = data['Symbol'].unique()
        for symbol in symbols:
            symbol_data = data[data['Symbol'] == symbol]  # Filter data for the current symbol
            symbol_dates = symbol_data['Date'].unique()  # Get unique dates for the symbol
            
            # Calculate the number of days that represent 80% of the data
            eighty_percent = int(len(symbol_dates) * 0.8)
            
            # Extract the last 20% of the days for the symbol
            dates_last_20_percent = symbol_dates[eighty_percent:]
            
            # Loop through the data for the last 20% of the days
            for date in dates_last_20_percent:
                subset_data = symbol_data[symbol_data['Date'] == date]  # Filter data for the current date
                for row in subset_data.iterrows():
                    buyTotal = 0
                    sellTotal = 0
                    rowData = row[1]
                    if not bought_stock:
                        if rowData["movingAvgBuy"] != 0:
                            buyTotal += individual[0] * rowData["movingAvgBuy"]
                        if rowData["rsiBuy"] != 0:
                            buyTotal += individual[1] * rowData["rsiBuy"]
                        if rowData["macdBuy"] != 0:
                            buyTotal += individual[2] * rowData["macdBuy"]
                        if rowData["morningStar"] > 0:
                            buyTotal += individual[3] * (rowData["morningStar"] / 100)
                        if rowData["hammer"] > 0:
                            buyTotal += individual[4] * (rowData["hammer"] / 100)
                        if rowData["piercing"] > 0:
                            buyTotal += individual[5] * (rowData["piercing"] / 100)
                        if rowData["invertedHammer"] > 0:
                            buyTotal += individual[6] * (rowData["invertedHammer"] / 100)
                        if rowData["threeWhiteSoldiers"] > 0:
                            buyTotal += individual[7] * (rowData["threeWhiteSoldiers"] / 100)
                        if rowData["engulfing"] > 0:
                            buyTotal += individual[8] * (rowData["engulfing"] / 100)
                        if rowData["hamari"] > 0:
                            buyTotal += individual[9] * (rowData["hamari"] / 100)
                        if rowData["beltHold"] > 0:
                            buyTotal += individual[10] * (rowData["beltHold"] / 100)
                        if rowData["threeInsideUp"] > 0:
                            buyTotal += individual[11] * (rowData["threeInsideUp"] / 100)
                        if rowData["kicker"] > 0:
                            buyTotal += individual[12] * (rowData["kicker"] / 100)
                        if buyTotal > 0.1 and rowData["mustSell"] != 1:
                            bought_stock = True
                            bought_stock_shares = capital/rowData["Close"]
                    else:
                        periodsHeld += 1
                        if rowData["movingAvgSell"] == 1:
                            sellTotal += individual[13]
                        if rowData["rsiSell"] == 1:
                            sellTotal += individual[14]
                        if rowData["macdSell"] == 1:
                            sellTotal += individual[15]
                        if rowData["engulfing"] < 0:
                            buyTotal += individual[16] * (rowData["engulfing"] / -100)
                        if rowData["hamari"] < 0:
                            buyTotal += individual[17] * (rowData["hamari"] / -100)
                        if rowData["beltHold"] < 0:
                            buyTotal += individual[18] * (rowData["beltHold"] / -100)
                        if rowData["threeInsideUp"] < 0:
                            buyTotal += individual[19] * (rowData["threeInsideUp"] / -100)
                        if rowData["kicker"] < 0:
                            buyTotal += individual[20] * (rowData["kicker"] / -100)
                        if rowData["shootingStar"] < 0:
                            buyTotal += individual[21] * (rowData["shootingStar"] / -100)
                        if rowData["darkCloudCover"] < 0:
                            buyTotal += individual[22] * (rowData["darkCloudCover"] / -100)
                        if rowData["eveningStarList"] < 0:
                            buyTotal += individual[23] * (rowData["eveningStarList"] / -100)
                        if rowData["hangingMan"] < 0:
                            buyTotal += individual[24] * (rowData["hangingMan"] / -100)
                        if rowData["threeBlackCrows"] < 0:
                            buyTotal += individual[25] * (rowData["threeBlackCrows"] / -100)
                        if sellTotal > 0.1 or periodsHeld >= 20 or rowData["mustSell"] == 1:
                            bought_stock = False
                            profit = (rowData["Close"] * bought_stock_shares) - capital
                            bought_stock_shares = 0
                            totalProfit += profit
                            periodsHeld = 0
        individual_dict = {
            "Individual": individual,
            "Profit": totalProfit
        }
        result.append(individual_dict)

    json_output = json.dumps(result, indent=4)
    output_file_json = 'test_individuals_11.json'
    with open(output_file_json, 'w') as json_file:
        json.dump(result, json_file, indent=4)