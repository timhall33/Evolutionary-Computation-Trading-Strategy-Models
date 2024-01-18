import pandas as pd
import json


if __name__ == "__main__": 

    # Read the CSV file into a DataFrame
    file_path = 'updated_file_2.csv'
    data = pd.read_csv(file_path)

    file_path = 'saved_individuals.json'

    with open(file_path, 'r') as json_file:
        jsonData = json.load(json_file)

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
                        if buyTotal > 0.5 and rowData["mustSell"] != 1:
                            bought_stock = True
                            bought_stock_shares = capital/rowData["Close"]
                    else:
                        periodsHeld += 1
                        if rowData["movingAvgSell"] == 1:
                            sellTotal += individual[3]
                        if rowData["rsiSell"] == 1:
                            sellTotal += individual[4]
                        if rowData["macdSell"] == 1:
                            sellTotal += individual[5]
                        if sellTotal > 0.5 or periodsHeld >= 20 or rowData["mustSell"] == 1:
                            bought_stock = False
                            profit = (rowData["Close"] * bought_stock_shares) - capital
                            bought_stock_shares = 0
                            totalProfit += profit
                            periodsHeld = 0
        print(f"Individual: {individual}, Profit: {totalProfit}")