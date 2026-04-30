# I want to scrap/collect data from Yahoo Finance
# The data is about Gold/USD
# I want to know the open, high, low, close (OHLC)
# I want this data from Jan 2023 to Dec 2024
# I want this data be saved as an excel file.

import yfinance as yf
import pandas as pd

stock_symbol = "GC=F"

start_date = "2023-01-01"
end_date = "2023-12-31"

stock_data = yf.download(stock_symbol, start_date, end_date)

selected_data = stock_data[["Open", "High", "Low", "Close"]]

selected_data.to_excel("Gold_USD_Prices.xlsx", index = True)

print("Data collected and saved as Gold_USD_Prices.xlsx")