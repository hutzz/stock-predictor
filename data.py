import yfinance as yf
import pandas as pd

msft = yf.Ticker("MSFT")
msft_hist = msft.history(period="max", interval="1d")
msft_hist.to_csv("data/msft.csv")

goog = yf.Ticker("GOOG")
goog_hist = goog.history(period="max", interval="1d")
goog_hist.to_csv("data/goog.csv")