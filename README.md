# CSC4343-Stock
Applied deep learning project with the stock market

Stock data comes from finnhub.io. Free API key was required to download. The data is the 1 minute OHLC(volume) data on the free tier. The data range from 1992 throught february 2026. (march data has not yet been released as of 4.12.26)
Since this data is going to be used for multiple stock market projects, it lives in a folder at equal level to the root of the project. This will allow the same data to function for multiple projects easily. 
Container Folder
|_ Project Folder
|_ OHLC 1 minute data
  |_ extracted files
    |_ 1992
    |_ 1993
    ...
    |_ 2026
      |_ 2026-01
      |_ 2026-02
        |_ TICKER.csv

The actual file name for TICKER.csv is formatted as the ticker in capital leters. For example Apple's ticker is AAPL, so the file for Apple's monthly level CSV is AAPL.csv. 

Economic data was pulled from the Federal Reserve Economic Database (FRED). Currently, we include the two-year U.S. Treasury yield and ten-year U.S. Treasury yield (both quoted on a market basis), the Economic Policy Uncertainty Index created by Stanford economist Dr. Nick Bloom, and five-year inflation expectations dervied from the yields of U.S. Treasury Inflation-Protected Securities (TIPS).
