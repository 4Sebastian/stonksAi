import math
import multiprocessing as mp
import os
import traceback
from datetime import datetime
from typing import List

import pandas as pd
import yfinance as yf
import talib as ta
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from pandas import DataFrame
from yfinance import Ticker
from yahoo_fin import stock_info as si

from general import get_next_weekday
from env_config import DELTA, PREDICTION_DELTA, SHOULD_PLOT_STOCK, PERCENTAGE_DELTA, NUM_RETRIEVED, CHUNK_SIZE, \
    NUM_ELEMENTS, SAVE_STOCK_DATA, STOCK_DATA_LOCATION


def get_stonk(ticker_symbol: str) -> Ticker:
    return yf.Ticker(ticker_symbol)


def grab_syp_tickers() -> set:
    df1 = pd.DataFrame(si.tickers_sp500())
    df1.to_csv('S&P ticker symbols.csv')
    symbols = set(symbol for symbol in df1[0].values.tolist())
    my_list = ['W', 'R', 'P', 'Q']
    del_set = set()
    sav_set = set()

    for symbol in symbols:
        if len(symbol) > 4 and symbol[-1] in my_list:
            del_set.add(symbol)
        else:
            sav_set.add(symbol)
    print(f'Removed {len(del_set)} unqualified stock symbols...')
    print(f'There are {len(sav_set)} qualified stock symbols...')
    return sav_set


def get_percentage_move(df: DataFrame, columns: List) -> DataFrame:
    temp_df = pd.DataFrame(columns=columns)
    for column in columns:
        changed_values = []
        for idx, value in enumerate(df[column]):
            if idx != 0:
                changed_values.append(value / df[column][idx - 1] * 100 - 50.0)
            else:
                changed_values.append(50.0)
        temp_df[column] = changed_values

    return temp_df


def get_stock_day_info(stonk: Ticker, date: datetime) -> DataFrame:
    buffer_date = get_next_weekday(
        date + relativedelta(days=8 - date.weekday() if date.weekday() == 5 or date.weekday() == 6 else 1))
    try:
        stonk_df = stonk.history(start=date, end=buffer_date)
        while len(stonk_df.index) < 1:
            buffer_date += relativedelta(days=1)
            stonk_df = stonk.history(start=date, end=buffer_date)
        return stonk_df
    except Exception as e:
        print("Failed to retrieve stock day info:", stonk.ticker, ":", date, ":", buffer_date, ":", e)
        traceback.print_exc()
        raise Exception(e)


def save_stonk_df(stonk: Ticker, df: DataFrame, start_date: datetime, end_date: datetime) -> None:
    format_name = [str(stonk.ticker), str(start_date.month), str(start_date.day), str(start_date.year), str(end_date.month), str(end_date.day),
                   str(end_date.year)]
    format_path = [STOCK_DATA_LOCATION, str(stonk.ticker)]

    if not os.path.isdir('/'.join(format_path)):
        os.mkdir('/'.join(format_path))

    format_path.append('_'.join(format_name)+'.csv')

    df.to_csv('/'.join(format_path))


def process_stock_record(stonk: Ticker, end_date: datetime = datetime(2022, 4, 5)) -> list:
    start_date = get_next_weekday(end_date - relativedelta(days=DELTA + 15))
    try:
        # Attempt Retrieval of stock history
        df = stonk.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    except Exception as e:
        print("Failed to retrieve stock data:", stonk.ticker, ":", e)
        traceback.print_exc()
        raise Exception(e)
    try:
        if SAVE_STOCK_DATA:
            save_stonk_df(stonk, df, start_date, end_date)
    except Exception as e:
        print("Failed to save stock data:", stonk.ticker, ":", e)
        traceback.print_exc()
    try:
        # Attempt Retrieval of stock history at End Date
        end_info = get_stock_day_info(stonk, end_date)
    except Exception as e:
        print("Failed to retrieve stock data at end date:", stonk.ticker, ":", e)
        traceback.print_exc()
        raise Exception(e)

    # Prediction Info for whether it was a legitimate buy/put
    prediction_date = end_date + relativedelta(days=PREDICTION_DELTA)
    try:
        # Attempt Retrieval of stock history at End Date
        prediction_info = get_stock_day_info(stonk, prediction_date)
    except Exception as e:
        print("Failed to retrieve stock data at prediction date:", stonk.ticker, ":", e)
        traceback.print_exc()
        raise Exception(e)

    try:
        df['RSI'] = ta.RSI(df['Close'], 14)
        df['up_band'], df['mid_band'], df['low_band'] = ta.BBANDS(df['Close'], timeperiod=14)
    except Exception as e:
        print("Failed to calculate stock analytics:", stonk.ticker, ":", e)
        traceback.print_exc()
        raise Exception(e)

    try:
        # Trimming DataFrame of NaN values and columns that are not in use
        df.drop(df.index[0:15], inplace=True)
        df.drop(["Open", "High", "Low", "Dividends", "Stock Splits"], axis=1, inplace=True)

        # Get Percentage move rather than actual value
        changed_columns = ["Volume", "Close", "up_band", "mid_band", "low_band"]
        df_new_values = get_percentage_move(df, changed_columns)
        df.drop(changed_columns, axis=1, inplace=True)
        df[changed_columns] = df_new_values[changed_columns].values
        df.drop(df.index[0:1], inplace=True)

        # Guaranteeing set data size, assisting with data size variation
        step = len(df.index) / float(NUM_ELEMENTS)
        idx_list = []
        [idx_list.append(x) for x in [round(i * step) for i in range(NUM_ELEMENTS)] if x not in idx_list]
        drop_list = [*(int(i) for i in [*range(0, len(df.index))] if i not in idx_list)]
        # print(drop_list)
        df.drop(df.index[drop_list], axis=0, inplace=True)

        df_list = [x for xs in df.values.tolist() for x in xs]

        if SHOULD_PLOT_STOCK:
            plt.style.use('fivethirtyeight')
            df[['Close', 'Volume', 'up_band', 'mid_band', 'low_band', 'RSI']].plot(figsize=(8, 8))
            plt.show()

        # append whether it should have been a buy, sell, nothing in delta days with change percentage threshold
        end_close = end_info['Close'].values[0]
        prediction_close = prediction_info['Close'].values[0]
        if abs(prediction_close / end_close - 1.0) * 100 < PERCENTAGE_DELTA:
            results = 0  # AKA do nothing
        else:
            if (prediction_close / end_close - 1) * 100 > PERCENTAGE_DELTA:
                results = 1  # AKA buy it and sell it in delta days
            else:
                results = 2  # AKA sell it and buy back in delta days
        df_list.append(results)
        # print(stonk.ticker, len(df_list), len(df.index))

        return df_list
    except Exception as e:
        print("General Error Failed:", stonk.ticker, ":", e)
        traceback.print_exc()
        raise Exception(e)


def get_processed_stock_data(stonk: Ticker, end_date: datetime) -> DataFrame:
    df = None
    for val in range(NUM_RETRIEVED, 0, -1):
        if (response := (end_date - relativedelta(
                days=val))) is not None and response.weekday() != 5 and response.weekday() != 6:
            try:
                processed_stock_list = process_stock_record(stonk, response)
            except Exception as e:
                print("Error retrieving stock data:", stonk.ticker, ":", e)
                traceback.print_exc()
                continue
            else:
                if df is None:
                    df = pd.DataFrame(processed_stock_list).T
                else:
                    df = pd.concat([df, pd.DataFrame(processed_stock_list).T])
        if val != 800 and val % 25 == 0:
            print(stonk.ticker, "Progress:", int((800-val)/NUM_RETRIEVED * 100))

    return df


def core_process(symbols: List[Ticker], cnt: int, end_date: datetime) -> None:
    df = None
    for symbol in symbols:
        print("Started Ticker Symbol:", symbol)
        try:
            stonk = get_stonk(str(symbol))
        except Exception as e:
            print("Error retreiving stock object:", str(symbol), ":", e)
            traceback.print_exc()
        else:
            try:
                processed_data = get_processed_stock_data(stonk, end_date)
            except Exception as e:
                print("Error with processing with stock:", symbol.ticker, ":", e)
                traceback.print_exc()
                continue
            else:
                if df is None:
                    df = processed_data
                else:
                    df = pd.concat([df, processed_data])
            print("Completed", symbol)

    filename = "syp_processed_data_" + str(NUM_RETRIEVED) + "_" + str(cnt * CHUNK_SIZE) + "_" + str(
        cnt * CHUNK_SIZE + len(symbols)) + ".csv"
    df.to_csv("S&P Processed Data/" + filename)
    print("Completed:", filename)


def get_processed_syp_data(end_date: datetime) -> None:
    tickers = list(grab_syp_tickers())

    tickers_lists = [tickers[x:x + CHUNK_SIZE] for x in range(0, len(tickers), CHUNK_SIZE)]

    cnt = 0
    core_cnt = 0
    processes = []
    is_finished = False
    while not is_finished:
        while cnt < len(tickers_lists):
            if core_cnt >= mp.cpu_count():
                core_cnt = 0
                break
            print("Creating Process:", cnt)
            symbols = tickers_lists[cnt]
            processes.append(mp.Process(target=core_process, args=(symbols, cnt, end_date)))
            cnt += 1
            core_cnt += 1
        if cnt >= len(tickers_lists):
            is_finished = True
        print("Running processes:", "Unique Iterated Process ->", cnt)
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            process.close()
        processes.clear()
        print("Total Progress:", int(cnt/len(tickers_lists) * 100))
