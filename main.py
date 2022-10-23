from datetime import datetime

from general import join_csv_files
from env_config import DATA_QUERY, MODEL_TRAINING, MODEL_PREDICTION
from predictor import train_model, predict_stonk
from stonks import get_processed_syp_data, get_stonk, process_stock_record, grab_syp_tickers, get_processed_stock_data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # for symbol in grab_syp_tickers():
    #     process_stock_record(get_stonk(symbol), datetime(2022, 4, 16))
    # exit(0)
    if DATA_QUERY:
        get_processed_syp_data(datetime(2022, 4, 10))
        join_csv_files("S&P Processed Data", "FULL SYP PROCESSED DATA")
    if MODEL_TRAINING:
        train_model("FULL SYP PROCESSED DATA.csv")
    if MODEL_PREDICTION:
        predict_stonk(get_stonk("FB"), datetime(2022, 3, 15))
