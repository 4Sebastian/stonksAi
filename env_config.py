# ----------------------------------------------------------------------------------------------------------------------
#
# File Path Specifications
STOCK_DATA_LOCATION = "Stock Data"
#
# Information Specific Towards Data Generated and Formatted
RESULTS = {
    0: 'nothing',
    1: 'buy',
    2: 'put'
}
NUM_ELEMENTS = 14  # Number of elements to retrieve from stock data (slight variances in stock data 0-3 extra entries)
DELTA = 30  # Time interval size in days to look at when retrieving data
PREDICTION_DELTA = 4  # Time between End Date and Prediction Date AKA how far the data should verify the future's result
PERCENTAGE_DELTA = 5  # Required Profit margin to accept whether a stock should be a buy/put or nothing
#
# Information Specific Towards Data Retrieval and Sizing
NUM_RETRIEVED = 800  # Time interval size in days to look at when retrieving ranges of data AKA # of ranges to collect
CHUNK_SIZE = 1  # Size of stock symbol lists going into each core process
#
# Information Specific Towards Machine Learning and TensorFlow
EPOCHS = 15
BATCH_SIZE = 5
TRAINING_PATH = "Trainings/model"
#
# Information Specific Towards Script Manipulation and Analysis
SHOULD_PLOT_STOCK = False  # Whether pulled stocks should be plotted and analyzed
DATA_QUERY = True  # Whether the script should query new stock data from S&P to train ML model
MODEL_TRAINING = False  # Whether the script should attempt training on the model with csv files from processed data
MODEL_PREDICTION = False  # Whether the script should attempt to predict the result of a stock with a loaded model
WIPE_PROCESSED = False  # Whether the script should wipe csv data after its all been compiled into one large csv
SAVE_STOCK_DATA = True  # Whether stock data should be saved when it is queried (Raw stock data)
#
#
# ----------------------------------------------------------------------------------------------------------------------
