import copy
import os
from datetime import datetime
from typing import Any, Tuple

import numpy
import pandas as pd
import tensorflow as tf
import numpy as np
from pandas import DataFrame
from yfinance import Ticker

from env_config import BATCH_SIZE, TRAINING_PATH, EPOCHS, RESULTS
from stonks import process_stock_record


def standardize_data(data: DataFrame) -> DataFrame:
    return data / 100.0


def get_prepped_data(input_filepath: str) -> tuple[int, DataFrame, DataFrame, DataFrame, DataFrame]:
    input_df = pd.read_csv(input_filepath)
    input_df.dropna(inplace=True)
    np.random.shuffle(input_df.values)
    input_df = input_df.iloc[:, 1:]
    # print(input_df)
    target = input_df.pop("84").to_frame()
    standardized_df = standardize_data(input_df)

    # Number Row to split dataframes at, separating training and testing data
    cnt = int(len(standardized_df.index) * 0.75)
    input_size = len(standardized_df.columns)

    train_data = standardized_df.iloc[:cnt, :]
    test_data = standardized_df.iloc[cnt:, :]
    train_target = target.iloc[:cnt, :]
    test_target = target.iloc[cnt:, :]
    return input_size, train_data, train_target, test_data, test_target


def get_prepped_target(input_target: DataFrame) -> Any:
    template_record = {
        "nothing": [0.0],
        "buy": [0.0],
        "put": [0.0]
    }
    df = DataFrame(columns=template_record)
    for idx, record in input_target.iterrows():
        temp_record = copy.deepcopy(template_record)
        temp_record[list(template_record.keys())[int(record.iloc[0])]] = [1.0]
        df = pd.concat([df, DataFrame(temp_record)])
    return tf.convert_to_tensor(numpy.asarray(df.to_numpy()).astype(np.float32))


def get_stock_data(stonk: Ticker, end_date: datetime) -> DataFrame:
    stonk_df = pd.DataFrame(process_stock_record(stonk, end_date=end_date)).T
    input_df = stonk_df.iloc[:, :len(stonk_df.columns)-1]
    return standardize_data(input_df)


def train_model(input_filepath: str) -> None:
    input_size, train_data, train_target, test_data, test_target = get_prepped_data(input_filepath)
    train_target = get_prepped_target(train_target)
    test_target = get_prepped_target(test_target)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=TRAINING_PATH,
                                                     save_weights_only=True,
                                                     verbose=1)

    model = get_basic_model(input_size)

    model.fit(train_data, train_target, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[cp_callback])
    model.save(TRAINING_PATH)
    model.save(TRAINING_PATH + '.h5')

    test_loss, test_acc = model.evaluate(test_data, test_target, verbose=2)
    print('\nTest accuracy:', test_acc)


def get_basic_model(input_size: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Normalization(axis=-1),
        tf.keras.layers.Dense(input_size, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(3)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def get_loaded_model():
    return tf.keras.Sequential([tf.keras.models.load_model(TRAINING_PATH), tf.keras.layers.Softmax()])


def predict_stonk(stonk: Ticker, end_date: datetime) -> Tuple[int, str]:
    prediction_model = get_loaded_model()
    stonk_data = get_stock_data(stonk, end_date)

    result_idx = numpy.argmax(prediction_model.predict(stonk_data))
    print(int(result_idx), RESULTS[result_idx])
    return int(result_idx), RESULTS[result_idx]



