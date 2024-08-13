import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the TensorFlow SavedModel
model = tf.saved_model.load("TFLm0_model")
infer = model.signatures["serving_default"]

# Load dataset
df = pd.read_csv('Weather historical 01_01_2023 to 31_12_2023.csv')

# Function to preprocess the input data
def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    df['Hour'] = df.index.hour
    df['Day'] = df.index.day
    df['Weekday'] = df.index.weekday
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Day_Sin'] = np.sin(2 * np.pi * df['Day'] / 31)
    df['Day_Cos'] = np.cos(2 * np.pi * df['Day'] / 31)
    df['Weekday_Sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
    df['Weekday_Cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['Season'] = df['Month'].apply(get_season)

    season_dummies = pd.get_dummies(df['Season'], prefix='Season')
    df = pd.concat([df, season_dummies], axis=1)

    for season in ['Season_Winter', 'Season_Spring', 'Season_Summer', 'Season_Fall']:
        if (season not in df.columns):
            df[season] = 0

    df['temp_lag_1'] = df['Temperature at 2M'].shift(1)
    df['wind_lag_1'] = df['Wind Direction at 10 M'].shift(1)
    df['solar_lag_1'] = df['Solar Irradiance'].shift(1)

    df = df.fillna(0)
    df.drop(columns=['Season'], inplace=True)

    X = df.values.astype('float32')

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    return X_scaled, df.index

# Streamlit UI
st.image('ss.jpg', caption='Durning Centre 39.02 kWh PV System')
st.title("An improved photovoltaic power production forecasting using a hybrid model of Bi-LSTM and Transformer Attention Mechanism")

#st.write("Data Preview:", df.head())

start_date = st.date_input('Start date', value=pd.to_datetime('2023-01-01'))
start_time = st.time_input('Start time', value=pd.to_datetime('2023-01-01 00:00').time(), step=3600)
end_date = st.date_input('End date', value=pd.to_datetime('2023-02-01'))
end_time = st.time_input('End time', value=pd.to_datetime('2023-02-01 23:00').time(), step=3600)

start_datetime = pd.to_datetime(f"{start_date} {start_time}")
end_datetime = pd.to_datetime(f"{end_date} {end_time}")

# Validate date range
data_min_date = pd.to_datetime(df['date'].min())
data_max_date = pd.to_datetime(df['date'].max())

if start_datetime < data_min_date or end_datetime > data_max_date:
    st.error(f"Please select a date range within the dataset's date range: {data_min_date.date()} to {data_max_date.date()}.")
else:
    if st.button('Process Prediction'):
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_datetime) & (df['date'] <= end_datetime)]

        if df.empty:
            st.warning("No data available for the selected date range.")
        else:
            X_scaled, datetime_index = preprocess_data(df)
            inputs = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
            predictions = infer(inputs)['output_0'].numpy()
            
            if predictions.size == 0:
                st.error("Failed to make predictions. Please check the input data and model.")
            else:
                predictions = predictions.reshape(-1, 1)
                adjustment_scale = 22.555  
                predictions_adjusted = predictions * adjustment_scale
                
                historical_yield_data = np.random.rand(len(predictions_adjusted), 1)  
                scaler_y = MinMaxScaler()
                scaler_y.fit(historical_yield_data)
                
                predictions_inverse = scaler_y.inverse_transform(predictions_adjusted)
                
                predictions_df = pd.DataFrame(predictions_inverse, index=datetime_index, columns=['Predicted Total Yield [kWh]'])
                
                st.write("Predicted Total Yield:", predictions_df.head(20))
                
                fig, ax = plt.subplots(figsize=(15, 8))
                ax.plot(predictions_df.index, predictions_df['Predicted Total Yield [kWh]'], label='Predicted')
                ax.set_xlabel('Datetime')
                ax.set_ylabel('Total Yield [kWh]')
                ax.legend()
                
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)

                csv = predictions_df.to_csv().encode('utf-8')
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv", key='download-csv')
