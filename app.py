import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Durning Centre PV Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# Lightweight visual styling
# No external fonts, JavaScript, image libraries or extra packages are used.
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(14, 116, 144, 0.08), transparent 32rem),
                linear-gradient(180deg, #f7faf8 0%, #f4f7f6 100%);
            color: #17312d;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {
            background: rgba(247, 250, 248, 0.88);
        }

        .hero {
            background: linear-gradient(135deg, #123f3a 0%, #17635b 55%, #0e7490 100%);
            border-radius: 20px;
            padding: 1.7rem 1.9rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 34px rgba(19, 67, 61, 0.12);
        }

        .hero-kicker {
            margin: 0 0 0.35rem 0;
            color: #ccece6;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .hero h1 {
            margin: 0;
            color: #ffffff;
            font-size: clamp(1.75rem, 3vw, 2.65rem);
            line-height: 1.12;
            letter-spacing: -0.025em;
        }

        .hero p {
            max-width: 900px;
            margin: 0.7rem 0 0 0;
            color: #e7f4f1;
            font-size: 1rem;
            line-height: 1.6;
        }

        .info-card {
            min-height: 105px;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #dce8e4;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 6px 22px rgba(21, 62, 57, 0.06);
        }

        .info-label {
            color: #67807a;
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }

        .info-value {
            color: #143d38;
            font-size: 1.25rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .info-note {
            color: #6b7f7b;
            font-size: 0.82rem;
            margin-top: 0.28rem;
            line-height: 1.35;
        }

        .section-title {
            margin-top: 0.55rem;
            margin-bottom: 0.2rem;
            color: #173f39;
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -0.01em;
        }

        .section-copy {
            margin-top: 0;
            margin-bottom: 0.9rem;
            color: #667c77;
            font-size: 0.92rem;
        }

        div[data-testid="stDateInput"] > div,
        div[data-testid="stTimeInput"] > div {
            border-radius: 12px;
        }

        div.stButton > button {
            width: 100%;
            border: 0;
            border-radius: 12px;
            padding: 0.72rem 1rem;
            background: linear-gradient(90deg, #17635b, #0e7490);
            color: white;
            font-weight: 750;
            box-shadow: 0 6px 18px rgba(14, 116, 144, 0.18);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 22px rgba(14, 116, 144, 0.24);
            color: white;
        }

        div.stDownloadButton > button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid #17635b;
            color: #17635b;
            background: #ffffff;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dce8e4;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            box-shadow: 0 4px 16px rgba(21, 62, 57, 0.05);
        }

        div[data-testid="stMetricLabel"] {
            color: #627a75;
        }

        div[data-testid="stMetricValue"] {
            color: #173f39;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #dce8e4;
            border-radius: 14px;
            overflow: hidden;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        .app-footer {
            margin-top: 1.6rem;
            padding-top: 1rem;
            border-top: 1px solid #dce8e4;
            color: #71837f;
            font-size: 0.78rem;
            text-align: center;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-top: 0.8rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero {
                padding: 1.25rem 1.2rem;
                border-radius: 16px;
            }

            .info-card {
                min-height: auto;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached resources
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.saved_model.load("TFLm0_model")
    infer_function = model.signatures["serving_default"]
    return model, infer_function


@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv("Weather historical 01_01_2023 to 31_12_2023.csv")


model, infer = load_model()
source_df = load_dataset()


# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
def preprocess_data(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    df["Hour"] = df.index.hour
    df["Day"] = df.index.day
    df["Weekday"] = df.index.weekday
    df["Month"] = df.index.month
    df["Year"] = df.index.year

    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Day_Sin"] = np.sin(2 * np.pi * df["Day"] / 31)
    df["Day_Cos"] = np.cos(2 * np.pi * df["Day"] / 31)
    df["Weekday_Sin"] = np.sin(2 * np.pi * df["Weekday"] / 7)
    df["Weekday_Cos"] = np.cos(2 * np.pi * df["Weekday"] / 7)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    df["Season"] = df["Month"].apply(get_season)

    season_dummies = pd.get_dummies(df["Season"], prefix="Season")
    df = pd.concat([df, season_dummies], axis=1)

    for season in [
        "Season_Winter",
        "Season_Spring",
        "Season_Summer",
        "Season_Fall",
    ]:
        if season not in df.columns:
            df[season] = 0

    df["temp_lag_1"] = df["Temperature at 2M"].shift(1)
    df["wind_lag_1"] = df["Wind Direction at 10 M"].shift(1)
    df["solar_lag_1"] = df["Solar Irradiance"].shift(1)

    df = df.fillna(0)
    df.drop(columns=["Season"], inplace=True)

    X = df.values.astype("float32")

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    X_scaled = X_scaled.reshape(
        (X_scaled.shape[0], 1, X_scaled.shape[1])
    )

    return X_scaled, df.index


# -----------------------------------------------------------------------------
# Existing seasonal adjustment scales
# -----------------------------------------------------------------------------
winter_adjustment_scale = 6
spring_adjustment_scale = 20.0
summer_adjustment_scale = 16.346
fall_adjustment_scale = 8.334


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Edge Hill University · Renewable Energy Forecasting</div>
        <h1>Durning Centre PV Power Forecasting</h1>
        <p>
            Interactive photovoltaic generation forecasting for the Durning Centre
            39.02 kWp PV system, using a hybrid Bi-LSTM and Transformer attention
            framework with historical weather and temporal features.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# System overview
# -----------------------------------------------------------------------------
overview_left, overview_mid_1, overview_mid_2, overview_right = st.columns(
    [1.45, 1, 1, 1]
)

with overview_left:
    st.image(
        "ss.jpg",
        caption="Durning Centre rooftop and façade PV installation",
    )

with overview_mid_1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Installed Capacity</div>
            <div class="info-value">39.02 kWp</div>
            <div class="info-note">Combined rooftop and façade PV system.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview_mid_2:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Forecast Model</div>
            <div class="info-value">Bi-LSTM + Attention</div>
            <div class="info-note">Hybrid deep-learning forecasting framework.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview_right:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Forecast Resolution</div>
            <div class="info-value">Hourly</div>
            <div class="info-note">Interactive date and time selection.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Forecast controls
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="section-title">Forecast Configuration</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-copy">'
    'Choose a historical weather window, then run the deployed forecasting model.'
    "</div>",
    unsafe_allow_html=True,
)

control_1, control_2, control_3, control_4 = st.columns(4)

with control_1:
    start_date = st.date_input(
        "Start date",
        value=pd.to_datetime("2023-01-01"),
    )

with control_2:
    start_time = st.time_input(
        "Start time",
        value=pd.to_datetime("2023-01-01 00:00").time(),
        step=3600,
    )

with control_3:
    end_date = st.date_input(
        "End date",
        value=pd.to_datetime("2023-02-01"),
    )

with control_4:
    end_time = st.time_input(
        "End time",
        value=pd.to_datetime("2023-02-01 23:00").time(),
        step=3600,
    )

start_datetime = pd.to_datetime(f"{start_date} {start_time}")
end_datetime = pd.to_datetime(f"{end_date} {end_time}")

data_min_date = pd.to_datetime(source_df["date"].min())
data_max_date = pd.to_datetime(source_df["date"].max())

st.caption(
    f"Available source-data period: "
    f"{data_min_date.strftime('%d %b %Y')} to "
    f"{data_max_date.strftime('%d %b %Y')}"
)

run_prediction = st.button("⚡ Process Prediction")


# -----------------------------------------------------------------------------
# Prediction workflow
# -----------------------------------------------------------------------------
if start_datetime > end_datetime:
    st.error("The start date and time must be earlier than the end date and time.")

elif start_datetime < data_min_date or end_datetime > data_max_date:
    st.error(
        "Please select a date range within the dataset's date range: "
        f"{data_min_date.date()} to {data_max_date.date()}."
    )

elif run_prediction:
    df = source_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[
        (df["date"] >= start_datetime)
        & (df["date"] <= end_datetime)
    ]

    if df.empty:
        st.warning("No data are available for the selected date range.")

    else:
        with st.spinner("Running PV generation forecast..."):
            X_scaled, datetime_index = preprocess_data(df)

            inputs = tf.convert_to_tensor(
                X_scaled,
                dtype=tf.float32,
            )

            predictions = infer(inputs)["output_0"].numpy()

        if predictions.size == 0:
            st.error(
                "Failed to make predictions. Please check the input data and model."
            )

        else:
            predictions = predictions.reshape(-1, 1)

            # Existing seasonal adjustment behaviour is preserved.
            if start_datetime.month in [12, 1, 2]:
                adjustment_scale = winter_adjustment_scale
            elif start_datetime.month in [3, 4, 5]:
                adjustment_scale = spring_adjustment_scale
            elif start_datetime.month in [6, 7, 8]:
                adjustment_scale = summer_adjustment_scale
            else:
                adjustment_scale = fall_adjustment_scale

            predictions_adjusted = predictions * adjustment_scale

            # Existing inverse-scaling behaviour is preserved so that this
            # redesign changes presentation rather than forecasting logic.
            historical_yield_data = np.random.rand(
                len(predictions_adjusted),
                1,
            )
            scaler_y = MinMaxScaler()
            scaler_y.fit(historical_yield_data)

            predictions_inverse = scaler_y.inverse_transform(
                predictions_adjusted
            )

            predictions_df = pd.DataFrame(
                predictions_inverse,
                index=datetime_index,
                columns=["Predicted Total Yield [kWh]"],
            )

            # -----------------------------------------------------------------
            # Results summary
            # -----------------------------------------------------------------
            st.markdown(
                '<div class="section-title">Forecast Results</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-copy">'
                "A concise operational summary of the selected forecasting window."
                "</div>",
                unsafe_allow_html=True,
            )

            metric_1, metric_2, metric_3, metric_4 = st.columns(4)

            with metric_1:
                st.metric(
                    "Forecast observations",
                    f"{len(predictions_df):,}",
                )

            with metric_2:
                st.metric(
                    "Peak predicted yield",
                    f"{predictions_df['Predicted Total Yield [kWh]'].max():.2f} kWh",
                )

            with metric_3:
                st.metric(
                    "Mean predicted yield",
                    f"{predictions_df['Predicted Total Yield [kWh]'].mean():.2f} kWh",
                )

            with metric_4:
                st.metric(
                    "Forecast start month",
                    start_datetime.strftime("%B"),
                )

            # -----------------------------------------------------------------
            # Forecast visualisation
            # -----------------------------------------------------------------
            st.markdown(
                '<div class="section-title">Predicted PV Generation Profile</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-copy">'
                "Hourly predicted total yield across the selected period."
                "</div>",
                unsafe_allow_html=True,
            )

            fig, ax = plt.subplots(figsize=(14, 5.8))

            line = ax.plot(
                predictions_df.index,
                predictions_df["Predicted Total Yield [kWh]"],
                linewidth=2.2,
                label="Predicted PV yield",
            )[0]

            ax.fill_between(
                predictions_df.index,
                predictions_df["Predicted Total Yield [kWh]"].values,
                0,
                alpha=0.10,
            )

            ax.set_xlabel("Datetime", fontsize=10)
            ax.set_ylabel("Total Yield [kWh]", fontsize=10)
            ax.set_title(
                "Durning Centre PV Generation Forecast",
                loc="left",
                fontsize=14,
                fontweight="bold",
                pad=14,
            )

            ax.grid(
                True,
                axis="y",
                alpha=0.16,
                linewidth=0.8,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_alpha(0.25)
            ax.spines["bottom"].set_alpha(0.25)

            # Keep hourly ticks for shorter selections while avoiding a
            # visually overcrowded chart for long date ranges.
            selected_hours = max(
                1,
                int(
                    (end_datetime - start_datetime).total_seconds()
                    / 3600
                ),
            )

            if selected_hours <= 72:
                locator = mdates.HourLocator(interval=6)
                date_format = "%d %b\n%H:%M"
            elif selected_hours <= 24 * 14:
                locator = mdates.DayLocator(interval=1)
                date_format = "%d %b"
            else:
                locator = mdates.DayLocator(interval=3)
                date_format = "%d %b"

            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter(date_format)
            )

            ax.legend(
                handles=[line],
                frameon=False,
                loc="upper right",
            )

            plt.xticks(rotation=0)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            # -----------------------------------------------------------------
            # Forecast data
            # -----------------------------------------------------------------
            st.markdown(
                '<div class="section-title">Forecast Data</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-copy">'
                "Review the generated values or export the complete forecast as CSV."
                "</div>",
                unsafe_allow_html=True,
            )

            display_df = predictions_df.copy()
            display_df.index.name = "Datetime"

            st.dataframe(
                display_df.head(20).style.format(
                    {"Predicted Total Yield [kWh]": "{:.3f}"}
                ),
                height=330,
            )

            csv = predictions_df.to_csv().encode("utf-8")

            st.download_button(
                "⬇ Download Complete Forecast CSV",
                csv,
                "predictions.csv",
                "text/csv",
                key="download-csv",
            )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        Durning Centre PV Forecasting Platform · Edge Hill University ·
        Research and analytical demonstration
    </div>
    """,
    unsafe_allow_html=True,
)
