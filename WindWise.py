"""
File Name: WindWise
Author: Maheen Raza
Creation Date: 2024/10/18
Final Date: 2024/11/12
"""


# Importing libraries that are needed for the ML and web applications aspects
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import os
import streamlit.components.v1 as components

#Usability:

# In order for the usability of the website to be up to par, I created a dark mode toggle for user
# that have preferences to view the website in dark mode over light mode.

# If a user has not selected/clicked the dark mode toggle, the website remains as it is
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# this function is used to enable and disable the dark mode toggle
def toggle_dark_mode():
    st.session_state["dark_mode"] = not st.session_state["dark_mode"]

# labelling the button depending on if the already existing website is in dark mode or not
mode_label = "Switch to Dark Mode" if not st.session_state["dark_mode"] else "Switch to Light Mode"
st.sidebar.button(mode_label, on_click=toggle_dark_mode)

# this portion focuses on the color of the sidebar, text, buttons, etc. certain items will go darker if the dark mode toggle is enabled
st.markdown(f"""
    <style>
    .appview-container .stApp {{
        background-color: {'#333333' if st.session_state["dark_mode"] else '#DDB8A6'}; 
        color: {'#ffffff' if st.session_state["dark_mode"] else '#333333'};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {'#2e2e2e' if st.session_state["dark_mode"] else '#D49B7E'};
    }}

    .stButton>button {{
        background-color: #4b79a1;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #283e51;
    }}

    .card {{
        background-color: {'#444444' if st.session_state["dark_mode"] else '#f8f9fa'};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        color: {'#ffffff' if st.session_state["dark_mode"] else '#000000'};
    }}

    h1 {{
        color: {'#ffffff' if st.session_state["dark_mode"] else '#333333'};
    }}
    h2 {{
        color: {'#ffffff' if st.session_state["dark_mode"] else '#333333'};
    }}
    h3 {{
        color: {'#ffffff' if st.session_state["dark_mode"] else '#333333'};
    }}
    </style>
""", unsafe_allow_html=True)

# setting up the web applications with the title and wind turbine animation
col1, col2 = st.columns([3, 1])

with col1:
    st.title("Welcome to: WindWise")

with col2:
    # design the wind turbine color based on the dark mode state 
    turbine_color = 'white' if st.session_state["dark_mode"] else 'black'
    components.html(f"""
        <div style="width: 200px; height: 250px;">
            <canvas id="turbineCanvas" width="200" height="250"></canvas>
            <script>
                const canvas = document.getElementById('turbineCanvas');
                const ctx = canvas.getContext('2d');
                let angle = 0;

                function drawTurbine() {{
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = '{turbine_color}'; 
                    ctx.fillRect(95, 100, 10, 120);
                    ctx.save();
                    ctx.translate(100, 100);
                    ctx.rotate(angle);
                    for (let i = 0; i < 3; i++) {{
                        ctx.fillStyle = '{turbine_color}'; 
                        ctx.fillRect(-5, -70, 10, 70);
                        ctx.rotate((2 * Math.PI) / 3);
                    }}
                    ctx.restore();
                    angle += 0.05;
                    requestAnimationFrame(drawTurbine);
                }}
                drawTurbine();
            </script>
        </div>""", height=250)

# sidebar made to select the two different pages
page = st.sidebar.selectbox("Choose a page", ["About the Project","WindWise", "About Me"])

# specifying the directory to save files
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    
# quick about page for the project
if page == "About the Project":
    st.header("About the Project: WindWise")
    st.write("With a bit of free time on my hands and escape from heavy academic pressure, I decided to build this basic app for the InnovateHacks2.0 online hackathon as my first try at doing a hackathon. I want to be able to build my skills over the next and participate in more hackathons in the future.")
    st.write("This application takes in an CSV file with the following format:")
    st.image("images/CSVformat.PNG", caption="CSV Format", use_column_width=True)
    st.write("The data I used comes from the NREL (National Renewable Energy Labratory) website. Feel free to check them out here:")
    st.markdown("""
        - [![NREL](https://img.icons8.com/ios-filled/20/26e07f/wind-turbine.png) NREL](https://www.nrel.gov/wind/data-tools.html)
    """, unsafe_allow_html=True)
    st.write("The web application will then give you the choice with the following tabs: Data Overview, EDA and Modeling & Predictions.")
    st.write("Please make sure none of your columns of data have NULL or NaN values.")

# if someone is on the wind data analysis page, give them the option of uploading a CSV file
elif page == "WindWise":
    st.header("WindWise: Wind Data Analysis")
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' saved successfully!")

        df = pd.read_csv(file_path, skiprows=1)
        st.write("### Raw Data")
        st.write(df.head())

        # after file is uploaded, rename the file columns
        df.columns = [
            "Year", "Month", "Day", "Hour", "Minute",
            "WindSpeed_100m", "AirTemperature_100m",
            "WindSpeed_120m", "WindDirection_100m",
            "AirPressure_100m"
        ]
        
        # column created for timestamp (combines the year, month, day, hour and minute columns into one column)
        df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])

        # for usability and to organize the data presented
        tab1, tab2, tab3 = st.tabs(["Data Overview", "EDA", "Modeling & Prediction"])

        # first tab is for basic data information, like mean, median, etc. that is given by the df.info() and df.describe() functions
        with tab1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("### Basic Data Information")
            st.write(df.info())
            st.write(df.describe())
            st.markdown("</div>", unsafe_allow_html=True)

            # this portion of the code is used to handle missing values and feature engineering
            
            # this uses the "forward fill method" in order to fill in missing values
            df.interpolate(method='ffill', inplace=True)
            # this creates a new column with the rolling mean of wind speed (calculates the average of 5 data points)
            # helps us determine long term trends
            df['WindSpeed_100m_rolling_mean'] = df['WindSpeed_100m'].rolling(window=5).mean()
            # extracts hour from the timestamp column
            df['hour'] = df['timestamp'].dt.hour
            # extracts the day of the week from the timestamp column
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            # extracts the month from the timestamp column
            df['month'] = df['timestamp'].dt.month
            # calculates the change in wind direction between consecutive rows by calculating the difference between each value in the 'WindDirection_100m' column and the value before it
            df['wind_direction_change'] = df['WindDirection_100m'].diff()

        # second tab is for exploratory data analysis
        with tab2:
            # creating the EDA part of our web applications
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("### Exploratory Data Analysis")

            # if someone selects to show the plot for wind speed over time, they are shown it
            if st.checkbox("Plot Wind Speed Over Time"):
                plt.figure(figsize=(10, 6))
                # plots windspeed at 100 meters
                plt.plot(df['timestamp'], df['WindSpeed_100m'], label='Wind Speed at 100m')
                # time will be the x axis, while the normalized windspeed will be the y axis
                plt.xlabel('Time')
                plt.ylabel('Wind Speed (normalized)')
                plt.title('Wind Speed Over Time')
                plt.legend()
                plt.grid()
                st.pyplot(plt)

            # if someone selects to show the plot for the rolling mean over time, they are shown it
            if st.checkbox("Show Rolling Mean Plot"):
                plt.figure(figsize=(12, 6))
                # plots the rolling mean average of the windspeed at 100 m vs time
                plt.plot(df['timestamp'], df['WindSpeed_100m_rolling_mean'], label='5-Timepoint Rolling Mean')
                plt.xlabel('Time')
                plt.ylabel('Wind Speed (normalized)')
                plt.title('5-Timepoint Rolling Mean of Wind Speed Over Time')
                plt.legend()
                plt.grid()
                st.pyplot(plt)

            # if someone selects to show the scatter plot for wind speed vs air temp, they are shown it
            if st.checkbox("Scatter Plot of Wind Speed vs Air Temperature"):
                plt.figure(figsize=(8, 6))
                # the x axis is the windspeed, while the y axis is the air temperature
                sns.scatterplot(data=df, x='WindSpeed_100m', y='AirTemperature_100m')
                plt.xlabel('Wind Speed (normalized)')
                plt.ylabel('Air Temperature (normalized)')
                plt.title('Relationship between Wind Speed and Air Temperature')
                st.pyplot(plt)
            st.markdown("</div>", unsafe_allow_html=True)

        # third and final tab looks at a more in-depth analysis
        with tab3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("### Modeling and Prediction")

            # first step involves scaling: standard scaler is used to normalize the given values, which helps improve accuracy
            scaler = StandardScaler()
            df[['WindSpeed_100m', 'AirTemperature_100m']] = scaler.fit_transform(df[['WindSpeed_100m', 'AirTemperature_100m']])

            # perform Linear Regression with NaN check
            if st.checkbox("Perform Linear Regression"):
                # define the features and target variable for regression
                features = ['AirTemperature_100m', 'WindSpeed_120m', 'WindDirection_100m', 
                            'AirPressure_100m', 'hour', 'day_of_week', 'month']
                target = 'WindSpeed_100m'

                # check and fill NaN values in the selected columns, otherwise linear regression won't work
                if df[features + [target]].isnull().any().any():
                    st.warning("NaN values detected in dataset. Applying forward fill to handle missing data.")
                    df[features + [target]] = df[features + [target]].fillna(method='ffill')

                # split the data into training and testing sets (80% for training, 20% for testing)
                X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

                # initialize and train the linear regression model
                lin_reg = LinearRegression()
                lin_reg.fit(X_train, y_train)

                # make predictions and calculate performance metrics
                y_pred_lin = lin_reg.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred_lin)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_lin))
                r2 = r2_score(y_test, y_pred_lin)

                # display the results of the calculating
                st.write("### Linear Regression Results")
                st.write(f"MAE: {mae:.2f}")
                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"R²: {r2:.2f}")

                # plotting the predictions vs the actual wind speed
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred_lin)
                plt.xlabel('Actual Wind Speed')
                plt.ylabel('Predicted Wind Speed')
                plt.title('Predicted vs Actual Wind Speed')
                plt.grid()
                st.pyplot(plt)



            if st.checkbox("Perform ARIMA"):
                # ARIMA aka Autoregressive Integrated Moving Average is used to predict future wind speeds
                # 5 indicates the order of the autoregressive part, 1 is the degree of differencing, and 0 is the order of the moving average
                model = ARIMA(df['WindSpeed_100m'], order=(5, 1, 0))
                model_fit = model.fit()
                st.write(model_fit.summary())

                # forecasts the next 10 values for wind speed and displays them in the app
                forecast = model_fit.forecast(steps=10)
                st.write("### ARIMA Forecast")
                st.write(f"Forecasted Values: {forecast}")

                # plot 1: historical data
                plt.figure(figsize=(10, 6))
                plt.plot(df['WindSpeed_100m'], label='Historical Data')
                plt.xlabel('Time')
                plt.ylabel('Wind Speed (normalized)')
                plt.title('Historical Wind Speed Data')
                plt.legend()
                plt.grid()
                st.pyplot(plt)

                # plot 2: forecasted data
                plt.figure(figsize=(10, 6))
                plt.plot(range(len(df), len(df) + len(forecast)), forecast, label='Forecast', color='red')
                plt.xlabel('Time')
                plt.ylabel('Wind Speed (normalized)')
                plt.title('ARIMA Forecast')
                plt.legend()
                plt.grid()
                st.pyplot(plt)


            st.markdown("</div>", unsafe_allow_html=True)
    # wait until someone uploads a CSV file
    else:
        st.info("Please upload a CSV file to start.")
# other page is just about me :)
elif page == "About Me":
    st.header("About Me")
    st.write("Hello! I'm Maheen Raza, a software engineering student at the University of Calgary, currently an intern at Enbridge, working in the remote operations center.")
    
    # displaying my image
    st.image("images/me.jfif", caption="Maheen Raza", use_column_width=True)
     # Links to GitHub and LinkedIn
    st.markdown("""
        **Connect with me:**
        - [![GitHub](https://img.icons8.com/ios-filled/20/000000/github.png) GitHub](https://github.com/maheenraza03)
        - [![LinkedIn](https://img.icons8.com/ios-filled/20/0077b5/linkedin.png) LinkedIn](https://www.linkedin.com/in/maheen-raza-40b780229/)
    """, unsafe_allow_html=True)

