"""
Enhanced Stock Prediction with Multiple Models
"""
from email import encoders
from email.mime.base import MIMEBase
import matplotlib
matplotlib.use('Agg')  # 使用 Agg 后端，避免 tkinter
import pandas_market_calendars as mcal
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import webbrowser
from fpdf import FPDF
import textwrap
import matplotlib
from matplotlib import font_manager
import warnings
import pandas_datareader.data as web


warnings.filterwarnings("ignore", message="cmap value too big/small:*")
warnings.filterwarnings("ignore", category=UserWarning)


# get stock data
def get_stock_data(ticker, period="10y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# get gpr
def get_gpr_data():
    url_gpr = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    try:
        gpr_df = pd.read_excel(url_gpr, engine='xlrd')
        gpr_df["Date"] = pd.to_datetime(gpr_df["month"].astype(str).str.replace("M", "-") + "-01")
        return gpr_df
    except Exception as e:
        print("Error from getting GPR data:", e)
        return None


#
def get_fred_data():
    try:

        start = datetime(2000, 1, 1)
        end = datetime.now()

        # get unemployment rate
        unemployment = web.DataReader('UNRATE', 'fred', start, end)
        # get inflation rate
        inflation = web.DataReader('CPIAUCSL', 'fred', start, end)
        inflation = inflation.pct_change(12) * 100  # year inflation
        # get FED rate
        fed_rate = web.DataReader('FEDFUNDS', 'fred', start, end)

        # merge data
        economic_data = pd.concat([unemployment, inflation, fed_rate], axis=1)
        economic_data.columns = ['Unemployment', 'Inflation', 'FedRate']
        return economic_data
    except Exception as e:
        print(" error occurred when getting FED data：", e)
        return None


def merge_stock_gpr(stock_df, gpr_df):
    stock_df = stock_df.copy()
    gpr_df = gpr_df.copy()

    # make sure the index is datetime and no zone
    stock_df.index = pd.to_datetime(stock_df.index)
    stock_df.index = stock_df.index.tz_localize(None)

    # keep the date index as a column and extract the year and month
    stock_df['Date'] = stock_df.index
    stock_df['Year'] = stock_df['Date'].dt.year
    stock_df['Month'] = stock_df['Date'].dt.month

    # deal with gpr_df：turn 'month' column into datetime and clear the zone
    gpr_df['month'] = pd.to_datetime(gpr_df['month'])
    gpr_df['month'] = gpr_df['month'].dt.tz_localize(None)
    gpr_df['Year'] = gpr_df['month'].dt.year
    gpr_df['Month'] = gpr_df['month'].dt.month

    #merge them together
    merged_df = pd.merge(
        stock_df,
        gpr_df[['Year', 'Month', 'GPR']],
        on=['Year', 'Month'],
        how='left'
    )
    # add the time index
    merged_df.set_index('Date', inplace=True)
    # delete the temporary column
    merged_df.drop(columns=['Year', 'Month'], inplace=True)

    return merged_df

#MERGE the stock data with economic data
def merge_economic_data(stock_df, economic_df):

    if economic_df is None:
        return stock_df

    stock_df = stock_df.copy()
    economic_df = economic_df.copy()

    # make sure that the index is datetime
    stock_df.index = pd.to_datetime(stock_df.index)
    economic_df.index = pd.to_datetime(economic_df.index)

    #fill the economic data as day
    economic_df = economic_df.resample('D').ffill()

    # merged the data with index
    merged_df = pd.merge(stock_df, economic_df, left_index=True, right_index=True, how='left')
    return merged_df


def is_trading_day(date, ticker):
        ticker = ticker.strip().lstrip('$')
        if ticker.endswith(".HK"):
            calendar = mcal.get_calendar("HKEX")
        elif ticker.endswith(".SS") or ticker.endswith(".SZ"):
            calendar = mcal.get_calendar("SSE")
        else:
            calendar = mcal.get_calendar("NYSE")

        date = pd.to_datetime(date).replace(hour=0, minute=0, second=0, microsecond=0)
        date_str = date.strftime('%Y-%m-%d')
        schedule = calendar.valid_days(start_date=date_str, end_date=date_str)


        is_trading = not schedule.empty


        return is_trading

    # calculate RSI
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
    #calculate MACD
def compute_MACD(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line



   #calculate bollinger bands
def compute_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band


def add_features(df):
    df = df.copy()
    # basic feature
    df['Return'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']

    # add Log_Close
    if 'Log_Close' not in df and 'Close' in df.columns:
        df['Log_Close'] = np.log(df['Close'].replace(0, 1e-8))
    df['Log_Return'] = df['Log_Close'].diff().fillna(0)
    # calculate the daily voliatitlity
    df['Volatility_Daily'] = df['Log_Return'].rolling(window=5, min_periods=1).std().fillna(
        0) * np.sqrt(252)

    # moving average feature
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA{window}_Ratio'] = df['Close'] / df[f'MA{window}']

    # the price range feature
    if all(col in df.columns for col in ['High', 'Low', 'Open']):
        df['Price_Range'] = (df['High'] - df['Low']) / df['Open'].replace(0, 1e-8)
    # volatility feature
    for window in [5, 10, 20, 50]:
        df[f'Volatility_{window}'] = df['Return'].rolling(window=window).std()

    # RSI feature
    for period in [7, 14, 21]:
        df[f'RSI{period}'] = compute_RSI(df['Close'], period=period)

    # MACD feature
    macd, signal_line = compute_MACD(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = macd - signal_line

    # Bollinger bands feature
    upper_band, lower_band = compute_bollinger_bands(df['Close'])
    df['BB_Upper'] = upper_band
    df['BB_Lower'] = lower_band
    df['BB_Width'] = (upper_band - lower_band) / df['Close']
    df['BB_Position'] = (df['Close'] - lower_band) / (upper_band - lower_band)

    # lag feature
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)
        df[f'Log_Return_Lag_{lag}'] = df['Log_Return'].shift(lag)

        if lag <= 5:  # only add feature when the lag is small
            df[f'Volatility_Lag_{lag}'] = df['Volatility_Daily'].shift(lag)

    # trend feature
    df['Trend_5_20'] = df['MA5'] - df['MA20']
    df['Trend_20_50'] = df['MA20'] - df['MA50']
    df['Trend_50_200'] = df['MA50'] - df['MA200']

    # Transaction volume feature
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']

    # Time feature - vectorized creation
    df['DayOfWeek'] = pd.Series(df.index.dayofweek, index=df.index)
    df['Month'] = pd.Series(df.index.month, index=df.index)
    df['Quarter'] = pd.Series(df.index.quarter, index=df.index)
    df['Day_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df['Day_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 5)
    df['Month_Sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['IsMonthEnd'] = pd.Series(df.index.is_month_end.astype(int), index=df.index)

    return df

#add future return(the target data) to our test
def add_future_return(df, N):
    df = df.copy()
    df['Future_Return'] = df['Close'].shift(-N) / df['Close'] - 1.0

    return df

#to fit the scale of x and y
def normalize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

#the models we use
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=4,
            min_samples_split=15,
            min_samples_leaf=15,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.7,
            min_samples_split=15,
            min_samples_leaf=15,
            random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.01,
            max_depth=3,
            gamma=0.1,
            min_child_weight=7,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.01,
            num_leaves=15,
            max_depth=4,
            min_child_samples=30,
            min_child_weight=1,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        ),
        'ElasticNet': ElasticNet(
            alpha=0.005,
            l1_ratio=0.7,
            max_iter=1000,
            tol=0.001,
            random_state=42
        ),
    }

    results = {}

    
    for name, model in models.items():
        print(f"training the model: {name}...")
        model.fit(X_train, y_train)

        #predict the value
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)

        results[name] = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'test_pred': y_pred_test
        }


    return results


def train_and_predict(df, N, feature_selection=None):
    # adding the target variable (future return)
    df_target = add_future_return(df, N)

    # choose the feature
    if feature_selection is None:
        feature_cols = [col for col in df_target.columns if col not in
                        ['Future_Return', 'Open', 'High', 'Low', 'Volume',
                         'Dividends', 'Stock Splits', 'Return']]
    else:
        feature_cols = feature_selection

    # data cleaning to solve nan data
    # make sure the data is ordered by time order
    df_target = df_target.sort_index(ascending=True)

    # deal with the infinity number
    df_target = df_target.replace([np.inf, -np.inf], np.nan)

    # filling the empty data
    # technical indicators
    ts_features = ['MA5', 'MA20', 'Volatility_20', 'RSI14', 'MACD']
    for col in ts_features:
        if col in df_target.columns:
            # Forward fill with the latest data
            df_target[col] = df_target[col].fillna(method='ffill')
            # Fill the remaining empty values with median
            df_target[col] = df_target[col].fillna(df_target[col].rolling(50, min_periods=1).median())

    # Use dynamic mean for other features
    other_features = list(set(feature_cols) - set(ts_features))
    for col in other_features:
        if df_target[col].isnull().sum() > 0:
            # 50 days mean
            df_target[col] = df_target[col].fillna(
                df_target[col].rolling(50, min_periods=1).mean().shift(1)
            )

    # Maintain data integrity for the past 6 months
    last_valid_date = df_target['Future_Return'].last_valid_index()
    if last_valid_date:
        cutoff_date = last_valid_date - pd.DateOffset(months=6)
        recent_data = df_target.loc[cutoff_date:]
        recent_data = recent_data.dropna(how='any')
        historical_data = df_target.loc[:cutoff_date].dropna(subset=feature_cols + ['Future_Return'])
        df_cleaned = pd.concat([historical_data, recent_data])
    else:
        df_cleaned = df_target.dropna(subset=feature_cols + ['Future_Return'])

    # Verify data integrity
    if df_cleaned.isnull().sum().sum() > 0:
        raise ValueError("Data cleaning failed，Residual NaN value！")
    if df_cleaned.empty:
        raise ValueError("No valid data after cleaning, please adjust the parameters")

    X = df_cleaned[feature_cols]
    y = df_cleaned['Future_Return']

    print("the shape of data X : ", X.shape)
    print("the shape of data y : ", y.shape)

    if X.empty or y.empty:
        raise ValueError("the train data is empty!。")

    # Training test set split (time series split, with the last 20% used as the test set)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # train all models and give the result
    models_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # generate the prediction dataframe
    full_pred = pd.DataFrame(index=df_target.index)
    full_pred['Actual'] = df_target['Future_Return']

    # Add predictions for each model
    for name, result in models_results.items():
        full_pred[f'Pred_{name}'] = np.nan
        # Only partially fill in the test set
        full_pred.loc[X_test.index, f'Pred_{name}'] = result['model'].predict(X_test)

    # Get the latest forecast for the last day
    X_latest = df_target[feature_cols].iloc[[-1]]
    predictions = {}
    for name, result in models_results.items():
        model = result['model']
        pred = model.predict(X_latest)[0]
        predictions[name] = pred

    return predictions, models_results, full_pred


def generate_pdf(report_text, filename="report.pdf"):
    pdf = FPDF()                              # create a pdf type object
    pdf.add_page()                            # add a new page to this object

    pdf.set_font("Helvetica", size=12)  # set the font of the pdf

    # Filter non-encodable characters in report_text


    for line in report_text.split("\n"):    # create lines separated by \n
        # Skip long and non-space lines for lines wrapping
        if len(line) > 300 and ' ' not in line:  # skip the line with more than 300 character
            print(f"Skip long and non-space lines: {line[:60]}...")
            continue

        # Line break and print
        wrapped_lines = textwrap.wrap(line, width=100, break_long_words=False)
        # Will not break a word to multiple lines. The maximum length for each line is 100 characters, will break into multiple lines if longer

        for wrapped_line in wrapped_lines:
            try:
            # Ensure text security without encoder error because default encoder for pdf is Latin-1
                pdf.multi_cell(0, 10, wrapped_line)  # put wrapped_line in pdf with given size cell with height h and line width
            except Exception as e:
                print(f"writing in failed: {wrapped_line[:50]}... error: {e}")     # indicate error with former 50 characters of the error line
                continue

    try:
        pdf.output(filename)                          # given the pdf object with a name and create a pdf file
        print(f"PDF1 Saved as{filename}")
    except Exception as e:
        print(f"Failed to save PDF1: {e}")

def generate_pdf2(userName, stocks, report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)


    # title
    str1 = f"{userName}'s Investment Report"
    pdf.cell(200, 10, txt=str1, ln=True, align='C')  # create a cell, add a title at centre, and change line


    pdf.set_font('Helvetica', size=12)
    pdf.ln(10)
    col_width = 25
    row_height = 10
    # first row: stock name
    stock_names = ['stockName'] + [stock['stockName'] for stock in stocks]    # stock_names is a list with string stockName and name of each input stock
    for name in stock_names:
        pdf.cell(col_width, row_height, name, border=1, align='C')   # create a cell, add each stock name to centre, and add the cell to pdf
    pdf.ln()                                                         # change line in pdf

    # second row: holding days
    holding_days = ['holdingDays'] + [str(stock['holdingDays']) for stock in stocks]  #create a cell for each holding day and add to pdf
    for days in holding_days:
        pdf.cell(col_width, row_height, days, border=1, align='C')
    pdf.ln()

    # report text
    pdf.ln(10)
    pdf.set_font('Helvetica', size=10)
    for line in report_text.split("\n"):         # create lines separated by \n
        # Skip long and space free lines
        if len(line) > 300 and ' ' not in line:
            continue

        wrapped_lines = textwrap.wrap(line, width=100, break_long_words=False)
        # Will not break a word to multiple lines. The maximum length for each line is 100 characters, will break into multiple lines if longer
        for wrapped_line in wrapped_lines:
            try:
            # Create cells for wrapped_lines
                pdf.multi_cell(0, 10, wrapped_line)
            except Exception as e:
                print(f"wrapped line failed: {wrapped_line[:50]}... error: {e}")
                continue

    pdf_file = "investment_report.pdf"
    try:
        pdf.output(pdf_file)
        print(f" PDF2 has been safed as {pdf_file}")
    except Exception as e:
        print(f"safe pdf2 failed: {e}")
    return pdf_file

def send_email(report_text, to_email, sender_email, sender_password, smtp_server, port):
    chart_html = ""

    # create HTML email content
    html = f"""
    <html>
    <body style="font-family:Arial;">
    <h2>Multi model Stock Return Prediction Report</h2>
    <br>
    <p>{report_text.replace('\n', '<br>')}</p>
    {chart_html}
    <hr><p style='color:gray; font-size:12px;'>The system automatically generates on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body>
    </html>
    """
    # <br> means change line in html

    # Write a local preview file and open it
    preview_path = os.path.abspath("email_preview.html")
    with open(preview_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("Preview file path:", preview_path)
    webbrowser.open(preview_path)            #open a test html website to test the correctness of text

    # generate PDF in the local computers running this program
    generate_pdf(report_text, filename="stock_prediction_report1.pdf")  # use the generate_pdf function to generate a pdf with filename as pdf name

    # send email
    msg = MIMEMultipart("alternative")                                  # create a MIMEMultipart object, specify the email type as alternative
    msg['Subject'] = "stock prediction report"                          # set email name
    msg['From'] = sender_email                                          # set senders' email address
    msg['To'] = to_email                                                # set receivers' email address
    msg.attach(MIMEText(report_text, 'plain', 'utf-8'))  # attach text in the pdf to the email
    try:
        server = smtplib.SMTP(smtp_server, port)                             # Create an SMTP server connection using the specified server and port
        server.starttls()                                                    # Enable TLS encryption for secure communication
        server.login(sender_email, sender_password)                          # Log in to the SMTP server with the sender's credentials
        server.sendmail(sender_email, [to_email], msg.as_string())   # Send the email to the recipient
        server.quit()
        print(" email 1 send！")
        return True
    except Exception as e:
        print(" email 1 not send：", e)
        return False

def send_email2(to_email, sender_email, sender_password, smtp_server, port, userName,
               overall_avg, current_time,stocks, earning_value, portfolio):

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = f"{userName}'s Investment Report"

    # email body
    body = f"""Dear Investor,

        Your projected portfolio return as of {current_time} is {overall_avg:.2f}%. 
        Total earning value is: {earning_value} and total value is: {portfolio}.

        Key Features:
        - Integrates stock history, geopolitical risks, and macroeconomic data.
        - Uses advanced algorithms (Random Forest, XGBoost) to identify volatility drivers.
        - Employs ensemble models with 30-day rolling validation for calibrated projections.
        - Refines outliers and ensures trend continuity for reliable insights.

        Our methodology cross-verifies market signals with quantitative frameworks, balancing 
        academic rigor with practical application. This equips institutional and active 
        investors with actionable, data-driven foresight.

        Thank you for your trust!

        Best regards,
        EclipseLoom Team
        Date: {current_time}
        """
    msg.attach(MIMEText(body, 'plain', 'utf-8'))      # attach body text to the email

    report_text = body
    # generate PDF in local computer running this program
    pdf_file = generate_pdf2(userName, stocks, report_text)

    # attach PDF to the email
    if os.path.exists(pdf_file):
        try:
            with open(pdf_file, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')       # Create a MIMEBase object for binary data
                part.set_payload(f.read())                                             # Read the file's content and set it as the payload for the MIME part
            encoders.encode_base64(part)                                                # Encode the payload in base64 to make it safe for transmission
            part.add_header('Content-Disposition', f'attachment; filename={pdf_file}')  # Add a header to specify that this part is an attachment with the original file name
            msg.attach(part)                                                            # Attach the encoded PDF file to the email
        except Exception as e:
            print(f" PDF failed: {e}")

    # send email
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        print(f" email 2 successfully send!")
        return True
    except Exception as e:
        print(f" email 2 send fail: {e}")
        return False


def main(userName, stocks, to_email, sender_email, sender_password, smtp_server="smtp.gmail.com", port=587):
    x = 0                   # x means the number of stocks input
    stock_hold = []         # a list to store the name of input stocks
    holding_days0 = []      # a list to store the holding days of input stocks
    holding_days = []       # a list to store the trading days of the stocks, which is calculated by the holding_days
    holding_value = []      # a list to store the holding values for each stock
    earning_value = 0       # total money earned

    #get data from stocks
    for stock in stocks:
        try:
            print("please input the stock code（like AAPL)：")
            stockName = stock['stockName']

            print("please input the holding time (day): ")
            holdingDays = stock['holdingDays']

            stockdata = get_stock_data(stockName, period="10y")
            lastPrice = stockdata.iloc[-1]['Close']
            print(f"stockvalue is:{lastPrice}")

            print("choose how much u need to buy： ")
            stockQuantity = stock['stockQuantity']

            stockvalue = stockQuantity * lastPrice

            stock_hold.append(stockName)
            holding_days0.append(holdingDays)
            holding_days.append(0)
            holding_value.append(stockvalue)
            x += 1
        except Exception:
            print("Error occurred, please re-enter")
    if x < 5:
        return False, "Error: There are less than 5 valid stocks"


    str = ""
    time_expense = max(holding_days0)
    # Replace the holding date with the number of trading days, and then 'holding days' represents the number of trading days
    trade_dates = {ticker: 0 for ticker in stock_hold}   # trade_dates
    for i, ticker in enumerate(stock_hold):  # （index, elements) in stock
        start_date = datetime.now()
        end_date = start_date + timedelta(days=holding_days0[i])  # use holding days
        current_date = start_date
        while current_date <= end_date:
            if is_trading_day(current_date, ticker):
                trade_dates[ticker] += 1
            current_date += timedelta(days=1)
        holding_days[i] = trade_dates[ticker]
        str += f"{ticker} on {holding_days0[i]}. The trading date for the holding day is {holding_days[i]}\n"
        print(str)

    # Build detailed report text
    report_text = "Enhanced Stock Prediction with Multiple Models\n\n"
    report_text += f" Prediction deadline: Future{time_expense} days\n\n"

    all_predictions = {}  # Store the predicted results of all stocks

    for i in range(5):
        ticker = stock_hold[i]
        all_models = {}  # Model results for each stock

        # Attempt to obtain economic data
        try:
            economic_data = get_fred_data()
            print("Successfully obtained economic data")
        except Exception as e:
            print(f"Failed to obtain economic data: {e}")
            economic_data = None

        print(f"\n=====  Processing stocks{ticker} =====")
        stock_df = get_stock_data(ticker, period="10y")
        if stock_df.empty:
            print(f"Failed to obtain stock data of{ticker} ！")
            continue

        # get gpr data
        gpr_df = get_gpr_data()
        if gpr_df is None or gpr_df.empty:
            print("GPR data acquisition failed, not using GPR index.")
            merged_df = stock_df
        else:
            merged_df = merge_stock_gpr(stock_df, gpr_df)

        # Merge economic data
        if economic_data is not None:
            merged_df = merge_economic_data(merged_df, economic_data)

        # featuring df
        print(f"featuring df for {ticker} ...")
        feature_df = add_features(merged_df)

        try:
            # Train multiple models and make predictions
            print(f"train model for {ticker} ...")
            predictions, models_results, prediction_df = train_and_predict(feature_df, holding_days[i])

            # Store results
            all_predictions[ticker] = predictions
            all_models[ticker] = models_results

        except ValueError as ve:
            print(f"There were errors in the training {ticker} : ", ve)
            continue
        except Exception as e:
            print(f"An unexpected error occurred during the processing{ticker} : ", e)
            continue

    if not all_predictions:
        return False, "No stocks were successfully predicted!"

    # Add the prediction results of each model for each stock
    for ticker, predictions in all_predictions.items():
        report_text += f"===== {ticker}predict  results  =====\n"
        model_predictions = sorted(
            [(model, pred * 100) for model, pred in predictions.items()],
            key=lambda x: x[1], reverse=True
        )
        for model_name, prediction in model_predictions:
            report_text += f" {model_name}: {prediction:.2f}%\n"
        report_text += "\n"

    # calculate the comprehensive prediction (average) of all models
    all_values = []
    for predictions in all_predictions.values():
        all_values.extend(predictions.values())

    # Check if all_values contains any data to avoid division by zero
    overall_avg = 0
    if all_values:
        overall_avg = sum(all_values) / len(all_values) * 100
        report_text += f"\nThe average predicted return rate of all models: {overall_avg:.2f}%\n"
        for i in range(5):
            earning_value += (overall_avg / 100) * holding_value[i]

    # Add results for the whole prediction
    report_text += "=============== total predict ===============\n"
    report_text += f"total return: {earning_value}\n"
    report_text += f"holding times: {time_expense} days\n"
    report_text += f"\ndata source: Yahoo Finance, Matteo Iacoviello's GPR, FRED\n"
    report_text += f"report generate time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    print("\npredict result: ")
    print(report_text)

    portfolio = earning_value + sum(holding_value)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    success1 = send_email(report_text, to_email, sender_email, sender_password, smtp_server, port)
    success2 = send_email2(to_email, sender_email, sender_password, smtp_server, port, userName, overall_avg,
                           current_time,  stocks, earning_value, portfolio)
    if success1 and success2:
        return True, "Prediction completed, results have been sent to your email"
    else:
        return False, "Email sending failed"

