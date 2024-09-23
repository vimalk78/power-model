# Import necessary libraries
from datetime import datetime, timedelta
import logging

import pandas as pd
from prometheus_api_client import MetricRangeDataFrame, PrometheusConnect
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from sklearn.metrics import mean_absolute_percentage_error


PROM_URL = "http://localhost:9090"

RATE_INTERVAL = "20s"

METRIC_BPF_CPU_TIME = "kepler_process_bpf_cpu_time_ms_total"
QUERY_BPF_CPU_TIME = "rate(kepler_process_bpf_cpu_time_ms_total{}[{}])"
COLUMNS_COMMAND_PID = ['command', 'pid']

METRIC_CPU_INSTRUCTIONS = "kepler_process_cpu_instructions_total"
QUERY_CPU_INSTRUCTIONS = "rate(kepler_process_cpu_instructions_total{}[{}])"

METRIC_CPU_TIME = "cpu_time"

METRIC_NODE_RAPL_PKG_JOULES_TOTAL = "node_rapl_package_joules_total"
QUERY_NODE_RAPL_PKG_JOULES_TOTAL = "rate(node_rapl_package_joules_total{}[{}])"
LABEL_RAPL_PATH = "/host/sys/class/powercap/intel-rapl:0"

METRIC_PKG_JOULES_TOTAL = "kepler_process_package_joules_total"
QUERY_KEPLER_PKG_JOULES_TOTAL = "rate(kepler_process_package_joules_total{}[{}])"

METRIC_UP = "up"

logging.basicConfig(level=logging.ERROR)

# Load the California Housing dataset and train the model
def train_lr(data):
    # Define the features (X) and the target (y)
    X = data.drop('PRICE', axis=1)
    y = data['PRICE']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Make predictions using the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"MAPE: {mape}")

    # Display the coefficients of the model
    print("Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef}")

    return model


def test_california_housing():
    california: Bunch = fetch_california_housing()
    data: pd.DataFrame = pd.DataFrame(california.data, columns=california.feature_names)
    data['PRICE'] = california.target

    # Display the first few rows of the dataset
    print(data.head())

    # Train the model
    model = train_lr(data)
    return model


def fetch_prometheus_data(start_time, end_time, query, rename_value_column, columns=[], label_config: dict = None):
    try:
        # Connect to Prometheus
        prom = PrometheusConnect(url=PROM_URL, disable_ssl=True)

        if label_config:
            label_list = [str(key + "=~" + "'" + label_config[key] + "'") for key in label_config]
            labels = "{" + ",".join(label_list) + "}"
        else:
            labels = ""
        query = query.format(labels, RATE_INTERVAL)

        # Fetch the metric data with optional labels
        metric_data = prom.custom_query_range(query=query, start_time=start_time, end_time=end_time, step="1s")

        if not metric_data:
            raise ValueError(f"No data found for metric: {query}")

        # print(f"Size of the list: {len(metric_data)}")
        # if metric_data:
        #     print(metric_data)

        # Convert the metric data to a DataFrame
        metric_df = MetricRangeDataFrame(data=metric_data, columns=(columns + ['timestamp', 'value']), ts_as_datetime=False)
        metric_df.index = metric_df.index.astype('int64')

        # Rename 'value' column to the value of parameter value_column
        metric_df.rename(columns={'value': rename_value_column}, inplace=True)

        # Sort the DataFrame by timestamp
        metric_df = metric_df.sort_values(by='timestamp')

        return metric_df

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return pd.DataFrame()


def printDF(df: pd.DataFrame):
    df_name = df.attrs['name']
    print(f"{df_name} shape: {df.shape}, Columns: {df.columns}")
    print(df)


def test_kepler_power_model(end_time, duration):
    start_time = end_time - timedelta(milliseconds=duration)

    # Call fetch_and_add_prometheus_data for each metric
    bpf_df = fetch_prometheus_data(
        start_time=start_time,
        end_time=end_time,
        query=QUERY_BPF_CPU_TIME,
        rename_value_column='bpf_cpu_time',
        # label_config={"command": ".*stress.*"},
        columns=COLUMNS_COMMAND_PID)
    bpf_df.attrs = {"name": "bpf_df"}
    bpf_cpu_time_total = bpf_df['bpf_cpu_time'].sum()
    bpf_df['bpf_cpu_time_ratio'] = bpf_df['bpf_cpu_time'] / bpf_cpu_time_total
    printDF(bpf_df)

    pkg_joules_df = fetch_prometheus_data(
        start_time=start_time,
        end_time=end_time,
        query=QUERY_NODE_RAPL_PKG_JOULES_TOTAL,
        rename_value_column='pkg_joules',
        label_config={"path": LABEL_RAPL_PATH})
    pkg_joules_df.attrs = {"name": "pkg_joules_df"}
    printDF(pkg_joules_df)

    cpu_inst_df = fetch_prometheus_data(
        start_time=start_time,
        end_time=end_time,
        query=QUERY_CPU_INSTRUCTIONS,
        rename_value_column='cpu_instructions',
        # label_config={"command": ".*stress.*"},
        columns=COLUMNS_COMMAND_PID)
    cpu_inst_df.attrs = {"name": "cpu_inst_df"}
    printDF(cpu_inst_df)



def test_main():
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Fetch Prometheus data and train models.")
    parser.add_argument('--end_time', type=int, default=int(datetime.now().timestamp()), help='End time as a Unix timestamp')
    parser.add_argument('--duration', type=int, default=300000, help='Duration in milliseconds')

    args = parser.parse_args()
    # Calculate start_time from end_time and duration
    end_time = datetime.fromtimestamp(args.end_time)
    start_time = end_time - timedelta(milliseconds=args.duration)

    print(f"End time  : {args.end_time}")
    print(f"Start time: {int(start_time.timestamp())}")
    print(f"Duration  : {args.duration}")

    test_kepler_power_model(end_time, args.duration)


def test_housing():
    # pd.options.display.float_format = '{:20.8f}'.format
    logging.basicConfig(level=logging.DEBUG)
    test_california_housing()

def test_labels():
    print(QUERY_BPF_CPU_TIME)
    s = QUERY_BPF_CPU_TIME.format("", RATE_INTERVAL)
    print(s)
    label_config = {"path": LABEL_RAPL_PATH}
    label_list = [str(key + "=" + "'" + label_config[key] + "'") for key in label_config]
    labels = "{" + ",".join(label_list) + "}"
    s = QUERY_NODE_RAPL_PKG_JOULES_TOTAL.format(labels, RATE_INTERVAL)
    print(s)

if __name__ == "__main__":
    print(f"scikit-learn version: {sklearn.__version__}")
    test_main()
