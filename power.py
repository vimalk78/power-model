# Import necessary libraries
import pandas as pd
import logging
from datetime import datetime, timedelta
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from prometheus_api_client import PrometheusConnect, MetricSnapshotDataFrame, MetricRangeDataFrame

PROM_URL = "http://localhost:9090"

METRIC_BPF_CPU_TIME = "kepler_process_bpf_cpu_time_ms_total"
COLUMNS_COMMAND_PID = ['command', 'pid']

METRIC_CPU_INSTRUCTIONS = "kepler_process_cpu_instructions_total"

METRIC_CPU_TIME = "cpu_time"

METRIC_NODE_RAPL_PKG_JOULES_TOTAL = "node_rapl_package_joules_total"
LABEL_RAPL_PATH = "/host/sys/class/powercap/intel-rapl:0"

METRIC_UP = "up"


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

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Display the coefficients of the model
    print("Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef}")

    return model


def test_california_housing():
    # Fetch the California Housing dataset
    california = fetch_california_housing()
    data = pd.DataFrame(california.data, columns=california.feature_names)
    data['PRICE'] = california.target

    # Display the first few rows of the dataset
    print(data.head())

    # Train the model
    model = train_lr(data)
    return model


def fetch_prometheus_data(start_time, end_time, metric_name, df, value_column, columns=[], labels=None):
    try:
        # Connect to Prometheus
        prom = PrometheusConnect(url=PROM_URL, disable_ssl=True)

        # Fetch the metric data with optional labels
        metric_data = prom.get_metric_range_data(metric_name=metric_name, start_time=start_time, end_time=end_time, label_config=labels)

        if not metric_data:
            raise ValueError(f"No data found for metric: {metric_name}")

        # print(f"Size of the list: {len(metric_data)}")
        # if metric_data:
        #     print(metric_data)

        # Convert the metric data to a DataFrame
        metric_df = MetricRangeDataFrame(data=metric_data, columns=(columns + ['timestamp', 'value']), ts_as_datetime=False)
        metric_df.index = metric_df.index.astype('int64')

        # Rename 'value' column to the value of parameter value_column
        metric_df.rename(columns={'value': value_column}, inplace=True)

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


def test_kepler_power_model():
    # Create an empty DataFrame
    df = pd.DataFrame()
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=10)

    # Call fetch_and_add_prometheus_data for each metric
    bpf_df = fetch_prometheus_data(
        start_time=start_time,
        end_time=end_time,
        metric_name=METRIC_BPF_CPU_TIME,
        df=df,
        value_column='bpf_cpu_time',
        columns=COLUMNS_COMMAND_PID)
    bpf_df.attrs = {"name": "bpf_df"}
    printDF(bpf_df)

    pkg_joules_df = fetch_prometheus_data(
        start_time=start_time,
        end_time=end_time,
        metric_name=METRIC_NODE_RAPL_PKG_JOULES_TOTAL,
        df=df,
        value_column='pkg_joules',
        labels={"path": LABEL_RAPL_PATH})
    pkg_joules_df.attrs = {"name": "pkg_joules_df"}
    printDF(pkg_joules_df)

    cpu_inst_df = fetch_prometheus_data(
        start_time=start_time,
        end_time=end_time,
        metric_name=METRIC_CPU_INSTRUCTIONS,
        df=df,
        value_column='cpu_instructions',
        columns=COLUMNS_COMMAND_PID)
    cpu_inst_df.attrs = {"name": "cpu_inst_df"}
    printDF(cpu_inst_df)

    # Display the DataFrame
    # print(df)

    return df


if __name__ == "__main__":
    # pd.options.display.float_format = '{:20.8f}'.format
    logging.basicConfig(level=logging.ERROR)
    test_kepler_power_model()

if __name__ == "##__main__":
    # pd.options.display.float_format = '{:20.8f}'.format
    logging.basicConfig(level=logging.DEBUG)
    test_california_housing()
