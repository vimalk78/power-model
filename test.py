import requests
import time

prometheus_url = 'http://localhost:9090'
metric_bpf_cpu_time = 'kepler_process_bpf_cpu_time_ms_total'
metric_node_rapl_power = 'node_rapl_core_joules_total'


def fetch_prometheus_metrics(url, metric_name):
    """
    Fetch a specific metric from a Prometheus server.

    Args:
    url (str): The URL of the Prometheus server's metrics endpoint.
    metric_name (str): The name of the metric to query.

    Returns:
    str: The raw metrics data for the specified metric.
    """
    response = requests.get(f"{url}/api/v1/query", params={'query': metric_name})
    response.raise_for_status()  # Raise an error for bad status codes
    data = response.json()
    return data['data']['result']

def compute_metric_sum(metrics):
    """
    Compute the sum of the values in the Prometheus metrics data.

    Args:
    metrics (list): The list of metric data returned by Prometheus.

    Returns:
    float: The sum of the metric values.
    """
    total = 0.0
    for metric in metrics:
        total += float(metric['value'][1])
    for metric in metrics:
        metric['ratio'] = float(metric['value'][1]) / total
        metric['total'] = total
    return total


def test():
    metrics = fetch_prometheus_metrics(prometheus_url, metric_bpf_cpu_time)
    compute_metric_sum(metrics)
    print(metrics)


def read_cpu_power_from_rapl():
    """
    Reads the CPU domain power from RAPL (Running Average Power Limit).

    Returns:
    str: The CPU power consumption data.
    """
    try:
        with open('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', 'r') as file:
            power_data = file.read().strip()
        return str(float(power_data) / 10**6)
    except FileNotFoundError:
        return "RAPL interface not found."
    except Exception as e:
        return f"Error reading RAPL data: {e}"



if __name__ == "__main__":
    while True:
        cpu_power = read_cpu_power_from_rapl()
        # print(f"CPU Power from RAPL: {cpu_power}")
        test()
        time.sleep(1)  # Sleep for 1 second before reading the value again
