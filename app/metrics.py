from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "api_requests_total", "Total number of API requests received", ["method", "endpoint","http_status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Latency of API requests in seconds", ["endpoint"]
)

PREDICTION_COUNT = Counter(
    "predictions_total", "Total number of predictions made", ["label"]
)