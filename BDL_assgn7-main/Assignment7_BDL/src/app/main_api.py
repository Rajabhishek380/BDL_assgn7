from fastapi import FastAPI, File, UploadFile, Request
import uvicorn
import os
import io
import sys
import numpy as np
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge
import time
import psutil

# Environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Prometheus metrics
API_USAGE_COUNTER = Counter("api_usage_counter", "API usage counter", ["client_ip"])
PROCESSING_TIME_GAUGE = Gauge("processing_time_gauge", "Processing time of the API", ["client_ip"])
CPU_UTILIZATION_GAUGE = Gauge("cpu_utilization_gauge", "CPU utilization during processing", ["client_ip"])
MEMORY_UTILIZATION_GAUGE = Gauge("memory_utilization_gauge", "Memory utilization during processing", ["client_ip"])
NETWORK_IO_BYTES_GAUGE = Gauge("network_io_bytes_gauge", "Network I/O bytes during processing", ["client_ip"])
NETWORK_IO_BYTES_RATE_GAUGE = Gauge("network_io_bytes_rate_gauge", "Network I/O bytes rate during processing", ["client_ip"])
API_RUNTIME_GAUGE = Gauge("api_runtime_gauge", "API runtime", ["client_ip"])
API_TL_TIME_GAUGE = Gauge("api_tl_time_gauge", "API T/L time", ["client_ip"])

app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Dummy model prediction function
def predict_digit(data_point: list) -> str:
    data_point = np.array(data_point, dtype=np.float32) / 255.0  # Normalize data
    data_point = data_point.reshape(1, -1)  # Reshape to (1, 784)
    return str(np.random.randint(10))  # Replace with actual model prediction

# Resize image to 28x28
def format_image(image: Image) -> Image:
    return image.resize((28, 28))

# Calculate processing time per character
def calculate_processing_time(start_time: float, length: int) -> float:
    total_time = time.time() - start_time
    return (total_time / length) * 1e6

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post('/predict')
async def predict_digit_api(request: Request, upload_file: UploadFile = File(...)):
    start_time = time.time()
    client_ip = request.client.host

    # Read and process the uploaded image
    contents = await upload_file.read()
    image = Image.open(io.BytesIO(contents)).convert('L')
    image = format_image(image)
    image_array = np.array(image)
    
    # Collect metrics
    API_USAGE_COUNTER.labels(client_ip=client_ip).inc()
    cpu_percent = psutil.cpu_percent(interval=1)
    CPU_UTILIZATION_GAUGE.labels(client_ip=client_ip).set(cpu_percent)
    memory_info = psutil.virtual_memory()
    MEMORY_UTILIZATION_GAUGE.labels(client_ip=client_ip).set(memory_info.percent)
    net_io = psutil.net_io_counters()
    NETWORK_IO_BYTES_GAUGE.labels(client_ip=client_ip).set(net_io.bytes_sent + net_io.bytes_recv)
    NETWORK_IO_BYTES_RATE_GAUGE.labels(client_ip=client_ip).set((net_io.bytes_sent + net_io.bytes_recv) / (time.time() - start_time))
    
    # Flatten image array and predict digit
    data_point = image_array.flatten().tolist()
    prediction = predict_digit(data_point)
    processing_time = calculate_processing_time(start_time, len(data_point))
    PROCESSING_TIME_GAUGE.labels(client_ip=client_ip).set(processing_time)
    
    # Calculate API runtime and T/L time
    api_runtime = time.time() - start_time
    API_RUNTIME_GAUGE.labels(client_ip=client_ip).set(api_runtime)
    api_tltime = api_runtime / len(image_array)
    API_TL_TIME_GAUGE.labels(client_ip=client_ip).set(api_tltime)

    return {"predicted_digit": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
