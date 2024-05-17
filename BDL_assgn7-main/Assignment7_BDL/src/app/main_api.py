from fastapi import FastAPI, File, UploadFile,Request
import uvicorn
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import io
import sys
#from keras.models import load_model
#from tensorflow.keras.models import  Sequential
import numpy as np
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge
import time
import psutil
#Write code for memory utilisation
API_USAGE_COUNTER = Counter("api_usage_counter", "API usage counter", ["client_ip"])
PROCESSING_TIME_GAUGE = Gauge("processing_time_gauge", "Processing time of the API", ["client_ip"])
CPU_UTIL_TIME=Gauge("cpu_utilization_gauge", "CPU utilization during processing", ["client_ip"])
MEMORY_UTILIZATION_GAUGE = Gauge("MEMORY_UTILIZATION_GAUGE", "Memory utilization during processing", ["client_ip"])
NETWORK_IO_BYTES_GAUGE = Gauge("NETWORK_IO_BYTES_GAUGE", "Network I/O bytes during processing", ["client_ip"])
NETWORK_IO_BYTES_RATE_GAUGE = Gauge("NETWORK_IO_BYTES_RATE_GAUGE", "Network I/O bytes rate during processing", ["client_ip"])
API_RUNTIME_GAUGE = Gauge("API_RUNTIME_GAUGE", "API runtime", ["client_ip"])
API_TL_TIME_GAUGE = Gauge("API_TL_TIME_GAUGE", "API T/L time", ["client_ip"])
model_path = sys.argv[1] if len(sys.argv) > 1 else None
app = FastAPI()
model=None
Instrumentator().instrument(app).expose(app)
def predict_digit(data_point:list)->str:
    data_point = np.array(data_point, dtype=np.float32) / 255.0 #Data normalisation
    data_point = data_point.reshape(1, -1) #reshaping to None,784
    #predict_digit = model.predict(data_point) #Get the predicitons
    return str(np.random.randint(10))
#resizes image of any size to 28x28
def calculate_processing_time(start_time: float, length: int) -> float:
    # Calculate processing time per character
    end_time = time.time()
    total_time = end_time - start_time
    return total_time / length * 1e6 
def format_image(image: Image) -> Image: #This is part 2 of the problem statement
    image = image.resize((28, 28))
    return image
@app.get("/")
async def root():
    return {"message": "Hello World"} #Test for checking whether  the server is running or not( Not a part of the original problem statement)
#Gets model from the local address where it is stored
@app.post('/predict',response_model=None)
async def predict_digit_api(request: Request,upload_file: UploadFile = File(...)):
    start_time = time.time()
    #global model
    #if not model:
        #return {"error": "Model is not loaded."}
    contents = await upload_file.read()
    client_ip = request.client.host #gets the client IP
    # Open the image using PIL
    API_USAGE_COUNTER.labels(client_ip=client_ip).inc() #Count the API calls
    image = Image.open(io.BytesIO(contents)).convert('L') #Converts image to black and white
    # Resize the image to 28x28
    image=format_image(image)
    # Convert the image to a numpy array
    image_array = np.array(image)
    input_len=len(image_array)
    cpu_percent=psutil.cpu_percent(interval=1) #Get the CPU usage percentage
    CPU_UTIL_TIME.labels(client_ip=client_ip).set(cpu_percent) #Change name to percentage
    memory_info = psutil.virtual_memory() #Get the memory information
    MEMORY_UTILIZATION_GAUGE.labels(client_ip=client_ip).set(memory_info.percent) #Set the memory utilization
    net_io = psutil.net_io_counters()   #Get the network I/O
    NETWORK_IO_BYTES_GAUGE.labels(client_ip=client_ip).set(net_io.bytes_sent + net_io.bytes_recv) #Adds the sent and received network bytes
    NETWORK_IO_BYTES_RATE_GAUGE.labels(client_ip=client_ip).set((net_io.bytes_sent + net_io.bytes_recv) / (time.time() - start_time)) #Calculates the rate
    # Flatten the image array to a 1D array
    data_point = image_array.flatten().tolist()
    prediction = predict_digit(data_point)
    processing_time = calculate_processing_time(start_time, len(data_point)) #Get the processing time
    PROCESSING_TIME_GAUGE.labels(client_ip=client_ip).set(processing_time) #Store it for prometheus
    # Calculate API runtime
    api_runtime = time.time() - start_time
    API_RUNTIME_GAUGE.labels(client_ip=client_ip).set(api_runtime) #Get the run time
    api_tltime=api_runtime/input_len
    # Calculate API T/L time (assuming T/L time is the total time for processing this request)
    API_TL_TIME_GAUGE.labels(client_ip=client_ip).set(api_tltime)
    return {"predicted_digit": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000) #This sets up host in 127.0.0.1:8000