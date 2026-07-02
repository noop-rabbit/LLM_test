from pynvml import *
import time

nvmlInit()

device_count = nvmlDeviceGetCount()
print(f"Total GPUs found --> {device_count}")

while True:
    
    handle = nvmlDeviceGetHandleByIndex(0)

    name = nvmlDeviceGetName(handle)
    driver_ver = nvmlSystemGetDriverVersion()

    memory_info = nvmlDeviceGetMemoryInfo(handle)
    
    total_mem = memory_info.total / (1024 ** 3)
    used_mem = memory_info.used / (1024 ** 3)

    utilization = nvmlDeviceGetUtilizationRates(handle)

    temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
    power = nvmlDeviceGetPowerUsage(handle) / 1000.0

    print(f"\n--- GPU {0}: {name} ---")
    print(f"Driver Version: {driver_ver}")
    print(f"Memory: {used_mem:.2f} GB / {total_mem:.2f} GB used")
    print(f"GPU Compute Utilization: {utilization.gpu}%")
    print(f"Temperature: {temp}°C")
    print(f"Power Usage: {power:.2f} W")
    time.sleep(0.1)

nvmlShutdown()