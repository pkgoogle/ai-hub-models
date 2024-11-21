#!/bin/bash

# --- Configuration ---
BENCHMARK_MODEL_EXE="benchmark_tflite_model"  # Replace with the name of your executable

AI_EDGE_TORCH_DIRECTORY="/data/local/tmp/ai_edge_torch_models"
QUALCOMM_DIRECTORY="/data/local/tmp/qualcomm_models"

# Check if ADB is available
if ! command -v adb &> /dev/null
then
    echo "ADB not found. Please install ADB and make sure it's in your PATH."
    exit 1
fi

# Check if a device is connected
if ! adb devices | grep -q "device$"
then
    echo "No Android device found. Connect a device and ensure USB debugging is enabled."
    exit 1
fi

echo "Entering ai_edge_torch directory"
# Benchmark all the AI-Edge-Torch Models and Log the results
models=($(adb shell "ls \"$AI_EDGE_TORCH_DIRECTORY\""))
for model in "${models[@]}"; do
    echo "Benchmarking: $model"
    # CPU
    COMMAND="/data/local/tmp/${BENCHMARK_MODEL_EXE} --graph=${AI_EDGE_TORCH_DIRECTORY}/${model} > /data/local/tmp/logs/${model}.aet.cpu.log 2>&1"
    adb shell "${COMMAND}"
    # GPU, cl backend
    COMMAND="/data/local/tmp/${BENCHMARK_MODEL_EXE} --graph=${AI_EDGE_TORCH_DIRECTORY}/${model} --use_gpuv3=true --gpu_backend=cl > /data/local/tmp/logs/${model}.aet.gpu.cl.log 2>&1"
    adb shell "${COMMAND}"
    # GPU, webgpu backend
    COMMAND="/data/local/tmp/${BENCHMARK_MODEL_EXE} --graph=${AI_EDGE_TORCH_DIRECTORY}/${model} --use_gpuv3=true --gpu_backend=webgpu > /data/local/tmp/logs/${model}.aet.gpu.web.log 2>&1"
    adb shell "${COMMAND}"
done

echo "Entering qualcomm directory"
# Benchmark all the Qualcomm Models and Log the results
models=($(adb shell "ls \"$QUALCOMM_DIRECTORY\""))
for model in "${models[@]}"; do
    echo "Benchmarking: $model"
    # CPU
    COMMAND="/data/local/tmp/${BENCHMARK_MODEL_EXE} --graph=${QUALCOMM_DIRECTORY}/${model} > /data/local/tmp/logs/${model}.qc.cpu.log 2>&1"
    adb shell "${COMMAND}"
    # GPU, cl backend
    COMMAND="/data/local/tmp/${BENCHMARK_MODEL_EXE} --graph=${QUALCOMM_DIRECTORY}/${model} --use_gpuv3=true --gpu_backend=cl > /data/local/tmp/logs/${model}.qc.gpu.cl.log 2>&1"
    adb shell "${COMMAND}"
    # GPU, webgpu backend
    COMMAND="/data/local/tmp/${BENCHMARK_MODEL_EXE} --graph=${QUALCOMM_DIRECTORY}/${model} --use_gpuv3=true --gpu_backend=webgpu > /data/local/tmp/logs/${model}.qc.gpu.web.log 2>&1"
    adb shell "${COMMAND}"
done
