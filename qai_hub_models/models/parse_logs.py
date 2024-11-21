import argparse
import csv
import os
import re


DELEGATE_NAME_MAP = {
    "cpu": "TfLiteXNNPackDelegate",
    "gpu": {
        "cl": "ML_DRIFT_CL",
        "web": "ML_DRIFT_WEBGPU",
    }
}


# Define a util function to parse the log
def extract_info_from_benchmark_log(text, delegate_name):
    """Extracts information from the given text using regular expressions.

    Args:
        text (str): The text to analyze.
        delegate_name (str): Name of the delegate used (e.g. "ML_DRIFT_CL", "ML_DRIFT_WEBGPU", or "TfLiteXNNPackDelegate").

    Returns:
        dict: A dictionary containing the extracted information.
    """

    info = {
        "inference_time": None,
        "gpu_nodes": None,
        "cpu_nodes": None,
        "partitions": None,
        "status": None,
        "unsupported_ops": []
    }

    # Extract inference time
    inference_time_pattern = r"Inference \(avg\): ([\d.e+]+)"  # Match integer part of microseconds
    match = re.search(inference_time_pattern, text)
    if match:
        info["inference_time"] = float(match.group(1)) / 1000  # Convert to milliseconds

    # Extract number of replaced nodes, total nodes, and partitions
    replace_nodes_partitions_pattern = (
        r"Replacing (\d+) out of (\d+) node\(s\) with delegate \("
        + delegate_name
        + r"\) node, yielding (\d+) partitions"
    )
    match = re.search(replace_nodes_partitions_pattern, text)
    if match:
        info["gpu_nodes"] = int(match.group(1))
        info["cpu_nodes"] = int(match.group(2)) - info["gpu_nodes"]
        info["partitions"] = int(match.group(3))

    # Extract information from the line indicating unsupported operations
    unsupported_ops_pattern = r"ERROR: Following operations are not supported by GPU delegate:(.*?)(\d+) operations will run on the GPU, and the remaining (\d+) operations will run on the CPU."
    match = re.search(unsupported_ops_pattern, text, re.DOTALL)
    if match:
        info["unsupported_ops"] = [line.strip() for line in match.group(1).strip().split('\n')]

    # extract status
    if info["inference_time"]:
        if info["cpu_nodes"] == 0:
            info["status"] = "fully delegated"
        else:
            info["status"] = "partly delegated"
    elif info["gpu_nodes"]:
        info["status"] = "error running on GPU"
    else:
        info["status"] = "not convertible to TFLite"

    return info


def extract_info(directory, output_csv_path):
    with open(os.path.join(output_csv_path, f"benchmark_results.csv"), "w", newline="") as csvfile:

        fieldnames = ["model_name", "converter", "status", "inference_time", "cpu_nodes", "gpu_nodes", "partitions", "unsupported_ops", "processor", "backend"]
        valid_cpu_fields = ["model_name", "converter", "inference_time", "processor"]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filename in os.listdir(directory):
            if filename.endswith(".log"):
                parts = filename.split('.')
                model_name = parts[0]
                # model_format = parts[1]
                conversion_library = parts[2]
                processor = parts[3]
                if processor == "gpu":
                    backend = parts[4]
                    delegate_name = DELEGATE_NAME_MAP["gpu"][backend]
                else:
                    backend = "n/a"
                    delegate_name = DELEGATE_NAME_MAP["cpu"]

                file_path = os.path.join(directory, filename)
                with open(file_path, "r") as file:
                    file_text = file.read()
                    info = extract_info_from_benchmark_log(file_text, delegate_name=delegate_name)

                info["model_name"] = model_name
                info["converter"] = conversion_library
                info["processor"] = processor
                info["backend"] = backend

                if processor == "cpu":
                    for k in info.keys():
                        if k not in valid_cpu_fields:
                            info[k] = "n/a"

                writer.writerow(info)

    print("Wrote benchmark result to CSV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parses all the logs in the directory into a csv")

    parser.add_argument("directory", help="The log directory")
    parser.add_argument("outpath", help="where to write the output")

    args = parser.parse_args()
    extract_info(args.directory, args.outpath)
