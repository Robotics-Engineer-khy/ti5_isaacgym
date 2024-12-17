import os
import subprocess

def export_mnn_t1(mnn_dir, onnx_file):

    # ONNX导出为mnn
    format_type = "ONNX"

    mnn_model = mnn_dir + "/ti5_dh_policy.mnn"
    biz_code = "biz"
    mnn_convert_tool = mnn_dir + "/MNNConvert"
    command = f"{mnn_convert_tool} -f {format_type} --modelFile {onnx_file} --MNNModel {mnn_model} --bizCode {biz_code}"
    print(f"Executing command: {command}")
    subprocess.run(command, shell=True)

    print(f"MNN model saved to {mnn_model}")