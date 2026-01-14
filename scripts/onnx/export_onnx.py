from ultralytics import YOLO
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

if __name__ == '__main__':
    # 加载最佳模型
    model = YOLO("runs/train/pcb_finetune_v1/weights/best.pt")

    # 导出目录
    export_dir = "exports/onnx"
    os.makedirs(export_dir, exist_ok=True)

    # FP32 ONNX
    fp32_path = model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
    fp32_final = os.path.join(export_dir, "yolov8n_pcb_fp32.onnx")
    os.rename(fp32_path, fp32_final)
    print(f"FP32 ONNX导出完成: {fp32_final}")

    # FP16 ONNX
    fp16_path = model.export(format="onnx", imgsz=640, half=True, dynamic=True, simplify=True)
    fp16_final = os.path.join(export_dir, "yolov8n_pcb_fp16.onnx")
    os.rename(fp16_path, fp16_final)
    print(f"FP16 ONNX导出完成: {fp16_final}")

    # INT8量化（兼容onnxruntime 1.17.1版本，移除activation_type参数，使用默认QUInt8激活）
    int8_final = os.path.join(export_dir, "yolov8n_pcb_int8.onnx")
    quantize_dynamic(
        model_input=fp32_final,
        model_output=int8_final,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True
    )
    print(f"INT8量化完成: {int8_final}")

    print("\n所有ONNX模型导出&量化完成！文件在 exports/onnx/")
    print("提示：YOLO检测模型动态量化默认使用QUInt8激活，兼容性好，精度损失小。")