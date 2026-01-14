from ultralytics import YOLO
import os

if __name__ == '__main__':
    int8_path = "exports/onnx/yolov8n_pcb_int8.onnx"
    data_yaml = "data/data.yaml"

    print(f"\n评估 INT8 ONNX (CPU): {int8_path}")
    model = YOLO(int8_path, task='detect')  # 显式指定 task='detect'，防止 YOLO 猜错任务类型

    val_results = model.val(
        data=data_yaml,
        imgsz=640,
        batch=1,                # CPU batch=1稳
        device="cpu",
        plots=False,
        verbose=True
    )

    print("INT8 mAP50:", val_results.box.map50)
    print("INT8 mAP50-95:", val_results.box.map)
    print("每类AP50:", val_results.box.maps)