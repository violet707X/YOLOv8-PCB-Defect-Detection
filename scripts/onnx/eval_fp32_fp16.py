from ultralytics import YOLO
import os

if __name__ == '__main__':
    onnx_dir = "exports/onnx"
    data_yaml = "data/data.yaml"

    models = {
        "PyTorch原生": "runs/train/pcb_finetune_v1/weights/best.pt",
        "FP32 ONNX": os.path.join(onnx_dir, "yolov8n_pcb_fp32.onnx"),
        "FP16 ONNX": os.path.join(onnx_dir, "yolov8n_pcb_fp16.onnx")
    }

    results = {}
    for name, path in models.items():
        print(f"\n评估 {name}: {path}")
        model = YOLO(path, task='detect')   # 显式指定 task='detect'，防止 YOLO 猜错任务类型
        val_results = model.val(
            data=data_yaml,
            imgsz=640,
            batch=4,                # MX450安全
            device=0,               # GPU
            plots=False,
            verbose=True
        )

        results[name] = {
            "mAP50-95": val_results.box.map,
            "mAP50": val_results.box.map50,
            "每类AP50": val_results.box.maps
        }

        print(f"{name} mAP50: {val_results.box.map50:.4f}")

    # 表格
    print("\nFP32/FP16精度对比总结：")
    print("| 模型       | mAP50   | mAP50-95 | 下降 (vs PyTorch) |")
    print("|------------|---------|----------|-------------------|")
    pytorch_map50 = results["PyTorch原生"]["mAP50"]
    for name, res in results.items():
        decline = "" if name == "PyTorch原生" else f"{pytorch_map50 - res['mAP50']:.4f}"
        print(f"| {name.ljust(10)} | {res['mAP50']:.4f} | {res['mAP50-95']:.4f} | {decline}      |")