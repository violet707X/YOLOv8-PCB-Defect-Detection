from ultralytics import YOLO
import torch
import os

if __name__ == '__main__':
    # 对于 2GB 显存的 GPU（MX450），使用更小的模型
    # 选项1：使用 yolov8n（nano 模型，参数量小，显存占用少，推荐）
    model = YOLO("weights/yolov8n.pt")
    
    # 选项2：使用 yolov8s（需要更小的 batch，batch=4 可能还不够）
    # model = YOLO("weights/yolov8s.pt")

    relative_project = "runs/val"  # 相对路径，便于管理
    absolute_project = os.path.abspath(relative_project)
    print("使用project绝对路径:", absolute_project)

    results = model.val(
        data="./data/data.yaml",       # 确认路径正确
        imgsz=640,
        batch=8,                       # yolov8n 可以用更大的 batch（2GB 显存建议 8-16）
        device=0 if torch.cuda.is_available() else "cpu",
        project=absolute_project,
        name="baseline",
        exist_ok=True,
        plots=True,                     # 生成PR曲线、confusion matrix图
        workers=0                        # 减少数据加载进程（Windows 上避免多进程问题）
    )

    print("基线 mAP50-95:", results.box.map)
    print("mAP50:", results.box.map50)
    print("每类mAP:", results.box.maps)
    print("结果保存路径:", results.save_dir)  # runs/val/baseline/ 有图表