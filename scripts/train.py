from ultralytics import YOLO
import torch
import os

if __name__ == '__main__':  # 关键：Windows必加这一行保护！
    # 用本地权重加载（解决下载损坏坑）
    # model = YOLO("weights/yolov8n.pt")  # 或 "weights/yolov8s.pt" 如果想精度更高（稍慢）
    
    # 不加载预训练，直接resume上一次实验
    model = YOLO("runs/train/pcb_finetune_v1/weights/last.pt")  # resume用last，或 "weights/yolov8n.pt" 新训练   
    
    # fine-tune

    relative_project = "runs/train"  # 相对路径，便于管理
    absolute_project = os.path.abspath(relative_project)
    print("使用project绝对路径:", absolute_project)

    results = model.train(
        resume=True,            # 关键：自动resume最新runs/train/下的实验（last.pt）
        data="data/data.yaml",  # 注意你的路径是 data/data.yaml（从结构看）
        epochs=100,
        imgsz=640,
        batch=4,               # MX450安全（如果OOM降到8）
        patience=20,
        amp=False,
        device=0 if torch.cuda.is_available() else "cpu",
        project=absolute_project,   # 保存到runs/train/
        name="pcb_finetune_v1", # 实验名（如果已存在，exist_ok=True覆盖）
        exist_ok=True,
        plots=True,
        verbose=True,
        save=True,                      # 保存weights
        save_period=10,                 # 关键：每10 epoch保存一次权重（last.pt + epochX.pt）
        save_json=True,                 # 保存val JSON（COCO格式指标）
        save_txt=True,                  # 保存预测txt（可选）
        save_crop=True,                 # 保存crop检测框图（可视化加分）
        save_conf=True                  # 保存置信度
    )

    # 训练后自动val最佳模型
    best_model = YOLO(results.best)
    val_results = best_model.val(data="data/data.yaml", plots=True, save=True)

    print("Fine-tune完成！")
    print("最佳模型路径:", results.best)
    print("mAP50-95:", val_results.box.map)
    print("mAP50:", val_results.box.map50)
    print("每类mAP:", val_results.box.maps)