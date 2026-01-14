from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 直接加载最佳模型路径（修复旧语法）
    best_model_path = "runs/train/pcb_finetune_v1/weights/best.pt"  # 确认路径
    model = YOLO(best_model_path)

    # 计算绝对project路径
    relative_project_val = "runs/val"  # 相对路径，便于管理
    absolute_project_val = os.path.abspath(relative_project_val)
    print("使用project绝对路径:", absolute_project_val)

    # 关键：如果文件夹不存在，自动创建（exist_ok=True避免报错）
    os.makedirs(absolute_project_val, exist_ok=True)

    # 1. val生成所有图表 + 最终指标
    val_results = model.val(
        data="data/data.yaml",
        imgsz=640,
        batch=16,
        device=0,                # GPU
        plots=True,              # 生成confusion_matrix/PR_curve/F1_curve等
        save=True,
        project=absolute_project_val,
        name="final_eval",
        exist_ok=True
    )

    # 打印最终指标
    print("最终 mAP50-95:", val_results.box.map)
    print("mAP50:", val_results.box.map50)
    print("每类mAP50-95:", val_results.box.maps)  # 6个类别mAP
    print("图表保存路径:", val_results.save_dir)