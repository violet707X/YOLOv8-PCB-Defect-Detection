from ultralytics import YOLO
import os

if __name__ == '__main__':
    model = YOLO("runs/train/pcb_finetune_v1/weights/best.pt")

    # 计算绝对project路径
    relative_project_pre = "runs/predict"  # 相对路径，便于管理
    absolute_project_pre = os.path.abspath(relative_project_pre)
    print("使用project绝对路径:", absolute_project_pre)

    # 关键：如果文件夹不存在，自动创建（exist_ok=True避免报错）
    os.makedirs(absolute_project_pre, exist_ok=True)

    predict_results = model.predict(
        source="data/test_images/images", 
        save=True,
        save_txt=True,
        save_conf=True,
        imgsz=640,
        conf=0.25,               # 置信阈值，可调
        project=absolute_project_pre,
        name="pcb_defect_demo",
        exist_ok=True
    )
    print("demo标注图保存路径:", predict_results[0].save_dir)