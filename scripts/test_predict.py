from ultralytics import YOLO
import torch
import os

# 打印环境信息（直观调试）
print("Torch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())  # True表示GPU版成功
print("GPU名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无GPU")

# 加载模型
model = YOLO("weights/yolov8s.pt")  # 指定你的路径，第二次就不会再下载  
                                    # 或 "yolov8n.pt" 如果想更大模型
# 自定义本地图片路径
local_path = "bus.jpg" 

if not os.path.exists(local_path):
    print(f"错误：图片不存在 {local_path}，请手动下载放置")
else:
    print(f"使用本地图片: {local_path}")

# 预测

    relative_project = "runs"  # 相对路径，便于管理
    absolute_project = os.path.abspath(relative_project)
    print("使用project绝对路径:", absolute_project)

    results = model.predict(
        source=local_path,
        save=True,
        imgsz=640,
        project=absolute_project,
        name="test",
        exist_ok=True
    )
# 打印信息（直观输出）
for result in results:
    print("检测结果保存路径:", result.save_dir)  # 结果图片保存到指定project/name
    print("检测到类别:", result.names)  # 所有类别名
    print("检测框数量:", len(result.boxes))  # 本图检测到多少框
    print("置信度示例:", result.boxes.conf.tolist()[:5])  # 前5个置信度
    print("保存路径:", result.save_dir)  # 结果图片保存位置

print("测试完成！结果在 runs/test/")