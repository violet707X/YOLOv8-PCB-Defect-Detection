import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image
import tempfile
import shutil

# 页面配置
st.set_page_config(page_title="PCB缺陷检测系统", layout="wide", page_icon="🔍")

# 加载模型
@st.cache_resource
def load_model():
    model = YOLO("runs/train/pcb_finetune_v1/weights/best.pt")  # 修改为您的路径
    return model

model = load_model()

# 类别名称
class_names = ['copper', 'mousebite', 'open', 'pin-hole', 'short', 'spur']

# 侧边栏
st.sidebar.title("📊 缺陷统计摘要")
st.sidebar.markdown("上传后，这里显示各类缺陷计数。")

# 主界面
st.title("🔍 工业PCB缺陷实时检测系统")
st.markdown("""
支持图像/视频上传，实时检测6类缺陷（copper, mousebite, open, pin-hole, short, spur）。
- 图像：即时标注。
- 视频：逐帧处理+输出标注视频。
""")

uploaded_file = st.file_uploader("上传图像（jpg/jpeg/png）或视频（mp4）", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_type = uploaded_file.type

    is_image = file_type.startswith('image')
    is_video = file_type.startswith('video')

    if is_image:
        # 图像：直接bytes推理（无需保存文件，高效）
        bytes_data = uploaded_file.getvalue()
        image = Image.open(uploaded_file)
        st.image(image, caption="上传图像", width="stretch")

        # bytes转numpy for YOLO
        img_cv = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("正在检测缺陷..."):
            results = model(img_cv, imgsz=640, conf=0.5)[0]  # 直接推理numpy图像

        # 绘制标注
        annotated_image = results.plot()
        annotated_pil = Image.fromarray(annotated_image[..., ::-1])  # BGR to RGB
        st.image(annotated_pil, caption="检测结果", width="stretch")

        # 统计
        if results.boxes is not None:
            cls_counts = np.bincount(results.boxes.cls.cpu().numpy().astype(int), minlength=len(class_names))
            defect_summary = {class_names[i]: int(count) for i, count in enumerate(cls_counts) if count > 0}
        else:
            defect_summary = {}

        st.sidebar.markdown("### 当前图像缺陷计数")
        if defect_summary:
            for defect, count in defect_summary.items():
                st.sidebar.markdown(f"- **{defect}**: {count} 个")
        else:
            st.sidebar.markdown("无缺陷检测到")

        st.success("图像检测完成！")

    elif is_video:
        # 视频：保存带扩展名临时文件
        st.video(uploaded_file)

        # 保存临时视频（带原扩展名）
        suffix = os.path.splitext(file_name)[1]
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded_file.getvalue())
        tfile.close()
        video_path = tfile.name

        output_path = "runs/predict_video/output_video.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with st.spinner("正在处理视频（可能需几分钟）..."):
            results = model.predict(
                source=video_path,
                save=True,
                project="runs/predict_video",
                name="output_video",
                exist_ok=True,
                imgsz=640,
                conf=0.5,
                vid_stride=1
            )

        st.video(output_path)

        # 视频总统计
        total_counts = np.zeros(len(class_names))
        for result in results:
            if result.boxes is not None:
                cls = result.boxes.cls.cpu().numpy().astype(int)
                total_counts += np.bincount(cls, minlength=len(class_names))

        defect_summary = {class_names[i]: int(count) for i, count in enumerate(total_counts) if count > 0}

        st.sidebar.markdown("### 视频总缺陷计数（所有帧）")
        if defect_summary:
            for defect, count in defect_summary.items():
                st.sidebar.markdown(f"- **{defect}**: {count} 个")
        else:
            st.sidebar.markdown("无缺陷检测到")

        # 清理临时文件
        os.unlink(video_path)

        st.success("视频检测完成！")

else:
    st.info("请上传图像或视频开始检测。")

st.markdown("---")
st.markdown("**模型精度**：mAP50 = 0.9826 | **支持格式**：jpg/jpeg/png/mp4")