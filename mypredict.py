from ultralytics import YOLO

model = YOLO(r"C:\homework\ultralytics\models\yolo26n-face.pt")
model.predict(
    source=r"C:\homework\ultralytics\mytest.jpg",
    save=True,
    show=False,
    save_txt=False,
    task='detect',
    classes=0,  # 仅检测类别0（人脸）
)
# model.predict(
#     source="",            # (str, 可选) 图像或视频的源目录
#     imgsz=640,            # (int | list) 预测的输入图像大小为整数或[w,h]列表
#     conf=0.25,            # (float) 最小置信度阈值
#     iou=0.7,              # (float) NMS的交并比(IoU)阈值
#     device=None,          # (int | str | list, 可选) 运行设备，例如cuda设备=0或device=0,1,2,3或device=cpu
#     batch=1,              # (int) 批次大小
#     half=False,           # (bool) 使用FP16半精度推理
#     max_det=300,          # (int) 限制每张图像的最大检测数量。在密集场景中很有用，可以防止过多的检测
#     vid_stride=1,         # (int) 视频帧率步长
#     stream_buffer=False,  # (bool) 缓冲所有流帧(True)或返回最新帧(False)
#     visualize=False,      # (bool) 可视化模型特征
#     augment=False,        # (bool) 对预测源应用图像增强
#     agnostic_nms=False,   # (bool) 类别不可知的NMS
#     classes=None,         # (int | list[int], 可选) 按类别过滤结果，例如classes=0或classes=[0,2,3]
#     retina_masks=False,   # (bool) 使用高分辨率分割掩码
#     embed=None,           # (list[int], 可选) 从给定层返回特征向量/嵌入
#     show=False,           # (bool) 如果环境允许，显示预测的图像和视频
#     save=True,            # (bool) 保存预测结果
#     save_frames=False,    # (bool) 保存预测的单个视频帧
#     save_txt=False,       # (bool) 将结果保存为.txt文件
#     save_conf=False,      # (bool) 保存带置信度分数的结果
#     save_crop=False,      # (bool) 保存带结果的裁剪图像
#     stream=False,         # (bool) 通过返回生成器来处理长视频或大量图像，减少内存使用
#     verbose=True,         # (bool) 在终端中启用/禁用详细推理日志
# )