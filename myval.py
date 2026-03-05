from ultralytics import  YOLO

if __name__ == '__main__':
    model = YOLO(r"C:\homework\ultralytics\runs\detect\train5\weights\best.pt")
    model.val(
        data=r"widerface.yaml",
        imgsz=640,
        split="val",
    )
    # model.val(
    #     data="coco8.yaml",  # (str) 指定数据集配置文件的路径（例如coco8.yaml）
    #     imgsz=640,          # (int) 定义输入图像的大小。所有图像在处理前都会调整为此尺寸
    #     batch=16,           # (int) 设置每批次的图像数量。使用-1表示AutoBatch，它会根据GPU内存可用性自动调整
    #     save_json=False,    # (bool) 如果为True，则将结果保存到JSON文件，以便进一步分析或与其他工具集成
    #     save_hybrid=False,  # (bool) 如果为True，则保存标签的混合版本，结合原始注释和额外的模型预测
    #     conf=0.001,         # (float) 设置检测的最小置信度阈值。低于此阈值的检测将被丢弃
    #     iou=0.6,            # (float) 设置非极大值抑制(NMS)的交并比(IoU)阈值。有助于减少重复检测
    #     max_det=300,        # (int) 限制每张图像的最大检测数量。在密集场景中很有用，可以防止过多的检测
    #     half=True,          # (bool) 启用半精度(FP16)计算，减少内存使用，并可能在最小影响精度的情况下提高速度
    #     device=None,        # (str | int) 指定验证设备（cpu、cuda:0等）。允许灵活使用CPU或GPU资源
    #     dnn=False,          # (bool) 如果为True，则使用OpenCV DNN模块进行ONNX模型推理，提供PyTorch推理方法的替代方案
    #     plots=False,        # (bool) 设置为True时，生成并保存预测与真实值的图表，用于可视化评估模型性能
    #     rect=False,         # (bool) 如果为True，则使用矩形推理进行批处理，减少填充，并可能提高速度和效率
    #     split="val",        # (str) 确定用于验证的数据集分割（val、test或train）
    # )