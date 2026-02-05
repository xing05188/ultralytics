from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"C:\homework\ultralytics\runs\yolo11n\weights\last.pt")  # 加载预训练的YOLOv8n模型
    model.train(
        data=r"widerface.yaml",
        epochs=200,
        imgsz=640,
        batch=4,
        cache=False,
        workers=4,
        device=0,
        project="runs",
        name="yolo11n",
        resume=True,
    )
    # model.train(
    #     data="coco8.yaml",           # (str, 可选) 数据文件路径，例如coco8.yaml
    #     epochs=100,                  # (int) 训练的轮数
    #     time=None,                   # (float, 可选) 训练的小时数，如果提供则覆盖epochs参数
    #     patience=100,                # (int) 等待没有明显改善的轮数，用于提前停止训练
    #     batch=16,                    # (int) 每批次的图像数量（-1表示自动批次）
    #     imgsz=640,                   # (int | list) 输入图像大小，训练和验证模式为整数，预测和导出模式为[w,h]列表
    #     save=True,                   # (bool) 保存训练检查点和预测结果
    #     save_period=-1,              # (int) 每x轮保存一次检查点（小于1时禁用）
    #     cache=False,                 # (bool) True/ram, disk或False。使用缓存进行数据加载
    #     device=None,                 # (int | str | list, 可选) 运行设备，例如cuda设备=0或device=0,1,2,3或device=cpu
    #     workers=8,                   # (int) 数据加载的工作线程数（如果使用DDP，则每个RANK的工作线程数）
    #     project="",                # (str, 可选) 项目名称
    #     name="",                   # (str, 可选) 实验名称，结果保存到'project/name'目录
    #     exist_ok=False,              # (bool) 是否覆盖现有实验
    #     val=True,                    # (bool) 训练期间验证/测试
    #     pretrained=True,             # (bool | str) 是否使用预训练模型(bool)或从模型加载权重的模型(str)
    #     optimizer="SGD",             # (str) 使用的优化器，选择=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    #     verbose=True,                # (bool) 是否打印详细输出
    #     seed=0,                      # (int) 用于可重复性的随机种子
    #     deterministic=True,          # (bool) 是否启用确定性模式
    #     single_cls=False,            # (bool) 将多类数据作为单类训练
    #     rect=False,                  # (bool) 如果mode='train'则使用矩形训练，如果mode='val'则使用矩形验证
    #     cos_lr=False,                # (bool) 使用余弦学习率调度器
    #     close_mosaic=10,             # (int) 在最后几轮禁用马赛克增强（0表示禁用）
    #     resume=False,                # (bool) 从最后一个检查点恢复训练
    #     amp=True,                    # (bool) 自动混合精度(AMP)训练，选择=[True, False]，True运行AMP检查
    #     fraction=1.0,                # (float) 训练使用的数据集比例（默认为1.0，训练集中的所有图像）
    #     profile=False,               # (bool) 在训练期间为记录器分析ONNX和TensorRT速度
    #     freeze=None,                 # (int | list, 可选) 冻结前n层，或冻结训练期间的层索引列表
    #     multi_scale=False,           # (bool) 训练期间是否使用多尺度
    #     plots=True,                  # (bool) 训练/验证期间保存图表和图像
    #     # 分割
    #     overlap_mask=True,           # (bool) 训练期间掩码应重叠（仅分割训练）
    #     mask_ratio=4,                # (int) 掩码下采样比例（仅分割训练）
    #     # 分类
    #     dropout=0.0,                 # (float) 使用dropout正则化（仅分类训练）
    #     # 超参数
    #     lr0=0.01,                    # (float) 初始学习率（例如SGD=1E-2, Adam=1E-3）
    #     lrf=0.01,                    # (float) 最终学习率（lr0 * lrf）
    #     momentum=0.937,              # (float) SGD动量/Adam beta1
    #     weight_decay=0.0005,         # (float) 优化器权重衰减5e-4
    #     warmup_epochs=3.0,           # (float) 预热轮数（分数可以）
    #     warmup_momentum=0.8,         # (float) 预热初始动量
    #     warmup_bias_lr=0.1,          # (float) 预热初始偏置学习率
    #     box=7.5,                     # (float) 边界框损失增益
    #     cls=0.5,                     # (float) 类别损失增益（与像素成比例）
    #     dfl=1.5,                     # (float) dfl损失增益
    #     pose=12.0,                   # (float) 姿态损失增益
    #     kobj=1.0,                    # (float) 关键点对象损失增益
    #     label_smoothing=0.0,         # (float) 标签平滑（分数）
    #     nbs=64,                      # (int) 标准批次大小
    #     hsv_h=0.015,                 # (float) 图像HSV-色调增强（分数）
    #     hsv_s=0.7,                   # (float) 图像HSV-饱和度增强（分数）
    #     hsv_v=0.4,                   # (float) 图像HSV-明度增强（分数）
    #     degrees=0.0,                 # (float) 图像旋转（+/- 度）
    #     translate=0.1,               # (float) 图像平移（+/- 分数）
    #     scale=0.5,                   # (float) 图像缩放（+/- 增益）
    #     shear=0.0,                   # (float) 图像剪切（+/- 度）
    #     perspective=0.0,             # (float) 图像透视（+/- 分数），范围0-0.001
    #     flipud=0.0,                  # (float) 图像上下翻转（概率）
    #     fliplr=0.5,                  # (float) 图像左右翻转（概率）
    #     bgr=0.0,                     # (float) 图像通道BGR（概率）
    #     mosaic=1.0,                  # (float) 图像马赛克（概率）
    #     mixup=0.0,                   # (float) 图像混合（概率）
    #     copy_paste=0.0,              # (float) 分段复制粘贴（概率）
    #     auto_augment="randaugment",  # (str) 分类的自动增强策略（randaugment, autoaugment, augmix）
    #     erasing=0.4,                 # (float) 分类训练期间随机擦除的概率[0-0.9]，0表示不擦除，必须小于1.0
    #     crop_fraction=1.0,           # (float) 分类的图像裁剪比例[0.1-1]，1.0表示不裁剪，必须大于0
    # )