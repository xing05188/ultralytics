from ultralytics import settings

# 禁用Comet集成
settings.update({"comet": False})

# 启用Comet集成
# settings.update({"comet": True})

# 打印当前设置以确认更改
print("当前设置:")
print(settings)