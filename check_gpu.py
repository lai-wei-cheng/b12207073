import tensorflow as tf

print("=== TensorFlow GPU 檢查 ===\n")
print(f"TensorFlow 版本: {tf.__version__}")
print(f"可用 GPU 數: {len(tf.config.list_physical_devices('GPU'))}")
print(f"GPU 設備列表:")
for gpu in tf.config.list_physical_devices('GPU'):
    print(f"  - {gpu}")

print(f"\nCPU 設備列表:")
for cpu in tf.config.list_physical_devices('CPU'):
    print(f"  - {cpu}")

# 檢查 CUDA 可用性
try:
    import tensorflow.python.client.device_lib as device_lib
    devices = device_lib.list_local_devices()
    print("\n詳細設備資訊:")
    for device in devices:
        print(f"  {device.device_type}: {device.name}")
except:
    pass

# 簡單測試
print("\n=== 簡單計算測試 ===")
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 2.0], [0.0, 1.0]])
c = tf.matmul(a, b)
print(f"矩陣乘法結果:\n{c}")
print(f"計算使用設備: {c.device}")
