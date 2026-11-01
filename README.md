深度学习框架目标检测算法yolov8模型训练室内积水数据集（ 浸水数据集）检测识别室内漏水检测数据集的积水监测
@[toc]
以下文字及代码仅供参考。同学，参考学习，仅供参考
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/39dd4ff8c29746c3b19655458c3d6b6f.png)

室内积水数据集 浸水数据集（数量2300标注）
项目背景：
室内漏水隐患频发（如管道破裂、设备故障等），传统人工巡检效率低、成本高。AI智能摄像头搭载深度学习算法，可实时捕捉水渍、积水等异常画面，毫秒级触发声光报警并推送至管理端，大幅降低财产损失风险。
数据量：2334（带标注）![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/326ee71f306f4261b393f63948b1d7ee.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aaed7edc8b7a49edaddb32bd3d0259d7.png)
 **浸水检测数据集**，训练 **深度学习模型实时检测漏水、积水等隐患**。使用 **YOLOv8 框架** 实现目标检测任务。

---![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/66386734bda6449dbf3dbaed94a85805.png)


## 📊 一、数据集概览

| 属性 | 内容 |
|------|------|
| 数据集名称 | 浸水数据集（Water Detection Dataset） |
| 数据量 | 2334 张图像（已标注） |
| 标注方式 | YOLO格式（每个图像对应 `.txt` 标签文件） |
| 目标类别 | 默认为 1 类：water（水渍/积水） |
| 应用场景 | 室内漏水隐患检测、AI摄像头报警系统 |

---

## 🧱 二、数据准备与预处理

### ✅ 1. 数据结构要求

同学们，确保同学你的数据集符合以下目录结构：

```
water_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
```

- 每张图片 `xxx.jpg` 对应一个标签文件 `xxx.txt`，内容如下：
  
  ```
  0 x_center y_center width height
  ```

> 其中 `0` 表示类别 ID（只有一个类“water”）

---

### ✅ 2. 划分训练集 / 验证集 / 测试集（建议比例 8:1:1）

```bash
Total images = 2334
Train: ~1867
Val: ~234
Test: ~233
```

同学你可以使用 Python 脚本进行自动划分（参考下方代码节）。

---

### ✅ 3. 创建 `data.yaml` 文件

```yaml
train: water_dataset/images/train/
val: water_dataset/images/val/
test: water_dataset/images/test/

nc: 1
names: ['water']
```

保存为 `data.yaml`，用于 YOLOv8 训练配置。

---

## 🛠️ 三、环境搭建

### 安装 Ultralytics YOLOv8 环境：

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -r requirements.txt
```

---

## 🚀 四、模型训练

### 使用 YOLOv8 命令行训练：

```bash
yolo train data=data.yaml model=yolov8s.pt epochs=100 batch=16 imgsz=640
```

#### 参数说明：

| 参数 | 含义 |
|------|------|
| `data` | 数据配置文件路径 |
| `model` | 使用的预训练模型权重（如 yolov8n/s/m/l/x） |
| `epochs` | 总训练轮次 |
| `batch` | 批大小（根据显存调整） |
| `imgsz` | 输入图像尺寸（默认640x640） |

---

## 🔍 五、推理与部署

### ✅ 单张图像推理：

```python
from ultralytics import YOLO

# 加载最佳模型
model = YOLO('runs/train/exp/weights/best.pt')

# 推理单张图像
results = model('test_images/water_001.jpg')

# 显示结果
for r in results:
    print(r.boxes)  # 输出预测框信息
    annotated_img = r.plot()
    cv2.imshow("Detection", annotated_img)
    cv2.waitKey(0)
```

### ✅ 批量图像推理：

```python
results = model(['image1.jpg', 'image2.jpg'])
```

要通过训练出的 **YOLOv8 权重文件**（如 `best.pt`）来构建一个完整的 **深度学习积水检测识别系统**，你可以按照以下步骤进行开发和部署。该系统可用于实时监控、视频分析或图像批量处理，适用于室内漏水检测、AI摄像头报警等场景。

---

## ✅ 一、系统目标

构建一个基于 YOLOv8 的 **积水检测识别系统**，具备以下功能：

- 图像/视频中积水区域的自动检测
- 检测框可视化展示
- 可扩展为：
  - 实时视频流检测（摄像头）
  - 批量图像检测
  - 报警触发机制（声光 / 邮件 / 推送通知）
  - Web API 接口服务

---

## 🧱 二、系统架构图（文字版）

```
[输入源] → [YOLOv8模型加载权重 + 推理] → [结果输出]
   ↑               ↓
数据源：    显示/保存/报警/API返回
图像/视频/摄像头
```

---

## 💻 三、代码实现（YOLOv8 + OpenCV）

### 1. 安装依赖

```bash
pip install ultralytics opencv-python flask numpy
```

> 若你已使用过 YOLOv8 的训练环境，可直接复用。

---

### 2. 加载模型 & 单张图片推理

#### 文件名：`detect_water.py`

```python
from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO('runs/train/exp/weights/best.pt')  # 替换为你自己的路径

# 图像路径
image_path = 'test_images/water_001.jpg'

# 推理
results = model(image_path)

# 获取绘制后的图像
annotated_image = results[0].plot()

# 显示结果
cv2.imshow("Water Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

📌 输出说明：
- `results[0].boxes` 可获取预测框坐标、类别、置信度等信息。
- `results[0].plot()` 返回带标注框的图像。

---

### 3. 视频/摄像头实时检测

```python
def detect_video(source=0):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("Real-time Water Detection", annotated_frame)
        if cv2.waitKey(1) == 27:  # ESC键退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_video()  # source=0 表示摄像头，source='video.mp4' 表示视频文件
```

---

### 4. 批量图像检测

```python
import os

image_folder = 'test_images/'
output_folder = 'output/'

os.makedirs(output_folder, exist_ok=True)

for img_file in os.listdir(image_folder):
    if img_file.endswith('.jpg') or img_file.endswith('.png'):
        img_path = os.path.join(image_folder, img_file)
        results = model(img_path)
        annotated_img = results[0].plot()
        output_path = os.path.join(output_folder, img_file)
        cv2.imwrite(output_path, annotated_img)

print("✅ 批量检测完成，结果已保存至 output/")
```

---

## 🌐 四、构建 Web API（Flask）

将系统封装为远程服务接口，可使用 Flask 构建 RESTful API。

#### 文件名：`app.py`

```python
from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('runs/train/exp/weights/best.pt')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json.get("image")  # base64编码的图像
    img_bytes = base64.b64decode(data)
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_np, flags=1)

    results = model(img)
    annotated_img = results[0].plot()

    _, buffer = cv2.imencode('.jpg', annotated_img)
    response_data = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({"result_image": response_data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

📌 请求格式（JSON）：

```json
{
  "image": "base64encoded_string_of_jpeg"
}
```

---

## 🔔 五、报警机制

可添加以下逻辑，在检测到积水时触发报警：

```python
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if cls == 0 and conf > 0.5:  # 如果是 water 类别且置信度高
            print("⚠️ 检测到积水！触发报警！")
            # 这里可以调用声光报警、发送邮件、微信推送等
```

---


---
## 📈 六、性能评估

使用验证命令评估 mAP、Recall、Precision 等指标：

```bash
yolo val data=data.yaml model=runs/train/exp/weights/best.pt imgsz=640
```

输出包括：

- `mAP@0.5`
- `mAP@0.5:0.95`
- Precision / Recall 曲线
- F1-score

---

## 📦 七、模型导出

将模型导出为 ONNX、TensorRT 或 OpenVINO 等格式以便部署：

```bash
yolo export model=runs/train/exp/weights/best.pt format=onnx
```

支持格式：`onnx`, `engine`, `coreml`, `tflite`, `pb`, `pt` 等


## 📌 十、总结

| 步骤 | 工具/方法 |
|------|-----------|
| 数据整理 | 统一格式、划分训练集 |
| 数据配置 | `data.yaml` |
| 环境搭建 | Ultralytics YOLOv8 |
| 模型训练 | `yolo train ...` |
| 推理测试 | `model.predict()` |
| 性能评估 | `yolo val ...` |
| 模型导出 | `yolo export ...` |

---


