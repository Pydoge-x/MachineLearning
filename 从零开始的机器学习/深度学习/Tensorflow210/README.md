# 深度学习配置-Tensorflow2.10

##  环境配置
- 项目开始之前，请仔细检查相关环境以及依赖是否安装完成
- 建议使用虚拟环境进行项目开发，避免环境冲突
- 以下为需要的环境：
  1. Anaconda3
  2. Python 3.8-3.10
  3. CUDA 11.2 (如果使用GPU)
  4. cuDNN 8.1 (如果使用GPU)
  5. TensorFlow 2.10.0
  6. Jupyter Notebook
  7. 其他依赖包（见requirements.txt）

## 安装相关依赖环境步骤

1. 创建并激活虚拟环境（可选，但推荐）：
   ```bash
   conda create -n tf_env python=3.9
   conda activate tf_env
   ```
2. 安装CUDA和cuDNN（如果使用GPU）：
   ```bash
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 
    ```
   - 请确保CUDA和cuDNN版本与TensorFlow 2.10.0兼容。
3. 安装TensorFlow 2.10.0：
   ```bash
   pip install tensorflow==2.10.0
   ```
4. 安装项目需要的其他依赖库
    ```bash
    pip install -r requirements.txt
   ```

## 环境配置完成后，可以进行以下检查：
1. 检查TensorFlow安装是否成功：
   ```python
   import tensorflow as tf
   print(tf.__version__)
   ```
2. 检查GPU是否可用（如果使用GPU）：
   ```python
   import tensorflow as tf
   print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    ```
