import json
import os
from deep_translator import GoogleTranslator

# 输入文件夹路径
input_folder = r'f:\PythonProject\MachineLearning\从零开始的机器学习\深度学习\pytorch-deep-learning-main'
# 输出文件夹路径
output_folder = r'f:\PythonProject\MachineLearning\从零开始的机器学习\深度学习\Pytorch29cu126'

# 要处理的文件列表
files_to_process = [
    '00_pytorch_fundamentals.ipynb',
    '01_pytorch_workflow.ipynb',
    '02_pytorch_classification.ipynb',
    '03_pytorch_computer_vision.ipynb',
    '04_pytorch_custom_datasets.ipynb',
    '06_pytorch_transfer_learning.ipynb',
    '07_pytorch_experiment_tracking.ipynb',
    '08_pytorch_paper_replicating.ipynb',
    '09_pytorch_model_deployment.ipynb'
]

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 初始化翻译器
translator = GoogleTranslator(source='auto', target='zh-CN')

# 翻译函数
def translate_text(text):
    try:
        # 翻译为中文
        translated = translator.translate(text)
        return translated
    except Exception as e:
        print(f"翻译出错: {e}")
        return text

# 处理每个文件
for file_name in files_to_process:
    input_file_path = os.path.join(input_folder, file_name)
    output_file_path = os.path.join(output_folder, file_name)
    
    print(f"正在处理: {file_name}")
    
    # 读取文件
    with open(input_file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # 处理每个单元格
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            # 翻译markdown单元格中的内容
            translated_source = []
            for line in cell['source']:
                # 跳过链接和代码块
                if line.strip().startswith('!') or line.strip().startswith('```') or line.strip().startswith('<a href'):
                    translated_source.append(line)
                else:
                    translated_line = translate_text(line)
                    translated_source.append(translated_line)
            cell['source'] = translated_source
    
    # 保存翻译后的文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    
    print(f"已保存到: {output_file_path}")

print("所有文件处理完成！")