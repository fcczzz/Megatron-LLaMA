#在此之前先把
from datasets import load_dataset

train_data = load_dataset('./code-search-net-python', split='train')
train_data.to_json("origin.json", lines=True)

import json

# 定义输入和输出文件路径
input_file_path = '/hy-tmp/dataset/origin.json'
output_file_path = '/hy-tmp/dataset/code-search-net.json'

# 打开输入文件并逐行读取数据，处理后写入输出文件
with open(input_file_path, 'r', encoding='utf-8') as infile, \
     open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        try:
            data = json.loads(line)
            # 拼接 docstring 和 code 字段
            text = f"{data.get('docstring', '')} {data.get('code', '')}".strip()
            # 创建新的 JSON 对象，只包含 text 字段
            new_data = {'text': text}
            # 将新的 JSON 对象写入输出文件
            outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

print(f"处理后的数据已保存到 {output_file_path}")