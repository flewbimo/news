import sys
import os
from dotenv import load_dotenv
from crew import *
import pandas as pd
import json


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

#训练集预处理
def preprocess_training_data():
    try:
        # 获取当前文件所在目录的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建数据文件的完整路径（数据文件在src/sets目录下）
        data_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'src', 'sets', 'snopeswithsum.csv')
        
        # 读取CSV文件，只读取前50行
        df = pd.read_csv(data_path, encoding='utf-8', nrows=50)
        
        return df
        
    except FileNotFoundError:
        print(f"错误：找不到数据文件 {data_path}")
        return None
    except pd.errors.EmptyDataError:
        print("错误：数据文件为空")
        return None
    except Exception as e:
        print(f"读取数据时发生错误：{str(e)}")
        return None

def run():
    df = preprocess_training_data()
    if df is None:
        return
        
    # 创建结果记录列表
    results = []
    total_count = 0
    correct_count = 0
    
    # 遍历数据集的每一行
    for index, row in df.iterrows():
        # 检查rate是否为"True"或"False"
        if row['rate'] in ["True", "False"]:
            inputs = {
                'news': row['question'],
            }
            print(f"处理第 {index + 1} 条新闻...")
            # 获取模型预测结果
            prediction = disentangling_crew.kickoff(inputs)
            
            # 记录结果
            result = {
                'index': index,
                'news': row['question'],
                'true_label': row['rate'],
                'prediction': prediction.raw,
                'is_correct': prediction.raw.strip().lower() == row['rate'].lower()
            }
            results.append(result)
            
            # 统计正确数量
            if result['is_correct']:
                correct_count += 1
            total_count += 1
            
            # 实时打印准确率
            accuracy = (correct_count / total_count) * 100
            print(f"当前准确率: {accuracy:.2f}%")
    
    # # 将结果保存到CSV文件
    # results_df = pd.DataFrame(results)
    # output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'results.csv')
    # results_df.to_csv(output_path, index=False, encoding='utf-8')
    # 将结果保存到 JSON 文件
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'results.json')
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    # 打印最终统计结果
    print("\n最终统计结果:")
    print(f"总样本数: {total_count}")
    print(f"正确预测数: {correct_count}")
    print(f"最终准确率: {(correct_count / total_count * 100):.2f}%")
    print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    run()
