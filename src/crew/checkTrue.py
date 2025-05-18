import sys
import os
from dotenv import load_dotenv
from crew import *
import pandas as pd
import json
import time
import litellm
import random
# 禁用 oneDNN 优化
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

litellm.set_verbose = True
experience_library = "暂无"

def call_api_with_retry(api_func, inputs, max_retries=5, initial_delay=10):
    """
    带重试机制的API调用函数
    
    Args:
        api_func: API调用函数
        inputs: 输入参数
        max_retries: 最大重试次数
        initial_delay: 初始等待时间（秒）
    
    Returns:
        API调用结果
    """
    for attempt in range(max_retries):
        try:
            response = api_func(inputs)
            if not response or not response.raw:
                raise ValueError("API返回空响应")
            return response.raw
        except litellm.RateLimitError:
            # 使用指数退避策略
            delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"触发限流，等待{delay:.1f}秒后重试... (尝试 {attempt + 1}/{max_retries})")
            time.sleep(delay)
        except Exception as e:
            print(f"发生错误: {str(e)} (尝试 {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                raise
            time.sleep(5)
    raise Exception(f"在{max_retries}次尝试后仍然失败")

#训练集预处理
def preprocess_training_data():
    try:
        # 获取当前文件所在目录的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建数据文件的完整路径（数据文件在src/sets目录下）
        data_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'src', 'sets', 'snopeswithsum.csv')
        
        # 读取CSV文件
        df = pd.read_csv(data_path, encoding='utf-8')
        
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
    global experience_library
    df = preprocess_training_data()
    if df is None:
        return
        
    # 创建训练和验证结果记录列表
    training_results = []
    validation_results = []
    training_total = 0
    training_correct = 0
    validation_total = 0
    validation_correct = 0
    
    # 遍历数据集的每一行
    for index, row in df.iterrows():
        # 检查rate是否为"True"或"False"
        if row['rate'] in ["True", "False"]:
            inputs = {
                'news': row['question'],
                'experience_library': experience_library,
            }
            
            try:
                # 使用新的重试机制调用API
                prediction = call_api_with_retry(disentangling_crew.kickoff, inputs)
                
                if not prediction:
                    print("警告：跳过当前样本，因为无法获取有效预测")
                    continue
                
                # 记录结果
                result = {
                    'index': index,
                    'news': row['question'],
                    'true_label': row['rate'],
                    'prediction': prediction.split('\n')[0].strip().replace(' ', '').replace('"', '').replace("'", ''),
                    'reason': next((line for line in prediction.split('\n')[1:] if line.strip()), ''),
                    'is_correct': prediction.split('\n')[0].strip().replace(' ', '').replace('"', '').replace("'", '') == row['rate']
                }
                
                # 根据处理数量决定是训练集还是验证集
                if len(training_results) < 140:
                    training_results.append(result)
                    if result['is_correct']:
                        training_correct += 1
                    training_total += 1
                    print(f"处理训练集第 {training_total} 条新闻...")
                    print(f"当前训练集准确率: {(training_correct / training_total * 100):.2f}%")
                    
                    # 给agent提供反馈
                    feedback_inputs = {
                        'news': row['question'],
                        'result': row['rate'],
                        'judgement': result['prediction'],
                        'reason': result['reason'],
                    }
                    
                    # 使用新的重试机制调用API
                    back = call_api_with_retry(disentangling_back_crew.kickoff, feedback_inputs)
                    
                    adjust_inputs = {
                        'experience': back,
                        'experience_library': experience_library,
                    }
                    experience_library = call_api_with_retry(disentangling_adjust_crew.kickoff, adjust_inputs)
                    
                    # 添加额外的冷却时间
                    time.sleep(2)
                    
                elif len(validation_results) < 60:
                    validation_results.append(result)
                    if result['is_correct']:
                        validation_correct += 1
                    validation_total += 1
                    print(f"处理验证集第 {validation_total} 条新闻...")
                    print(f"当前验证集准确率: {(validation_correct / validation_total * 100):.2f}%")
                
                # 当处理完所有需要的样本后退出循环
                if len(training_results) >= 140 and len(validation_results) >= 60:
                    break
                    
            except Exception as e:
                print(f"处理样本时发生错误: {str(e)}")
                continue
    
    # 保存训练集结果
    training_output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'training_results.json')
    with open(training_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(training_results, json_file, ensure_ascii=False, indent=4)
    
    # 保存验证集结果
    validation_output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'validation_results.json')
    with open(validation_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(validation_results, json_file, ensure_ascii=False, indent=4)

    # 打印最终统计结果
    print("\n最终统计结果:")
    print("训练集:")
    print(f"总样本数: {training_total}")
    print(f"正确预测数: {training_correct}")
    print(f"最终准确率: {(training_correct / training_total * 100):.2f}%")
    print(f"训练集结果已保存到: {training_output_path}")
    
    print("\n验证集:")
    print(f"总样本数: {validation_total}")
    print(f"正确预测数: {validation_correct}")
    print(f"最终准确率: {(validation_correct / validation_total * 100):.2f}%")
    print(f"验证集结果已保存到: {validation_output_path}")

if __name__ == "__main__":
    run()
