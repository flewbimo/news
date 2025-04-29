import sys
import os
from dotenv import load_dotenv
from crew import *
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

# 训练集预处理


def preprocess_training_data():
    try:
        # 获取当前文件所在目录的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建数据文件的完整路径
        news_path = os.path.join(os.path.dirname(
            os.path.dirname(current_dir)), 'src', 'sets', 'news.tsv')
        behaviors_path = os.path.join(os.path.dirname(
            os.path.dirname(current_dir)), 'src', 'sets', 'behaviors.tsv')

        # 读取TSV文件
        news_df = pd.read_csv(news_path, sep='\t')
        behaviors_df = pd.read_csv(behaviors_path, sep='\t')

        return news_df, behaviors_df

    except FileNotFoundError as e:
        print(f"错误：找不到数据文件 {str(e)}")
        return None, None
    except Exception as e:
        print(f"读取数据时发生错误：{str(e)}")
        return None, None

def run():
    # 获取数据
    news_df, behaviors_df = preprocess_training_data()
    if news_df is None or behaviors_df is None:
        return
        
    # 创建结果记录列表
    results = []
    total_count = 0
    correct_count = 0
    
    # 遍历行为数据集的每一行
    for index, row in behaviors_df.iterrows():
        if pd.notna(row['History']):
            historyNews = row['History'].spilt();
            news_data = []
            for news_id in historyNews:
                news_data.append(news_df[news_df['News ID'] == news_id])
            # 准备输入数据
            inputs = {
                'news_data': news_df
            }
            
            print(f"处理第 {index + 1} 个用户的推荐...")
            # 获取模型预测结果
            prediction = next_news_crew.kickoff(inputs)
            
            # 处理实际点击数据
            actual_clicks = []
            for imp in row['Impressions'].split():
                news_id, clicked = imp.split('-')
                if clicked == '1':
                    actual_clicks.append(news_id)
            
            # 记录结果
            result = {
                'index': index,
                'user_id': row['User ID'],
                'actual_clicks': actual_clicks,
                'predicted_recommendations': prediction.raw,
                'is_correct': any(rec in actual_clicks for rec in prediction.raw.split())
            }
            results.append(result)
            
            # 统计正确数量
            if result['is_correct']:
                correct_count += 1
            total_count += 1
            
            # 实时打印准确率
            accuracy = (correct_count / total_count) * 100
            print(f"当前准确率: {accuracy:.2f}%")
    
    # 将结果保存到CSV文件
    results_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'recommend_results.csv')
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    
    # 打印最终统计结果
    print("\n最终统计结果:")
    print(f"总样本数: {total_count}")
    print(f"正确预测数: {correct_count}")
    print(f"最终准确率: {(correct_count / total_count * 100):.2f}%")
    print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    run()

