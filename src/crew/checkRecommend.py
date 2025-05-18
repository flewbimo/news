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

def analyze_user_history(news_df, history_news_ids):
    """分析用户历史点击的新闻特征"""
    history_news = news_df[news_df['NewsId'].isin(history_news_ids)]
    news_titles = []
    if history_news.empty:
        return ""
    for index, row in history_news.iterrows():
        if pd.notna(row['Title']):
            news_titles.append(row['Title'])
    # 使用hierarchical_crew分析用户历史点击
    inputs = {
        'news_data': news_titles
    }
    
    # 获取大模型分析结果
    analysis_result = hierarchical_crew.kickoff(inputs)
    return analysis_result.raw

def get_news_info(news_df, news_id):
    """获取新闻的详细信息"""
    news = news_df[news_df['NewsId'] == news_id]
    if news.empty:
        return None
    return news.iloc[0]

def run():
    news_df, behaviors_df = preprocess_training_data()
    if news_df is None or behaviors_df is None:
        return
        
    results = []
    total_count = 0
    correct_count = 0
    processed_users = 0  # 添加计数器
    
    for index, row in behaviors_df.iterrows():
        if processed_users >= 5:  # 只处理前5个客户
            break
            
        if pd.notna(row['History']):
            processed_users += 1  # 增加计数器
            history_news_ids = row['History'].split()
            
            # 分析用户历史点击
            history_analysis = analyze_user_history(news_df, history_news_ids)
            
            # 处理当前展示的新闻
            impressions = row['Impressions'].split()
            impression_news = []
            
            # 收集所有展示新闻的信息
            for imp in impressions:
                news_id, actual_click = imp.split('-')
                news_info = get_news_info(news_df, news_id)
                if news_info is not None:
                    impression_news.append({
                        'news_id': news_id,
                        'title': news_info['Title'],
                        'actual_click': int(actual_click)
                    })
            
            # 准备输入数据
            inputs = {
                'user_history': history_analysis,
                'news_data': [news['title'] for news in impression_news]
            }
            
            print(f"处理用户 {row['UserId']} 的推荐...")
            
            # 获取模型预测结果
            prediction = next_news_crew.kickoff(inputs)
            
            # 处理预测结果
            predicted_clicks = [int(pred)  for pred in prediction.raw.split(',')]
            
            # 记录结果
            for i, news in enumerate(impression_news):
                result = {
                    'index': index,
                    'user_id': row['UserId'],
                    'news_id': news['news_id'],
                    'news_title': news['title'],
                    'actual_click': news['actual_click'],
                    'predicted_click': predicted_clicks[i] if i < len(predicted_clicks) else 0,
                    'is_correct': (predicted_clicks[i] if i < len(predicted_clicks) else 0) == news['actual_click']
                }
                results.append(result)
                
                if result['is_correct']:
                    correct_count += 1
                total_count += 1
            
            accuracy = (correct_count / total_count) * 100
            print(f"当前准确率: {accuracy:.2f}%")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'recommend_results.csv')
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print("\n最终统计结果:")
    print(f"总样本数: {total_count}")
    print(f"正确预测数: {correct_count}")
    print(f"最终准确率: {(correct_count / total_count * 100):.2f}%")
    print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    run()

