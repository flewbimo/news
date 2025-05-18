from transformers import pipeline
import os
from dotenv import load_dotenv
from crewai_tools import RagTool
from langchain.tools import QuerySQLDataBaseTool
from langchain.tools.base import StructuredTool
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
import ssl
import warnings
from crewai_tools import (
    ScrapeWebsiteTool,
    SerperDevTool
)
import pandas as pd


ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

load_dotenv()

# host = os.getenv("MYSQL_HOST", "localhost")
# user = os.getenv("MYSQL_USER", "root")
# password = os.getenv("MYSQL_PASSWORD", "password")
# database = os.getenv("MYSQL_DATABASE", "test_db")
# engine = create_engine(
#     f"mysql+pymysql://{user}:{password}@{host}/{database}")
# db = SQLDatabase(engine)
# query_tool = QuerySQLDataBaseTool(db=db)
# query = "SELECT * FROM myurl"


# class MySQLQueryTool(RagTool):
#     name: str = "MySQLQueryTool"  
#     description: str = "执行 SQL 查询并返回 MySQL 数据库的查询结果"

#     def _run(self, query: str) -> str:
#         """执行 SQL 查询"""
#         try:
#             return db.run(query)
#         except Exception as e:
#             return f"MySQL 查询失败: {str(e)}"


        
class AnalysisTools:
    def __init__(self):
        pass

    def search_tool(self):       
        return SerperDevTool()
    
    def scrape_tool(self):   
        return ScrapeWebsiteTool()
    
    def database_tool(self):
        return MySQLQueryTool()

    def checkTrue_tool(self):
        return SnopesFactCheckTool()


sentiment_analyzer = pipeline("sentiment-analysis")
news_classifier = pipeline("zero-shot-classification",
                           model="facebook/bart-large-mnli")
fact_checker = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli")
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(os.path.dirname(
    current_dir)), 'src', 'sets', 'snopeswithsum.csv')
# 读取CSV文件，只读取前50行
df = pd.read_csv(data_path, encoding='utf-8', nrows=50)

class SentimentAnalysisTool(RagTool):
    name: str = "SentimentAnalysisTool"
    description: str = "执行情感分析，返回新闻的情感倾向（积极、消极或中性）"

    def _run(self, text: str) -> str:
        """分析新闻的情感"""
        try:
            result = sentiment_analyzer(text)[0]
            # 返回情感类别，如 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
            return result["label"].upper()
        except Exception as e:
            return f"情感分析失败: {str(e)}"


class NewsClassificationTool(RagTool):
    name: str = "NewsClassificationTool"
    description: str = "分析新闻内容并返回最相关的新闻主题标签"

    def _run(self, text: str) -> str:
        """根据新闻内容分析其主题"""
        try:
            candidate_labels = ["politics", "sports",
                                "technology", "economy", "entertainment"]
            result = news_classifier(text, candidate_labels)
            highest_score_index = result['scores'].index(max(result['scores']))
            return result['labels'][highest_score_index]  # 返回最相关的标签
        except Exception as e:
            return f"新闻内容分析失败: {str(e)}"


class FactCheckingTool(RagTool):
    name: str = "FactCheckingTool"
    description: str = "执行新闻事实验证，判断其真假并给出相应的可信度分数"

    def _run(self, text: str) -> str:
        """验证新闻的真实性"""
        try:
            candidate_labels = ["real", "fake"]
            result = fact_checker(text, candidate_labels)
            label = result['labels'][0]
            score = round(result['scores'][0] * 100, 2)
            return f"{label} ({score}%)"  # 返回真假和可信度百分比
        except Exception as e:
            return f"事实验证失败: {str(e)}"


class SnopesFactCheckTool(RagTool):
    name: str = "SnopesFactCheckTool"
    description: str = "根据新闻标题查询Snopes数据库中的新闻真假信息"

    def __init__(self):
        pass   
    def _run(self, title: str) -> str:
        """根据新闻标题查询真假信息"""
        try:
            # 在数据集中查找匹配的标题
            result = self.df[self.df['question'].str.contains(title, case=False, na=False)]
            if len(result) > 0:
                # 返回第一个匹配结果的真假信息
                return f"标题: {result.iloc[0]['question']}\n真假: {result.iloc[0]['rate']}"
            else:
                return f"未找到标题为 '{title}' 的新闻"
        except Exception as e:
            return f"查询失败: {str(e)}"


class ModelsTools:
    def __init__(self):
        pass

    def sentiment_tool(self):
        return SentimentAnalysisTool()

    def classification_tool(self):
        return NewsClassificationTool()

    def factChecking_tool(self):
        return FactCheckingTool()
        
    
    
   
    
    
    






