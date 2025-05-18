import ssl
import warnings
from crewai import Agent, LLM
from tools import AnalysisTools
from tools import ModelsTools

from config import MODEL_CONFIG
from config import API_CONFIG

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

article_analysis_tools = AnalysisTools()
moduls_tools = ModelsTools()


class CustomAgents:
    def __init__(self):
       
        self.CurrentLLM = LLM(
            model=MODEL_CONFIG["current_model"],
            api_key=API_CONFIG["deepseek"]["api_key"],
            base_url=API_CONFIG["deepseek"]["api_base"],
        )
    
    #用于构建用户对新闻的兴趣
    #暂定:从数据库中读取用户的点击信息,构建用户的个性化兴趣
    def hierarchical_interest_learning_module(self):
        return Agent(
            role="hierarchical interest learning module",
            goal="""
            你的目标是对用户点击的新闻分别生成能表征高层次兴趣和低层次兴趣的向量,以此构建个性化兴趣图谱
            """,
            backstory="""   
            你在阅读新闻声明时非常注重细节。你能够在理解这些细节的同时挑选出小细节
            适应大局。此外，你有很多为各种新闻机构工作的经验。因此，你知道如何
            轻松识别新闻报道中的判断新闻的政治极性和判断新闻的真实性。
            此外，你有新闻学博士学位，在新闻处理方面有很强的基础。
            你花了 10 多年的时间在大学里学习和教授新闻学。
            因此，您可以轻松准确地识别通过以上方式传达的各种不合时宜的新闻学问题。
            """,
            allow_delegation=False,
            verbose=True,
            llm=self.CurrentLLM,
            memory=True,
        )
    # 用于辨别真实新闻和去除偏见
    def disentangling_interest_learning_module(self):
        return Agent(
            role="news detection module",
            goal="""
            你的目标是判断新闻的政治极性和判断新闻的真实性,并根据结果正确与否做出改进
            """,
            backstory="""
            你在阅读新闻声明时非常注重细节。你能够在理解这些细节的同时挑选出小细节
            适应大局。此外，你有很多为各种新闻机构工作的经验。因此，你知道如何
            轻松识别新闻报道中的判断新闻的政治极性和判断新闻的真实性。
            此外，你有新闻学博士学位，在新闻处理方面有很强的基础。
            你花了 10 多年的时间在大学里学习和教授新闻学。
            因此，您可以轻松准确地识别通过以上方式传达的各种不合时宜的新闻学问题。
            """,
            allow_delegation=False,
            verbose=True,
            llm=self.CurrentLLM,
            tools=[article_analysis_tools.search_tool(),article_analysis_tools.scrape_tool(),moduls_tools.classification_tool(),moduls_tools.sentiment_tool()],
            #memory=True,
        )
    #用于综合以上信息进行推荐
    def next_news_prediction_module(self):
        return Agent(
            role="next news prediction module",
            goal="""
            你的目标是综合以上信息进行新闻推荐。
            """,
            backstory="""
            你在阅读新闻声明时非常注重细节。你能够在理解这些细节的同时挑选出小细节
            适应大局。此外，你有很多为各种新闻机构工作的经验。因此，你知道如何
            轻松识别新闻报道中的判断新闻的政治极性和判断新闻的真实性。
            此外，你有新闻学博士学位，在新闻处理方面有很强的基础。
            你花了 10 多年的时间在大学里学习和教授新闻学。
            因此，您可以轻松准确地识别通过以上方式传达的各种不合时宜的新闻学问题。
            """,
            allow_delegation=False,
            verbose=True,
            llm=self.CurrentLLM,
            memory=True,
        )

    
    
   
    