import ssl
import warnings

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

import os
from dotenv import load_dotenv
from crewai import Agent , LLM
from langchain_openai import ChatOpenAI
from tools import AnalysisTools

load_dotenv()

class CustomAgents:
    def __init__(self):
       
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        #这里选择用chatgpt还是deepseek
        # self.OpenAIGPT4 = ChatOpenAI(
        #     model='deepseek-ai/DeepSeek-V2.5', temperature=0, base_url=base_url, api_key=api_key)
        self.OpenAIGPT4 = LLM(
            model="openai/deepseek-ai/DeepSeek-V2.5",
            api_key=api_key,
            base_url=base_url,
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
            llm=self.OpenAIGPT4,
        )
    # 用于辨别真实新闻和去除偏见
    def disentangling_interest_learning_module(self):
        return Agent(
            role="disentangling interest learning module",
            goal="""
            你的目标是判断新闻的政治极性和判断新闻的真实性
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
            llm=self.OpenAIGPT4,

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
            llm=self.OpenAIGPT4,
        )

    
    
   
    