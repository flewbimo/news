
from crewai import Task
from textwrap import dedent
from agents import CustomAgents
from tools import AnalysisTools
article_analysis_tools = AnalysisTools()

class AnalysisTasks:
    def __init__(self):
        pass
    
    #目前是纯LLM任务,后续可以加上其他的工具
    def hierarchical_interest_learning(self):
        return Task(
            description=(
                """
                对于用户点击的新闻({news}),
                得到新闻文章的高层次兴趣向量和低层次兴趣向量.
                基于语言知识和语义理解能力，对新闻进行深度分析。比如对 “人工智能”“芯片” 相关的新闻，
                理解这些新闻背后的技术概念、行业动态等信息，进而生成能表征高层次兴趣的向量表示。
                这个向量不仅包含关键词的表面信息，还涵盖了对相关领域知识的理解，能更全面地反映用户
                对新闻事件或主题的高层次兴趣。
                对新闻内容进行细致解读，理解文章的结构、细节和语义逻辑。比如对于一篇关于特定手机型号的新
                闻，可以分析出手机的各项功能、用户评价等内容，从而得到对该新闻文章的低层次兴趣向量。
                结合高层次兴趣向量和低层次兴趣向量，全面地了解用户对新闻的兴趣。
                
                
                """
            ),
            expected_output="""
                输出根据高层次兴趣向量和低层次兴趣向量得到的用户兴趣图谱.
            """,
            agent=CustomAgents().hierarchical_interest_learning_module(),
        )

    def disentangling_interest_learning(self):
        return Task(
            description=(
                """
                检查新闻"({news})"的真实性,
                如果新闻是虚假的,则返回"False",
                如果新闻是真实的,则返回"True".
                """
            ),
            expected_output="""
                输出新闻"True"或"False".
            """,
            agent=CustomAgents().disentangling_interest_learning_module(),
        )

    def next_news_prediction(self):
        return Task(
            description=(
                """
                根据用户的兴趣图谱和过滤的新闻集中的新闻,推荐一个用户最感兴趣的新闻.
                """
            ),
            expected_output="""
                提供推荐的一个新闻.
            """,
            agent=CustomAgents().next_news_prediction_module(),
        )

    
    
   



