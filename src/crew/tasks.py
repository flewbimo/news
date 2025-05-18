
from crewai import Task
from textwrap import dedent
from agents import CustomAgents
from tools import AnalysisTools
from tools import ModelsTools
article_analysis_tools = AnalysisTools()
moduls_tools = ModelsTools()

class AnalysisTasks:
    def __init__(self):
        pass
    
    #目前是纯LLM任务,后续可以加上其他的工具
    def hierarchical_interest_learning(self):
        return Task(
            description=(
                """
                对于用户点击的新闻集合({news_data}),
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
                根据之前总结的经验库({experience_library}),
                你需要判断以下新闻的真实性：
                "{news}"
                请根据你的判断自由选择以下一个或多个工具：
                - Search the internet with Serper：搜索新闻
                - Read website content：提取网页内容
                - SentimentAnalysisTool：分析情感倾向
                - NewsClassificationTool：判断新闻主题

                """
            ),
            expected_output="""
                输出"True"或"False".然后回车,输出判断依据
            """,
            agent=CustomAgents().disentangling_interest_learning_module(),
            tools=[article_analysis_tools.search_tool(), article_analysis_tools.scrape_tool(
            ), moduls_tools.classification_tool(), moduls_tools.sentiment_tool()],
            #tools=[moduls_tools.classification_tool()]
            #tools=[moduls_tools.classification_tool(),moduls_tools.sentiment_tool()]
        )

    def disentangling_back(self):
        return Task(
            description=(
                """
                该任务用来得到经验或者反思,无需使用任何工具
                新闻({news})的真实性结果是({result}),而你的上一个任务判断其真实性为({judgement}),上一个任务做出这个选择的原因是({reason})。
                如果上一个任务选择正确，请记录经验；
                如果上一个任务选择错误，请进行反思并记录。
                """
            ),
            expected_output="""
                输出得到的经验或者反思
            """,
            agent=CustomAgents().disentangling_interest_learning_module(),
            tools=[],            
            # tools=[moduls_tools.classification_tool()]
        )

    def disentangling_adjust(self):
        return Task(
            description=(
                """
                该任务用来总结之前的反思并更新新闻判断策略,
                对于反思得到的经验({experience}),请总结关键之处并统合到目前的经验库({experience_library})中,
                请尽量精简经验库,最好保持在10条以内,否则为已有的经验库先进行总结
                将统合完成的经验库输出,
                """
            ),
            expected_output="""
                输出统合完成的经验库
            """,
            agent=CustomAgents().disentangling_interest_learning_module(),
            tools=[],
            # tools=[moduls_tools.classification_tool()]
        )
    def next_news_prediction(self):
        return Task(
            description=(
                """
                根据用户的兴趣图谱({user_history}),判断新闻集({news_data})是否是用户感兴趣的新闻.
                如果新闻是用户感兴趣的,则返回"1",
                如果新闻不是用户感兴趣的,则返回"0".
                """
            ),
            expected_output="""
                输出"1"或"0",用逗号隔开.
            """,
            agent=CustomAgents().next_news_prediction_module(),
        )

    
    
   



