
import ssl
import warnings

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

from crewai import Crew, Process
from agents import CustomAgents
from tasks import AnalysisTasks


hierarchical_agent = CustomAgents().hierarchical_interest_learning_module()
disentangling_agent = CustomAgents().disentangling_interest_learning_module()
next_news_agent = CustomAgents().next_news_prediction_module()



hierarchical_task = AnalysisTasks().hierarchical_interest_learning()
disentangling_task = AnalysisTasks().disentangling_interest_learning()
next_news_task = AnalysisTasks().next_news_prediction()



NewsCrew = Crew(
    agents=[
        hierarchical_agent,
        disentangling_agent,
        next_news_agent,
    ],
    tasks=[
        hierarchical_task,
        disentangling_task,
        next_news_task,
    ],
    process=Process.sequential,
    verbose=True,
)

disentangling_crew = Crew(
    agents=[
        disentangling_agent,
    ],
    tasks=[
        disentangling_task,
    ],
    process=Process.sequential,
    verbose=True,
)
hierarchical_crew = Crew(
    agents=[
        hierarchical_agent,
    ],
    tasks=[
        hierarchical_task,
    ],
    process=Process.sequential,
    verbose=True,
)
next_news_crew = Crew(
    agents=[
        next_news_agent,
    ],
    tasks=[
        next_news_task,
    ],
    process=Process.sequential,
    verbose=True,
)