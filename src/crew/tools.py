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


ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

load_dotenv()

host = os.getenv("MYSQL_HOST", "localhost")
user = os.getenv("MYSQL_USER", "root")
password = os.getenv("MYSQL_PASSWORD", "password")
database = os.getenv("MYSQL_DATABASE", "test_db")
engine = create_engine(
    f"mysql+pymysql://{user}:{password}@{host}/{database}")
db = SQLDatabase(engine)
query_tool = QuerySQLDataBaseTool(db=db)
query = "SELECT * FROM myurl"


class MySQLQueryTool(RagTool):
    name: str = "MySQLQueryTool"  
    description: str = "执行 SQL 查询并返回 MySQL 数据库的查询结果"

    def _run(self, query: str) -> str:
        """执行 SQL 查询"""
        try:
            return db.run(query)
        except Exception as e:
            return f"MySQL 查询失败: {str(e)}"


        
class AnalysisTools:
    def __init__(self):
        pass

    def search_tool(self):       
        return SerperDevTool()
    
    def scrape_tool(self):   
        return ScrapeWebsiteTool()
    
    def database_tool(self):
        return MySQLQueryTool()

    
    
    
   
    
    
    






