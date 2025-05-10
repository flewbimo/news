新闻真实性与个性化推荐系统
项目简介
本项目旨在利用大语言模型（LLM）和多智能体协作，实现新闻的真实性判别、用户兴趣建模以及个性化新闻推荐。系统通过多种工具和任务模块，自动化处理新闻数据，分析用户兴趣，并对新闻进行真假判断和推荐。
目录结构
src/
├── crew/           # 核心功能模块（智能体、任务、工具、主控脚本等）
│   ├── agents.py           # 智能体定义
│   ├── checkRecommend.py   # 推荐系统评测脚本
│   ├── checkTrue.py        # 新闻真假判别评测脚本
│   ├── config.py           # 配置文件（模型与API）
│   ├── crew.py             # 多智能体流程与任务编排
│   ├── main.py             # 示例主入口
│   ├── tasks.py            # 任务定义
│   ├── tools.py            # 工具类（情感分析、主题分类、事实核查等）
│   └── ...
├── sets/           # 数据集
│   ├── snopeswithsum.csv   # 新闻真假标注数据
│   ├── news.tsv            # 新闻内容数据
│   ├── behaviors.tsv       # 用户行为数据
│   ├── gossip_news.tsv     # 八卦新闻数据
│   ├── politic_news_pics.tsv # 政治新闻图片数据
│   └── ...
├── results/        # 结果输出
│   ├── training_results.json      # 训练集判别结果
│   ├── validation_results.json    # 验证集判别结果
│   ├── recommend_results.csv      # 推荐系统评测结果
│   ├── results.json/csv           # 其他结果文件
│   └── ...
Apply to checkTrue.py
主要功能模块
多智能体协作（crew/crew.py）
通过 CrewAI 框架，定义了多个智能体（Agent）和任务（Task），包括兴趣建模、新闻真假判别、推荐等，支持串行或并行处理。
智能体定义（crew/agents.py）
分层兴趣建模智能体：分析用户点击新闻，生成高层次和低层次兴趣向量。
新闻判别智能体：判断新闻的真实性和政治极性。
推荐智能体：基于兴趣图谱进行新闻推荐。
任务定义（crew/tasks.py）
兴趣建模任务
新闻真假判别任务（支持工具链调用：搜索、网页抓取、情感分析、主题分类等）
推荐任务
工具集成（crew/tools.py）
情感分析、主题分类、事实核查、Snopes数据库查证等工具，部分基于 Huggingface Transformers。
评测脚本
checkTrue.py：批量评测新闻真假判别能力，输出准确率与详细结果。
checkRecommend.py：评测个性化推荐效果，输出推荐准确率。
数据说明
sets/snopeswithsum.csv：包含新闻文本及其真假标签（True/False），用于训练和验证真假判别模块。
sets/news.tsv、sets/behaviors.tsv：分别为新闻内容和用户行为数据，用于兴趣建模与推荐系统评测。
其他数据文件为扩展数据集，可根据需要使用。
依赖环境
Python 3.8+
主要依赖包（部分）：
pandas
numpy
scikit-learn
transformers
crewai
crewai_tools
langchain
python-dotenv
建议使用如下命令安装依赖（需根据实际 requirements.txt 文件补充）：
Apply to checkTrue.py
Run
环境变量配置
请在项目根目录下创建 .env 文件，配置所需的 API Key 及模型服务地址，例如：
Apply to checkTrue.py
运行方式
1. 新闻真假判别评测
Apply to checkTrue.py
Run
自动读取 sets/snopeswithsum.csv，输出训练集与验证集的判别结果和准确率，结果保存在 results/ 目录。
2. 个性化推荐评测
Apply to checkTrue.py
Run
自动读取 sets/news.tsv 和 sets/behaviors.tsv，评测推荐准确率，结果保存在 results/ 目录。
3. 示例主入口
Apply to checkTrue.py
Run
运行一个简单的新闻判别示例。
结果输出
所有评测结果均保存在 results/ 目录下，包括训练/验证集判别结果、推荐系统评测结果等，便于后续分析。
备注
本项目支持自定义扩展智能体、任务和工具，便于适配不同的新闻分析与推荐场景。
如需处理更大规模数据或接入数据库，请根据 tools.py 中的注释进行相应配置。
