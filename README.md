# 新闻真实性与个性化推荐系统

## 一、项目简介

本项目基于大语言模型（LLM）与多智能体协作技术，实现新闻真实性判别、用户兴趣建模及个性化新闻推荐功能。系统通过集成多种工具与任务模块，自动化处理新闻数据，精准分析用户兴趣，同时对新闻进行真假判断与智能推荐，为用户提供真实、符合兴趣的新闻内容。

## 二、目录结构



```
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

│   ├── gossip\_news.tsv     # 八卦新闻数据

│   ├── politic\_news\_pics.tsv # 政治新闻图片数据

│   └── ...

├── results/        # 结果输出

│   ├── training\_results.json      # 训练集判别结果

│   ├── validation\_results.json    # 验证集判别结果

│   ├── recommend\_results.csv      # 推荐系统评测结果

│   ├── results.json/csv           # 其他结果文件

│   └── ...
```

## 三、主要功能模块

### （一）多智能体协作（`crew/``crew.py`）

基于 **CrewAI 框架**，系统定义了多个智能体（Agent）与任务（Task），涵盖兴趣建模、新闻真假判别、推荐等核心功能，支持串行或并行处理，实现高效协同工作。

### （二）智能体定义（`crew/``agents.py`）

**分层兴趣建模智能体**：通过分析用户点击新闻记录，生成高层次和低层次兴趣向量，精准捕捉用户兴趣偏好。

**新闻判别智能体**：具备判断新闻真实性与政治极性的能力，为用户筛选可靠新闻。

**推荐智能体**：基于构建的兴趣图谱，为用户推荐个性化新闻内容。

### （三）任务定义（`crew/``tasks.py`）

**兴趣建模任务**：从用户行为数据中挖掘兴趣特征，构建用户兴趣模型。

**新闻真假判别任务**：支持调用搜索、网页抓取、情感分析、主题分类等工具链，全方位核查新闻真实性。

**推荐任务**：根据用户兴趣模型与新闻内容，完成个性化新闻推荐。

### （四）工具集成（`crew/``tools.py`）

集成情感分析、主题分类、事实核查、Snopes 数据库查证等工具，部分工具基于 **Huggingface Transformers** 实现，为新闻分析与处理提供强大支持。

## 四、评测脚本

`checkTrue.py`：批量评测新闻真假判别能力，输出准确率及详细结果，帮助评估模型在真实性判断方面的性能。

`checkRecommend.py`：评测个性化推荐效果，输出推荐准确率，衡量推荐系统的有效性。

## 五、数据说明

`sets/snopeswithsum.csv`：包含新闻文本及其真假标签（`True/False`），用于训练和验证新闻真假判别模块。

`sets/news.tsv`**、**`sets/behaviors.tsv`：分别存储新闻内容和用户行为数据，为兴趣建模与推荐系统评测提供数据基础。

其他数据文件为扩展数据集，可根据项目实际需求灵活使用。

## 六、依赖环境

**Python 版本**：3.8+

**主要依赖包（部分）**：

pandas

numpy

scikit-learn

transformers

crewai

crewai\_tools

langchain

python-dotenv

**安装命令**：



```
\# 需根据实际 requirements.txt 文件补充完整依赖

pip install -r requirements.txt
```

## 七、运行指南

### （一）环境变量配置

在项目根目录下创建 `.env` 文件，配置所需的 **API Key** 及模型服务地址，示例配置如下：



```
\# 大语言模型 API Key

LLM\_API\_KEY=your\_api\_key

\# 其他 API 配置

OTHER\_API\_URL=your\_api\_url
```

### （二）运行方式

**新闻真假判别评测**

运行 `checkTrue.py` 脚本，自动读取 `sets/snopeswithsum.csv` 数据，输出训练集与验证集的判别结果和准确率，结果保存至 `results/` 目录。



```
python src/crew/checkTrue.py
```

**个性化推荐评测**

运行 `checkRecommend.py` 脚本，自动读取 `sets/news.tsv` 和 `sets/behaviors.tsv` 数据，评测推荐准确率，结果保存至 `results/` 目录。



```
python src/crew/checkRecommend.py
```

**示例主入口**

运行 `main.py` 脚本，体验简单的新闻判别示例。



```
python src/crew/main.py
```

## 八、结果输出

所有评测结果均保存在 `results/` 目录下，包括训练 / 验证集判别结果、推荐系统评测结果等，方便后续分析与评估。

## 九、备注

本项目支持自定义扩展智能体、任务和工具，可灵活适配不同的新闻分析与推荐场景。

如需处理更大规模数据或接入数据库，请参考 `tools.py` 中的注释进行相应配置。
