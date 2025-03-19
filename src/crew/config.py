import os
from dotenv import load_dotenv

load_dotenv()

# AI模型配置
MODEL_CONFIG = {
    # 可选模型列表
    "available_models": [
        "openai/deepseek-ai/DeepSeek-V2.5",
        "openai/Qwen/Qwen2.5-VL-72B-Instruct",
        "openai/deepseek-ai/DeepSeek-R1",
    ],
    # 当前选择的模型
    "current_model": "openai/deepseek-ai/DeepSeek-V2.5"
}

# API配置
API_CONFIG = {
    "deepseek": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_base": os.getenv("OPENAI_BASE_URL")
    }
}



# 主要功能配置
APP_CONFIG = {
    
} 