# app.py
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from vanna.flask import VannaFlaskApp
from vanna.base import VannaBase

# 1. 选择并配置 Vanna 类
# class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
#     def __init__(self, config=None):
#         # 配置 ChromaDB (例如，使用内存模式)
#         # config['path'] = '/path/to/chroma/data' # 持久化路径
#         ChromaDB_VectorStore.__init__(self, config=config)
#         # 配置 OpenAI
#         OpenAI_Chat.__init__(self, config=config)

# vn = MyVanna(config={'api_key': 'sk-...', 'model': 'gpt-4-...'})
import os
from openai import OpenAI

class DeepseekLLM(VannaBase):
    def __init__(self, config=None):
        if config is None:
            config = {}

        # 从环境变量获取API密钥
        api_key = config.get("api_key") or os.environ.get("API_KEY")
        model = config.get("model") or os.environ.get("LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free")
        base_url = config.get("base_url") or os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")

        if not api_key:
            raise ValueError("必须提供 Deepseek api_key，可通过配置或环境变量 API_KEY 设置")

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def system_message(self, message: str) -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        return {"role": "assistant", "content": message}

    def generate_sql(self, question: str, **kwargs) -> str:
        # 使用父类的 generate_sql
        sql = super().generate_sql(question, **kwargs)

        # 替换 "\_" 为 "_"
        sql = sql.replace("\\_", "_")

        return sql

    def submit_prompt(self, prompt, **kwargs) -> str:
        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
        )

        return chat_response.choices[0].message.content

class MyVanna(ChromaDB_VectorStore, DeepseekLLM):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        DeepseekLLM.__init__(self, config=config)

# 从环境变量获取配置
# config = {
#     "path": "./chroma_db",  # 与train_data.py使用相同的ChromaDB存储路径
#     "api_key": os.environ.get("API_KEY"),
#     "model": os.environ.get("LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free"),
#     "base_url": os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
# }
config = {
    "path": "./chroma_db",  # 与train_data.py使用相同的ChromaDB存储路径
    "api_key": os.environ.get("API_KEY"),
    "model": os.environ.get("LLM_MODEL", "deepseek-v3"),
    "base_url": os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
}
vn = MyVanna(config=config)

# 2. (可选) 连接数据库
vn.connect_to_postgres(host='localhost', dbname='postgres', user='postgres', password='123456', port='5432')

# 3. 创建 Flask App
app = VannaFlaskApp(vn)

# 4. 运行 App
if __name__ == '__main__':
    app.run()