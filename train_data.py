import os
import json
import argparse
from lxml import etree as ET
from vanna.chromadb import ChromaDB_VectorStore
from vanna.base import VannaBase
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
            raise ValueError("必须提供  api_key，可通过配置或环境变量 API_KEY 设置")

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

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Vanna 训练脚本，用于加载不同类型的训练数据。")
    parser.add_argument(
        "data_type",
        choices=["ddl", "sql_with_question", "doc"],
        help="要加载的训练数据类型 (ddl, sql_with_question, doc)。",
    )
    args = parser.parse_args()
    
    # 初始化Vanna实例
    config = {
        "path": "./chroma_db",  # ChromaDB存储目录路径
        "api_key": os.environ.get("API_KEY"),  #  API密钥
        "model": os.environ.get("LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free"),  # 模型名称
        "base_url": os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")  # API基础URL
    }
    
    # 确保目录存在
    os.makedirs(config["path"], exist_ok=True)
    
    try:
        vn = MyVanna(config=config)
        
        # 根据参数选择训练文件和方法
        if args.data_type == "ddl":
            file_path = 'training_data/ddl.xml'
            print(f"开始训练 {file_path} 数据...")
            
            # 使用lxml解析XML文件，更宽松的解析方式
            parser = ET.XMLParser(recover=True)
            tree = ET.parse(file_path, parser=parser)
            root = tree.getroot()
            
            # 获取所有<table-ddl>节点
            table_ddls = root.xpath('//table-ddl')
            total_records = len(table_ddls)
            
            for i, ddl_node in enumerate(table_ddls):
                ddl_statement = ddl_node.text.strip()
                if ddl_statement:
                    print(f"正在训练第 {i+1}/{total_records} 条 DDL...")
                    vn.train(ddl=ddl_statement)
            print(f"{file_path} 训练完成！成功训练了 {total_records} 条 DDL。")
    
        elif args.data_type == "sql_with_question":
            file_path = 'training_data/sql_with_question.json'
            print(f"开始训练 {file_path} 数据...")
            with open(file_path, 'r', encoding='utf-8') as file:
                training_data = json.load(file)
            
            total_records = len(training_data)
            for i, item in enumerate(training_data):
                question = item.get("question")
                sql = item.get("answer")
                
                if question and sql:
                    print(f"正在训练第 {i+1}/{total_records} 条 SQL-Question 对...")
                    vn.train(question=question, sql=sql)
            print(f"{file_path} 训练完成！成功训练了 {total_records} 条 SQL-Question 对。")
    
        elif args.data_type == "doc":
            file_path = 'training_data/doc.json'
            print(f"开始训练 {file_path} 数据...")
            with open(file_path, 'r', encoding='utf-8') as file:
                training_data = json.load(file)
            
            total_records = len(training_data)
            for i, item in enumerate(training_data):
                documentation = item.get("documentation")
                if documentation:
                    print(f"正在训练第 {i+1}/{total_records} 条文档...")
                    vn.train(documentation=documentation)
            print(f"{file_path} 训练完成！成功训练了 {total_records} 条文档。")
    
        else:
            print(f"错误：未知的 data_type 参数值 '{args.data_type}'")
        
        print("所有训练数据处理完成！")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")

if __name__ == "__main__":
    main()
