"""
智能体大脑（核心）
作用：整个系统的大脑，决定要不要调用工具、调用哪个工具。
功能：
组装大模型、工具、提示词、中间件
接收用户问题
按照 ReAct 逻辑思考
判断是否需要调用工具
流式输出回答
"""

from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompt
from agent.tools.agent_tools import (rag_summerize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model = chat_model,
            system_prompt = load_system_prompt(),
            tools = [rag_summerize, get_weather, get_user_location, get_user_id,
                     get_current_month, fetch_external_data, fill_context_for_report],
            middleware = [monitor_tool, log_before_model, report_prompt_switch]
        )


    def execute_stream(self, query: str):
        input_dict = {
            "messages":[
                {"role": "user", "content": query}
            ]
        }

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            lastest_message = chunk["messages"][-1]
            if lastest_message:
                yield lastest_message.content.strip() + "\n"



if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("扫地机器人在我所在的地区的气温下如何保养"):
        print(chunk, end="", flush=True)

    # for chunk in agent.execute_stream("给我生成我的使用报告"):
    #     print(chunk, end="", flush=True)