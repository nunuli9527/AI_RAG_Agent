"""
中间件（监控 + 切换提示词）
作用：监控工具调用 + 动态切换提示词。
功能：
日志记录：谁调用了什么工具、参数是什么
报错捕获
报告场景自动切换提示词（普通对话 → 报告生成）
"""

from typing import Callable
from utils.prompt_loader import load_rag_prompt, load_report_prompt, load_system_prompt
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


# 工具执行的监控
@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,                                       # 请求数据的封装
        handler: Callable[[ToolCallRequest], ToolMessage | Command]     # 执行函数本身
) -> ToolMessage | Command:
    logger.info(f"[monitor_tool]工具名称: {request.tool_call['name']}")
    logger.info(f"[monitor_tool]工具参数: {request.tool_call['args']}")

    try:
        result =  handler(request)
        logger.info(f"[monitor_tool]工具{request.tool_call['name']}调用成功")

        if request.tool_call["name"] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return  result
    except Exception as e:
        logger.error(f"[monitor_tool]工具{request.tool_call['name']}调用失败, 原因: {str(e)}")
        raise e


# 在模型执行前输出日志
@before_model
def log_before_model(
        state: AgentState,      # 整个Agent智能体中的状态记录
        runtime: Runtime        # 记录了整个执行过程中的的上下文信息
):
    logger.info(f"[log_before_model]即将调用模型, 带有{len(state['messages'])}条消息.")

    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__}{state['messages'][-1].content.strip()}")

    return None


# 动态切换提示词
@dynamic_prompt     # 每一次生成提示词前都会调用这个函数
def report_prompt_switch(request: ModelRequest):     # 动态切换提示词
    is_report = request.runtime.context.get("report", False)
    if is_report:       # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompt()

    return load_system_prompt()