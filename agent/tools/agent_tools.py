"""
工具集（能力库）
作用：给 Agent 提供所有可调用的功能。
里面包含的工具：
rag_summerize：查知识库
get_weather：查天气
get_user_location：获取用户城市
get_user_id：获取用户 ID
get_current_month：获取当前月份
fetch_external_data：读取用户使用记录
fill_context_for_report：标记进入报告模式
"""

import os.path
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
import requests

rag = RagSummarizeService()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

external_data = {}


@tool(description="从向量存储中检索参考资料")
def rag_summerize(query: str):
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    """对接 wttr.in 实时天气 API，无需注册，返回结构化 JSON"""
    try:
        session = requests.Session()
        session.trust_env = False

        url = f"https://wttr.in/{city}?format=j1"
        resp = session.get(url)
        data = resp.json()

        current = data["current_condition"][0]
        return (
            f"城市{city}天气为{current['weatherDesc'][0]['value']}，"
        )
    except Exception as e:
        logger.error(f"[get_weather]获取{city}天气失败: {e}")
        return f"获取{city}天气失败"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    return random.choice(["北京", "上海", "广州", "深圳", "杭州"])


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data():
    """
        {
            "user_id": {
                "month" : {"特征": xxx, "效率": xxx, ...},
                "month" : {"特征": xxx, "效率": xxx, ...},
                "month" : {"特征": xxx, "效率": xxx, ...},
                ...
            },
            "user_id": {
                "month" : {"特征": xxx, "效率": xxx, ...},
                "month" : {"特征": xxx, "效率": xxx, ...},
                "month" : {"特征": xxx, "效率": xxx, ...},
                ...
            },
            ...
        }
        :return:
        """
    if not external_data:
        external_data_path =  get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"[generate_external_data]外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "消耗": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未找到用户{user_id}在{month}的记录")
        return ""


@tool(description="无入参, 无返回值, 调用后触发中间件自动为报告生成的场景动态注入上下文信息, 为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"


# if __name__ == '__main__':
    # print(fetch_external_data("1021", "2025-01"))

