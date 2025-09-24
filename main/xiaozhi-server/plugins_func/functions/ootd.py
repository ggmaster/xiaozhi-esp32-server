from typing import Optional
import os
import base64
import json

from config.logger import setup_logging
from config.config_loader import get_project_dir
from core.utils.vllm import create_instance
from plugins_func.register import register_function, ToolType, ActionResponse, Action

TAG = __name__
logger = setup_logging()

OOTD_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "ootd",
        "description": (
            "从本地data目录读取图片，调用视觉大模型分析图片中人物，生成人物外貌描述，"
            "并据此生成穿搭建议提示输入（交由后续LLM输出自然语言建议）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "位于项目 data/ 目录下的图片文件名，例如 image_xxx.jpg",
                },
                "lang": {
                    "type": "string",
                    "description": "返回语言，默认zh_CN，可选zh_CN/zh_HK/en_US/ja_JP等",
                },
                "status": {
                    "type": "string",
                    "description": "可选，用户当前场景/需求描述，例如：通勤/面试/逛街等",
                },
            },
            "required": ["file_name","status"],
        },
    },
}


def _select_vllm(conn):
    """根据当前连接配置选择并构建VLLM实例。"""
    current_config = getattr(conn, "config", None)
    if not current_config:
        raise RuntimeError("未找到连接配置，无法初始化视觉模型")

    select_vllm_module = current_config.get("selected_module", {}).get("VLLM")
    if not select_vllm_module:
        raise RuntimeError("您还未设置默认的视觉分析模块")

    vllm_cfg = current_config.get("VLLM", {}).get(select_vllm_module)
    if not vllm_cfg:
        raise RuntimeError("视觉分析模块配置缺失")

    vllm_type = vllm_cfg.get("type", select_vllm_module)
    return create_instance(vllm_type, vllm_cfg)


def _read_image_to_base64(file_name: str) -> str:
    project_dir = get_project_dir()
    image_path = os.path.join(project_dir, "data", file_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"未找到指定文件：{file_name}（应位于 data/ 目录）")
    with open(image_path, "rb") as f:
        data = f.read()
    if not data:
        raise ValueError("读取图片失败或内容为空")
    # 限制大小为5MB以内
    if len(data) > 5 * 1024 * 1024:
        raise ValueError("图片大小超过限制（最大5MB）")
    return base64.b64encode(data).decode("utf-8")


@register_function("ootd", OOTD_FUNCTION_DESC, ToolType.SYSTEM_CTL)
def ootd(conn, file_name: str, lang: str = "zh_CN", status: Optional[str] = None):
    """
    读取本地图片，调用视觉大模型提取人物外貌描述，并构造穿搭建议提示词，
    返回 ActionResponse(Action.REQLLM) 供后续LLM生成自然语言输出。
    """
    try:
        # 1) 读取图片并转base64
        image_base64 = _read_image_to_base64(file_name)

        # 2) 构造视觉描述问题
        question = (
            "请对图片中的人物进行详细描述，只关注人物的细节（性别、年龄段、身材特征、衣着单品、颜色、款式、风格等），"
            "不要描述背景或与人物无关的内容。"
        )
        if lang:
            question += f" 请使用{lang}回答。"

        # 3) 调用视觉模型
        vllm = _select_vllm(conn)
        desc = vllm.response(question, image_base64)

        # 4) 组织穿搭建议提示词，交由后续LLM生成最终自然语言回复
        scene = status or "日常通勤/外出（未提供具体场景）"
        guidance = (
            f"请用{lang}根据以下人物描述与场景，给出可执行的穿搭建议：\n\n"
            f"【人物外貌】{desc}\n\n"
            f"【场景】{scene}\n\n"
        )

        logger.bind(tag=TAG).info("OOTD 图像分析完成，已生成穿搭建议提示词")
        return ActionResponse(action=Action.REQLLM, result=guidance, response=None)

    except Exception as e:
        logger.bind(tag=TAG).error(f"ootd 处理异常: {e}")
        # 直接回复错误信息
        return ActionResponse(action=Action.RESPONSE, result=None, response=f"生成失败：{e}")
