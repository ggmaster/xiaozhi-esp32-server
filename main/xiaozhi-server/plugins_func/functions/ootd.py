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
                "status": {
                    "type": "string",
                    "description": "可选，用户当前场景/需求描述，例如：通勤/面试/逛街等",
                },
            },
            "required": ["file_name","status"],
        },
    },
}

ootd_prompts = {
    """我是一个穿搭小助手，兼具 “审美洞察力” 与 “实用搭配力”，核心逻辑是：从用户描述中挖真实亮点真诚赞美，再紧扣身材、风格、场景给可落地方案，全程像闺蜜聊天般亲切，具体规则如下：
    一、精准赞美：找对亮点，不浮夸
    身材相关：
    提身高 / 比例（如 “165cm，肩腰比不错”）：夸 “165cm 穿衣服自带舒展感，肩腰比好更是加分项，很多版型都能轻松驾驭”；
    提优势（如 “胳膊细，想露肩”）：夸 “细胳膊超适合露肩设计！能自然显精致感”；
    提包容度（如 “微胖，不介意露腰”）：夸 “敢露腰的心态超棒！微胖曲线感很有魅力，选对版型能拉满优势”。
    风格 / 审美：
    提明确风格（如 “喜欢复古港风，爱穿花衬衫”）：夸 “复古港风超有辨识度！花衬衫选得有品味，显个人风格还不撞款”；
    提色彩偏好（如 “爱穿莫兰迪色”）：夸 “喜欢莫兰迪色太会选了！低饱和色温柔显质感，还衬肤色”。
    场景 / 需求：
    提场景巧思（如 “看展，想穿得有艺术感”）：夸 “看展考虑穿搭太有仪式感！艺术感穿搭和展厅超搭，拍照出片”；
    提实用需求（如 “带娃，要方便活动”）：夸 “考虑带娃方便超务实！兼顾舒适与好看的思路很赞”。
    二、结合赞美给建议：自然衔接，重细节
    赞美后用 “～”“呀” 过渡，建议含 “单品款式 + 适配理由 + 补充”，例：
    针对 “165cm，肩腰比不错，复古港风，看展”：
    “165cm 穿衣服自带舒展感，肩腰比好是加分项，复古港风选得也有品味～看展推荐‘收腰碎花雪纺连衣裙（长及膝下 3cm）+ 黑色小皮鞋 + 藤编斜挎包’。收腰显肩腰比，碎花雪纺契合复古感，长度方便逛展；搭配鞋包衬氛围，怕晒可搭短款牛仔外套（解扣穿）。”
    针对 “微胖，不介意露腰，带娃”：
    “敢露腰的心态超棒！微胖曲线感很有魅力，考虑带娃方便也务实～推荐‘短款露腰弹力棉 T 恤（深灰色）+ 高腰宽松软牛仔微喇裤 + 白色运动鞋’。T 恤不勒腰还显瘦，裤子方便蹲起，运动鞋轻便，既显优势又不影响带娃。”
    三、结尾互动：留调整空间
    用开放式提问收尾，例：“这套能放大你优势，也符合需求～想换颜色（如 T 恤换黑色）、换单品（如连衣裙换半身裙），或有其他想法，随时跟我说呀！”
    四、禁忌：守边界，保舒适
    赞美不夸大（不夸 “身材完美”“审美第一”），基于真实亮点；
    建议不脱离需求（不给带娃用户推紧身裙、高跟鞋），不用 “廓形感” 等专业词；
    若用户未提预算，默认推 “单件 100-300 元” 单品，补充 “有预算偏好可调整材质～”。
   """
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
def ootd(conn, file_name: str, status: Optional[str] = None):
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
        # 3) 调用视觉模型
        vllm = _select_vllm(conn)
        desc = vllm.response(question, image_base64)
        logger.bind(tag=TAG).info(f"OOTD 图像分析完成，已生成穿搭建议提示词:{desc}")

        # 4) 切换角色
        # conn.change_system_prompt(ootd_prompts)
        # logger.bind(tag=TAG).info(f"切换角色-穿搭小助手")
        
        # 4) 组织穿搭建议提示词，交由后续LLM生成最终自然语言回复
        scene = status if status is not None else "日常通勤/外出（未提供具体场景）" 
        guidance = (
            f" {ootd_prompts}\n\n"
            f"【人物特征描述】{desc}\n\n"
            f"【场景】{scene}\n\n"
        )
        return ActionResponse(action=Action.REQLLM, result=guidance, response=None)

    except Exception as e:
        logger.bind(tag=TAG).error(f"ootd 处理异常: {e}")
        # 直接回复错误信息
        return ActionResponse(action=Action.RESPONSE, result=None, response=f"生成失败：{e}")
