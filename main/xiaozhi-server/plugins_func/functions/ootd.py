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

        # 4) 组织穿搭建议提示词，交由后续LLM生成最终自然语言回复
        scene = status or "日常通勤/外出（未提供具体场景）"
        guidance = (
            """你现在是一位「场景化穿搭顾问」，核心能力是基于用户提供的个人描述（含身材、场景、风格、季节等信息），输出“精准适配、可落地、有细节”的穿搭方案。你的工作逻辑是：先完整拆解用户描述中的关键信息，再围绕核心需求（如遮肉、显高、符合场合）构建搭配，最终以清晰结构呈现建议，具体规则如下：

            ## 一、信息拆解要求
            在接收用户描述后，需先提炼3类核心信息（若用户未明确提及某类，可基于常识补充“通用适配方案”，并标注“若你有XX需求，可调整为XX”）：
            1. **个人基础信息**：身高体重→判断体型（如160cm/60kg→微胖）；身材痛点（如腰腹肉多、胯宽、腿短）→明确穿搭优化重点；
            2. **场景与目的**：场景（如职场面试、周末露营、闺蜜约会、冬季通勤）→确定穿搭正式度/功能性（如露营需耐磨、面试需专业）；目的（如“想显气质”“想藏肉”“想亮眼”）→锚定搭配核心方向；
            3. **偏好与客观条件**：风格偏好（如极简、甜辣、复古、日系）→统一单品风格调性；季节/气温（如夏季35℃、冬季-5℃）→匹配面料（棉麻/雪纺、加绒/羊毛）；预算范围（如单件200元内、整套800元内）→推荐对应价位单品类型（基础款/轻奢款）。


            ## 二、穿搭方案输出标准
            1. **开头呼应**：先总结用户关键信息，让方案更具针对性，示例：“从你的描述中了解到，你身高158cm、体重52kg，想改善腿短问题，周末要去公园露营（需要舒服方便活动），喜欢日系休闲风，所在城市现在25℃左右，单品预算100-300元，下面为你推荐适配方案～”
            2. **方案结构（分模块清晰呈现）**：
            - **整体风格定位**：一句话概括（如“日系休闲露营风——舒适不拖沓，兼顾活动便利性与拍照氛围感”）；
            - **核心单品清单**（每类单品需含“具体款式+颜色+适配理由+预算参考”）：
                - 上衣：例“宽松短款条纹T恤（白色+浅蓝条纹）——短款版型提高腰线，显腿长；宽松设计不贴腰，活动自在；预算80-120元”；
                - 下装：例“高腰直筒牛仔短裤（浅牛仔色，裤长到大腿中部）——高腰呼应短上衣，进一步拉长比例；直筒版型遮大腿肉，不挑腿型；预算150-200元”；
                - 鞋履：例“低帮帆布运动鞋（白色）——轻便防滑，适合露营走路；白色百搭，不抢上衣亮点；预算100-150元”；
                - 配饰（可选，需贴合场景）：例“草编渔夫帽（米色）——遮阳防晒，契合日系露营氛围；预算50-80元；帆布斜挎包（浅灰色）——容量大，能装手机、纸巾等露营小物，轻便不压肩；预算80-120元”；
            - **细节优化技巧**：结合用户需求补充“避坑点+加分项”，例“短裤选‘微阔腿’而非‘紧身’，避免显大腿粗；上衣塞进1/3到裤子里，比全塞更自然，还能精准露腰线；如果怕晒，可外搭一件轻薄的浅卡其色防晒衫，拉链拉到胸口，不闷热也不破坏搭配”；
            3. **结尾预留调整空间**：主动询问用户反馈，例“这个方案是否符合你的预期呀？如果想调整风格（比如更甜一点）、更换某类单品（比如不喜欢短裤），或者有其他需求，都可以告诉我，我再优化～”


            ## 三、特殊情况处理规则
            1. 若用户描述信息模糊（如只说“想穿去约会，喜欢好看的”）：先基于“约会场景”推荐通用温柔风方案，同时补充“如果能告诉我你的身高体重、所在季节，或者更具体的风格（比如甜妹/御姐），方案会更贴合你的需求哦～”；
            2. 若用户有“矛盾需求”（如“想穿紧身裙显身材，但怕显腰腹肉”）：优先平衡需求，例“推荐‘高腰收腰A字紧身裙（面料选有弹性的针织棉）’——高腰收腰凸显曲线，A字裙摆从腰腹下方散开，刚好遮住赘肉；颜色选深肤色/黑色，显瘦效果更好”；
            3. 若用户提及“小众场景”（如“汉服出游”“海边度假拍写真”）：需匹配场景专属单品，同时兼顾实用性，例“海边写真穿搭——上衣选短款露背吊带（浅粉色雪纺材质，带蕾丝花边），下装选高腰阔腿沙滩裤（白色薄款，裤脚带开叉），鞋履选银色夹脚凉鞋，配饰加珍珠项链+草帽；适配理由：浅粉显白，露背贴合海边氛围，阔腿裤防走光，雪纺面料风吹起来有飘逸感，拍照出片；预算参考：吊带80-120元，沙滩裤100-150元，凉鞋100-150元，配饰100-150元”。

            请根据以下人物描述与场景，严格遵循以上规则生成穿搭建议，确保每一条建议都有“用户需求支撑”，不推荐无关、不实用或不符合用户偏好的单品。"""            
            f"【人物外貌】{desc}\n\n"
            f"【场景】{scene}\n\n"
        )

        logger.bind(tag=TAG).info("OOTD 图像分析完成，已生成穿搭建议提示词")
        return ActionResponse(action=Action.REQLLM, result=guidance, response=None)

    except Exception as e:
        logger.bind(tag=TAG).error(f"ootd 处理异常: {e}")
        # 直接回复错误信息
        return ActionResponse(action=Action.RESPONSE, result=None, response=f"生成失败：{e}")
