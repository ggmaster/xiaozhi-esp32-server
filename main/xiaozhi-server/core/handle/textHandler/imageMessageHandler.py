import json
import base64
import os
from typing import Dict, Any

from core.handle.receiveAudioHandle import handleAudioMessage, startToChat
from core.handle.textMessageHandler import TextMessageHandler
from core.handle.textMessageType import TextMessageType
from core.utils.util import is_valid_image_file
from core.utils.vllm import create_instance
from config.config_loader import get_project_dir
TAG = __name__


class ImageTextMessageHandler(TextMessageHandler):
    """IMAGE 消息处理器

    预期消息格式：
    {
        "type": "image",
        "file_name": "image_xxx.jpg",  # 位于项目 data/ 目录下的文件名
        "question": "请描述下图片"  # 可选
    }
    """

    @property
    def message_type(self) -> TextMessageType:
        return TextMessageType.IMAGE

    async def handle(self, conn, msg_json: Dict[str, Any]) -> None:
        try:
            file_name = msg_json.get("file_name")
            status = msg_json.get("status", "逛街")
            question = msg_json.get("question", "ootd")

            if "ootd" not in question:
                await startToChat(conn, "您的提问不正确，我无法回答您")
                return
            else:
                question = "穿搭建议"    
                await startToChat(conn, f"{question}：\n,file_name:{file_name}\n,status:{status}")
                return
            
            if "ootd" not in question:
                await startToChat(conn, "您的提问不正确，我无法回答您")
                return
            else:
                question = "请对图片中的人物进行详细描述，只关注图片中人物的细节，不包括背景信息"    


            if not file_name or not isinstance(file_name, str):
                await conn.websocket.send(
                    json.dumps(
                        {
                            "type": "stt",
                            "status": "error",
                            "message": "缺少有效的 file_name 字段（data/目录下的图片文件名）",
                        }
                    )
                )
                return

            # 从本地 data 目录读取图片为字节
            project_dir = get_project_dir()
            image_path = os.path.join(project_dir, "data", file_name)
            if not os.path.exists(image_path):
                await conn.websocket.send(
                    json.dumps(
                        {
                            "type": "stt",
                            "status": "error",
                            "message": f"未找到指定文件：{file_name}（应位于 data/ 目录）",
                        }
                    )
                )
                return

            with open(image_path, "rb") as f:
                image_data = f.read()
            if not image_data:
                await conn.websocket.send(
                    json.dumps(
                        {
                            "type": "stt",
                            "status": "error",
                            "message": "读取图片失败或内容为空",
                        }
                    )
                )
                return
            # 简单大小限制：5MB
            if len(image_data) > 5 * 1024 * 1024:
                await conn.websocket.send(
                    json.dumps(
                        {
                            "type": "stt",
                            "status": "error",
                            "message": "图片大小超过限制（最大5MB）",
                        }
                    )
                )
                return

            if not is_valid_image_file(image_data):
                await conn.websocket.send(
                    json.dumps(
                        {
                            "type": "llm",
                            "status": "error",
                            "message": "不支持的文件格式，请提供有效的图片文件（JPEG、PNG、GIF、BMP、TIFF、WEBP）",
                        }
                    )
                )
                return

            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # 使用当前连接已生效的配置选择 VLLM
            current_config = conn.config
            select_vllm_module = current_config["selected_module"].get("VLLM")
            if not select_vllm_module:
                await conn.websocket.send(
                    json.dumps(
                        {
                            "type": "llm",
                            "status": "error",
                            "message": "您还未设置默认的视觉分析模块",
                        }
                    )
                )
                return

            vllm_type = (
                select_vllm_module
                if "type" not in current_config["VLLM"][select_vllm_module]
                else current_config["VLLM"][select_vllm_module]["type"]
            )
            vllm = create_instance(vllm_type, current_config["VLLM"][select_vllm_module])
            
            conn.logger.bind(tag=TAG).info(f"开始调用视觉模型分析图片：{question}")
            result = vllm.response(question, image_base64)
            conn.logger.bind(tag=TAG).info(
                f"视觉模型返回结果：{result}"
            )
            result = "根据图片里面的人物描述："+result+",评价一下穿搭，并给出穿搭建议"
            await startToChat(conn, result)
            # await conn.websocket.send(
            #     json.dumps(
            #         {
            #             "type": "image",
            #             "status": "success",
            #             "response": result,
            #         }
            #     )
            # )
        except Exception as e:
            conn.logger.bind(tag=TAG).error(f"异常：{str(e)}")
            # 统一异常处理
            await conn.websocket.send(
                json.dumps(
                    {
                        "type": "llm",
                        "status": "error",
                        "message": f"处理图片解析时发生错误: {str(e)}",
                    }
                )
            )
