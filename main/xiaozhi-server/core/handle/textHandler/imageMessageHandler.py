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
            status = msg_json.get("status")
            question = msg_json.get("question", "ootd")

            if "ootd" not in question:
                await startToChat(conn, "您的提问不正确，我无法回答您")
                return
            else:
                question = "穿搭建议"    
                await startToChat(conn, f"{question}：\n图片路径:{file_name}\n,场景:{status}")
                return
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
