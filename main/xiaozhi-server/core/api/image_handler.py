import os
import json
import time
import uuid
from typing import Optional
from aiohttp import web
from core.api.base_handler import BaseHandler
from core.utils.util import is_valid_image_file
from config.config_loader import get_project_dir

TAG = __name__

# 设置最大文件大小为5MB
MAX_FILE_SIZE = 5 * 1024 * 1024


class ImageHandler(BaseHandler):
    """
    无需认证的图片上传处理器：
    - 接收 multipart/form-data 的图片文件
    - 校验文件大小与格式
    - 生成不重复的文件名并保存到项目 data/ 目录
    - 返回保存后的文件名与相对路径
    """

    def __init__(self, config: dict):
        super().__init__(config)

    def _create_error_response(self, message: str) -> dict:
        return {"success": False, "message": message}

    def _detect_ext(self, data: bytes) -> Optional[str]:
        """根据文件魔数判断图片扩展名"""
        signatures = {
            b"\xff\xd8\xff": "jpg",  # JPEG
            b"\x89PNG\r\n\x1a\n": "png",  # PNG
            b"GIF87a": "gif",  # GIF
            b"GIF89a": "gif",  # GIF
            b"BM": "bmp",  # BMP
            b"II*\x00": "tiff",  # TIFF
            b"MM\x00*": "tiff",  # TIFF
            b"RIFF": "webp",  # WEBP
        }
        for sig, ext in signatures.items():
            if data.startswith(sig):
                return ext
        return None

    def _unique_filename(self, original_filename: Optional[str], data: bytes) -> str:
        # 优先使用上传文件的扩展名
        ext = None
        if original_filename and "." in original_filename:
            ext = original_filename.rsplit(".", 1)[-1].lower()
        if not ext:
            ext = self._detect_ext(data) or "bin"
        unique = f"image_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.{ext}"
        return unique

    async def handle_post(self, request):
        """处理图片上传请求（无需认证）"""
        response = None
        try:
            reader = await request.multipart()
            if reader is None:
                raise ValueError("请求格式错误，应为multipart/form-data")

            # 查找第一个带文件名的字段，或名为'image'的字段
            field = await reader.next()
            target_field = None
            while field is not None:
                if getattr(field, "filename", None):
                    target_field = field
                    break
                if getattr(field, "name", None) == "image":
                    target_field = field
                    break
                field = await reader.next()

            if target_field is None:
                raise ValueError("未找到图片文件，请使用表单字段'image'上传文件")

            file_bytes = await target_field.read()
            if not file_bytes:
                raise ValueError("图片数据为空")

            if len(file_bytes) > MAX_FILE_SIZE:
                raise ValueError(
                    f"图片大小超过限制，最大允许{MAX_FILE_SIZE/1024/1024}MB"
                )

            if not is_valid_image_file(file_bytes):
                raise ValueError(
                    "不支持的文件格式，请上传有效的图片文件（支持JPEG、PNG、GIF、BMP、TIFF、WEBP格式）"
                )

            # 生成文件名并保存到 data/ 目录
            project_dir = get_project_dir()
            data_dir = os.path.join(project_dir, "data")
            os.makedirs(data_dir, exist_ok=True)

            filename = self._unique_filename(getattr(target_field, "filename", None), file_bytes)
            save_path = os.path.join(data_dir, filename)
            with open(save_path, "wb") as f:
                f.write(file_bytes)
                
            self.logger.bind(tag=TAG).info(f"图片上传成功: {save_path}")

            return_json = {
                "success": True,
                "filename": filename,
                "path": f"data/{filename}",
            }
            response = web.Response(
                text=json.dumps(return_json, separators=(",", ":")),
                content_type="application/json",
                status=200,
            )
        except ValueError as e:
            self.logger.error(f"Image upload error: {e}")
            return_json = self._create_error_response(str(e))
            response = web.Response(
                text=json.dumps(return_json, separators=(",", ":")),
                content_type="application/json",
                status=400,
            )
        except Exception as e:
            self.logger.error(f"Image upload exception: {e}")
            return_json = self._create_error_response("服务器内部错误")
            response = web.Response(
                text=json.dumps(return_json, separators=(",", ":")),
                content_type="application/json",
                status=500,
            )
        finally:
            if response is not None:
                self._add_cors_headers(response)
            return response

    async def handle_get(self, request):
        """简单健康检查"""
        response = web.Response(text="Image upload endpoint is running", content_type="text/plain")
        self._add_cors_headers(response)
        return response
