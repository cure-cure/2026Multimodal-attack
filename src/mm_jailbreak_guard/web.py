from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import tempfile
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .contracts import DetectionRequest
from .io import JsonlAuditLog
from .pipeline import DetectionPipeline


MAX_REQUEST_BYTES = 12 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
STATIC_DIR = Path(__file__).with_name("static")


def _looks_like_supported_image(payload: bytes, suffix: str) -> bool:
    signatures = {
        ".png": lambda data: data.startswith(b"\x89PNG\r\n\x1a\n"),
        ".jpg": lambda data: data.startswith(b"\xff\xd8\xff"),
        ".jpeg": lambda data: data.startswith(b"\xff\xd8\xff"),
        ".webp": lambda data: len(data) >= 12
        and data.startswith(b"RIFF")
        and data[8:12] == b"WEBP",
        ".bmp": lambda data: data.startswith(b"BM"),
    }
    return bool(payload) and signatures[suffix](payload)


def _parse_multipart(content_type: str, body: bytes) -> dict[str, dict[str, Any]]:
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: "
        + content_type.encode("ascii", errors="strict")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    if not message.is_multipart():
        raise ValueError("请求必须使用 multipart/form-data")
    fields: dict[str, dict[str, Any]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        fields[str(name)] = {
            "filename": part.get_filename(),
            "content_type": part.get_content_type(),
            "data": part.get_payload(decode=True) or b"",
        }
    return fields


def _field_text(fields: dict[str, dict[str, Any]], name: str) -> str:
    field = fields.get(name)
    if field is None:
        return ""
    return bytes(field["data"]).decode("utf-8", errors="replace").strip()


def make_handler(pipeline: DetectionPipeline, audit_log: JsonlAuditLog):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MMGuard/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[web] {self.address_string()} - {format % args}")

        def end_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob: data:; "
                "style-src 'self'; script-src 'self'; connect-src 'self'",
            )
            super().end_headers()

        def do_GET(self) -> None:
            if self.path == "/api/status":
                self._json(HTTPStatus.OK, pipeline.status())
                return
            if self.path.startswith("/api/records"):
                self._json(HTTPStatus.OK, {"records": audit_log.recent(10)})
                return
            if self.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            route = "/index.html" if self.path == "/" else self.path.split("?", 1)[0]
            file_map = {
                "/index.html": STATIC_DIR / "index.html",
                "/style.css": STATIC_DIR / "style.css",
                "/app.js": STATIC_DIR / "app.js",
            }
            target = file_map.get(route)
            if target is None or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = target.read_bytes()
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            if self.path != "/api/detect":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "Content-Length 必须是整数"})
                return
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "请求为空或超过 12 MB"},
                )
                return
            temp_path: str | None = None
            try:
                fields = _parse_multipart(
                    self.headers.get("Content-Type", ""), self.rfile.read(length)
                )
                sample_id = _field_text(fields, "sample_id")
                user_text = _field_text(fields, "user_text")
                if not sample_id:
                    raise ValueError("样本编号不能为空")
                if len(user_text) > 5000:
                    raise ValueError("用户文字不能超过 5000 个字符")

                image_field = fields.get("image")
                image_name: str | None = None
                if image_field and image_field.get("filename"):
                    image_name = Path(str(image_field["filename"])).name
                    suffix = Path(image_name).suffix.casefold()
                    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
                        raise ValueError("图片格式仅支持 PNG、JPG、JPEG、WEBP、BMP")
                    image_bytes = bytes(image_field["data"])
                    if len(image_bytes) > 10 * 1024 * 1024:
                        raise ValueError("图片不能超过 10 MB")
                    if not _looks_like_supported_image(image_bytes, suffix):
                        raise ValueError("图片内容与扩展名不匹配或文件已损坏")
                    with tempfile.NamedTemporaryFile(
                        prefix="mmguard_", suffix=suffix, delete=False
                    ) as handle:
                        handle.write(image_bytes)
                        temp_path = handle.name

                request = DetectionRequest(
                    sample_id=sample_id,
                    user_text=user_text,
                    image_path=temp_path,
                    image_name=image_name,
                )
                report = pipeline.detect(request, mode=_field_text(fields, "mode") or "full")
                payload = report.as_dict(include_upstream=True)
                audit_log.append(
                    {
                        "sample_id": sample_id,
                        "image_name": image_name,
                        "user_text_sha256": hashlib.sha256(
                            user_text.encode("utf-8")
                        ).hexdigest(),
                        "result": report.as_prediction_record(),
                    }
                )
                self._json(HTTPStatus.OK, payload)
            except (ValueError, UnicodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"系统处理失败：{type(exc).__name__}: {exc}"},
                )
            finally:
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except FileNotFoundError:
                        pass

        def _json(self, status: HTTPStatus, data: dict[str, Any]) -> None:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def run_server(
    pipeline: DetectionPipeline,
    host: str,
    port: int,
    records_path: str | Path,
) -> None:
    audit_log = JsonlAuditLog(records_path)
    server = ThreadingHTTPServer((host, port), make_handler(pipeline, audit_log))
    print(f"本地演示已启动：http://{host}:{port}")
    print("按 Ctrl+C 停止。检测记录不保存原始图片或原始文字。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止本地演示。")
    finally:
        server.server_close()
