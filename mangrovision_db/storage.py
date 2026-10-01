"""Private S3-compatible storage for analysis assets."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Iterable

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.client import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    ConnectionClosedError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from PIL import Image, UnidentifiedImageError

from .config import get_settings

_DATA_URL = re.compile(
    r"^data:(image/(?:jpeg|png|webp));base64,(.+)$", re.IGNORECASE | re.DOTALL
)
_EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
_PREVIEW_MAX_SIZE = (1280, 1280)
_MULTIPART_THRESHOLD_BYTES = 6 * 1024 * 1024
_MULTIPART_CONFIG = TransferConfig(
    multipart_threshold=_MULTIPART_THRESHOLD_BYTES,
    multipart_chunksize=_MULTIPART_THRESHOLD_BYTES,
    max_concurrency=2,
    use_threads=True,
)
_RETRYABLE_UPLOAD_ERRORS = (
    ConnectTimeoutError,
    ConnectionClosedError,
    EndpointConnectionError,
    ReadTimeoutError,
)
_SIGNED_URL_CACHE: dict[tuple[str, str, int], tuple[str, float]] = {}
_SIGNED_URL_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class StoredAsset:
    kind: str
    object_key: str
    content_type: str
    byte_size: int
    sha256: str


@lru_cache(maxsize=1)
def s3_client():
    settings = get_settings()
    if not settings.s3_access_key_id or not settings.s3_secret_access_key:
        raise RuntimeError("S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY must be configured")
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path" if settings.s3_force_path_style else "virtual"},
            connect_timeout=15,
            read_timeout=180,
            tcp_keepalive=True,
            retries={"max_attempts": 4, "mode": "standard"},
        ),
    )


def decode_image_data_url(data_url: str) -> tuple[bytes, str]:
    match = _DATA_URL.match((data_url or "").strip())
    if not match:
        raise ValueError("Expected a base64 JPEG, PNG, or WebP data URL")
    content_type = match.group(1).lower()
    try:
        payload = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Image data URL contains invalid base64") from error
    if not payload:
        raise ValueError("Image asset is empty")
    return payload, content_type


def _is_retryable_upload_error(error: BaseException) -> bool:
    if isinstance(error, _RETRYABLE_UPLOAD_ERRORS):
        return True
    if not isinstance(error, ClientError):
        return False
    response = error.response or {}
    error_detail = response.get("Error") or {}
    code = str(error_detail.get("Code") or "").strip()
    status = int((response.get("ResponseMetadata") or {}).get("HTTPStatusCode") or 0)
    return (
        not code
        or status in {408, 429}
        or status >= 500
        or code in {"InternalError", "RequestTimeout", "ServiceUnavailable", "SlowDown"}
    )


def _upload_error_summary(error: BaseException) -> str:
    if isinstance(error, ClientError):
        response = error.response or {}
        error_detail = response.get("Error") or {}
        code = str(error_detail.get("Code") or "unknown")
        message = str(error_detail.get("Message") or "").strip()
        status = int((response.get("ResponseMetadata") or {}).get("HTTPStatusCode") or 0)
        request_id = str(
            (response.get("ResponseMetadata") or {}).get("RequestId") or ""
        ).strip()
        parts = [f"HTTP {status or 'unknown'}", f"code {code}"]
        if message:
            parts.append(f"message {message}")
        if request_id:
            parts.append(f"request {request_id}")
        return ", ".join(parts)
    return type(error).__name__


def _put_payload(
    object_key: str,
    payload: bytes,
    content_type: str,
    digest: str,
    kind: str,
) -> None:
    settings = get_settings()
    extra_args = {
        "ContentType": content_type,
        "CacheControl": "private, max-age=3600",
        "Metadata": {"sha256": digest, "kind": kind},
    }
    client = s3_client()
    if len(payload) >= _MULTIPART_THRESHOLD_BYTES:
        client.upload_fileobj(
            io.BytesIO(payload),
            settings.s3_bucket,
            object_key,
            ExtraArgs=extra_args,
            Config=_MULTIPART_CONFIG,
        )
    else:
        client.put_object(
            Bucket=settings.s3_bucket,
            Key=object_key,
            Body=payload,
            **extra_args,
        )


def _upload_bytes(prefix: str, kind: str, payload: bytes, content_type: str) -> StoredAsset:
    clean_prefix = prefix.strip("/") or str(uuid.uuid4())
    extension = _EXTENSIONS[content_type]
    object_key = f"analyses/{clean_prefix}/{kind}.{extension}"
    digest = hashlib.sha256(payload).hexdigest()
    for attempt in range(3):
        try:
            _put_payload(object_key, payload, content_type, digest, kind)
            break
        except (BotoCoreError, ClientError) as error:
            if attempt >= 2 or not _is_retryable_upload_error(error):
                summary = _upload_error_summary(error)
                size_mib = len(payload) / (1024 * 1024)
                upload_mode = (
                    "multipart"
                    if len(payload) >= _MULTIPART_THRESHOLD_BYTES
                    else "single-request"
                )
                raise RuntimeError(
                    "Private Storage upload failed "
                    f"for {kind} ({size_mib:.1f} MiB, {upload_mode}) after "
                    f"{attempt + 1} attempt(s) ({summary})."
                ) from error
            time.sleep(0.75 * (2 ** attempt))
    return StoredAsset(
        kind=kind,
        object_key=object_key,
        content_type=content_type,
        byte_size=len(payload),
        sha256=digest,
    )


def upload_data_url(prefix: str, kind: str, data_url: str) -> StoredAsset:
    payload, content_type = decode_image_data_url(data_url)
    return _upload_bytes(prefix, kind, payload, content_type)


def _webp_preview(payload: bytes) -> bytes:
    """Create a compact display copy while retaining the full-resolution asset."""
    with Image.open(io.BytesIO(payload)) as source:
        source.thumbnail(_PREVIEW_MAX_SIZE, Image.Resampling.LANCZOS)
        if source.mode in {"RGBA", "LA"} or (
            source.mode == "P" and "transparency" in source.info
        ):
            rgba = source.convert("RGBA")
            image = Image.new("RGB", rgba.size, "white")
            image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            image = source.convert("RGB")
        output = io.BytesIO()
        image.save(output, format="WEBP", quality=78, method=4)
        return output.getvalue()


def upload_analysis_data_urls(
    original_data_url: str | None,
    visualization_data_url: str | None,
    prefix: str | None = None,
) -> list[StoredAsset]:
    prefix = prefix or str(uuid.uuid4())
    stored: list[StoredAsset] = []
    try:
        if original_data_url:
            original_payload, original_content_type = decode_image_data_url(original_data_url)
            stored.append(
                _upload_bytes(prefix, "original", original_payload, original_content_type)
            )
            try:
                stored.append(
                    _upload_bytes(
                        prefix,
                        "original_preview",
                        _webp_preview(original_payload),
                        "image/webp",
                    )
                )
            except (OSError, UnidentifiedImageError):
                # The full asset is still valid and history can fall back to it.
                pass
        if visualization_data_url:
            visualization_payload, visualization_content_type = decode_image_data_url(
                visualization_data_url
            )
            stored.append(
                _upload_bytes(
                    prefix,
                    "visualization",
                    visualization_payload,
                    visualization_content_type,
                )
            )
            try:
                stored.append(
                    _upload_bytes(
                        prefix,
                        "visualization_preview",
                        _webp_preview(visualization_payload),
                        "image/webp",
                    )
                )
            except (OSError, UnidentifiedImageError):
                pass
        return stored
    except Exception:
        delete_assets(stored)
        raise


def signed_download_url(object_key: str) -> str:
    settings = get_settings()
    ttl = settings.s3_presigned_url_ttl_seconds
    cache_key = (settings.s3_bucket, object_key, ttl)
    now = time.monotonic()
    with _SIGNED_URL_CACHE_LOCK:
        cached = _SIGNED_URL_CACHE.get(cache_key)
        if cached and cached[1] > now:
            return cached[0]

    url = s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": object_key},
        ExpiresIn=ttl,
    )
    # Reusing the exact signed URL lets the browser reuse its private cache.
    # Keep a safety margin so this process never returns a nearly-expired URL.
    cache_seconds = max(1, ttl - min(60, max(1, ttl // 4)))
    with _SIGNED_URL_CACHE_LOCK:
        expired = [key for key, (_, expires_at) in _SIGNED_URL_CACHE.items() if expires_at <= now]
        for key in expired:
            _SIGNED_URL_CACHE.pop(key, None)
        if len(_SIGNED_URL_CACHE) >= 1024:
            _SIGNED_URL_CACHE.pop(next(iter(_SIGNED_URL_CACHE)))
        _SIGNED_URL_CACHE[cache_key] = (url, now + cache_seconds)
    return url


def delete_assets(assets: Iterable[StoredAsset | str]) -> None:
    for asset in assets:
        object_key = asset.object_key if isinstance(asset, StoredAsset) else asset
        try:
            delete_object(object_key)
        except (BotoCoreError, ClientError):
            continue


def delete_object(object_key: str) -> None:
    settings = get_settings()
    s3_client().delete_object(Bucket=settings.s3_bucket, Key=object_key)


def stale_object_keys(prefix: str, older_than: datetime) -> list[str]:
    """List object keys below a prefix whose last modification is older."""
    cutoff = older_than
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    settings = get_settings()
    paginator = s3_client().get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            modified = item.get("LastModified")
            if modified is not None and modified.astimezone(timezone.utc) < cutoff:
                keys.append(str(item["Key"]))
    return keys


def storage_ready() -> bool:
    try:
        s3_client().head_bucket(Bucket=get_settings().s3_bucket)
        return True
    except (BotoCoreError, ClientError, RuntimeError):
        return False
