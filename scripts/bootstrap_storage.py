"""Create and probe the configured private S3-compatible analysis bucket."""

from __future__ import annotations

import argparse
import hashlib
import sys
import uuid
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mangrovision_db.config import get_settings
from mangrovision_db.storage import s3_client


def is_missing_bucket(error: ClientError) -> bool:
    response = error.response or {}
    code = str((response.get("Error") or {}).get("Code") or "")
    status = int((response.get("ResponseMetadata") or {}).get("HTTPStatusCode") or 0)
    return code in {"404", "NoSuchBucket", "NotFound"} or status == 404


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the bucket when it does not exist (new buckets are private by default).",
    )
    args = parser.parse_args()
    settings = get_settings()

    try:
        client = s3_client()
        try:
            client.head_bucket(Bucket=settings.s3_bucket)
            created = False
        except ClientError as error:
            if not is_missing_bucket(error) or not args.create:
                raise
            client.create_bucket(Bucket=settings.s3_bucket)
            client.head_bucket(Bucket=settings.s3_bucket)
            created = True

        payload = b"MangroVision private storage readiness probe\n"
        object_key = f"readiness/{uuid.uuid4()}.txt"
        checksum = hashlib.sha256(payload).hexdigest()
        try:
            client.put_object(
                Bucket=settings.s3_bucket,
                Key=object_key,
                Body=payload,
                ContentType="text/plain",
                CacheControl="no-store",
                Metadata={"sha256": checksum},
            )
            head = client.head_object(Bucket=settings.s3_bucket, Key=object_key)
            if int(head.get("ContentLength", -1)) != len(payload):
                raise RuntimeError("Storage readiness object size did not match")
            if (head.get("Metadata") or {}).get("sha256") != checksum:
                raise RuntimeError("Storage readiness object checksum metadata did not match")
        finally:
            client.delete_object(Bucket=settings.s3_bucket, Key=object_key)
    except (BotoCoreError, ClientError, RuntimeError) as error:
        print(f"Storage bootstrap failed: {error}", file=sys.stderr)
        return 1

    action = "created and verified" if created else "verified"
    print(f"Private analysis bucket '{settings.s3_bucket}' {action}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
