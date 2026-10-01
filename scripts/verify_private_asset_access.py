"""Verify private denial and signed download integrity for one analysis asset."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mangrovision_db.config import get_settings
from mangrovision_db.storage import signed_download_url


def main() -> int:
    settings = get_settings()
    engine = create_engine(
        settings.database_url,
        connect_args={"sslmode": settings.db_sslmode} if settings.db_sslmode else {},
    )
    try:
        with engine.connect() as connection:
            asset = connection.execute(
                text(
                    "SELECT object_key, byte_size, sha256 FROM analysis_assets "
                    "WHERE lifecycle_state = 'ready' ORDER BY id LIMIT 1"
                )
            ).mappings().one_or_none()
        if asset is None:
            raise RuntimeError("No ready analysis asset exists to verify")

        unsigned_url = (
            f"{settings.s3_endpoint_url}/{quote(settings.s3_bucket, safe='')}"
            f"/{quote(asset['object_key'], safe='/')}"
        )
        unsigned_denied = False
        try:
            with urlopen(unsigned_url, timeout=30):
                pass
        except HTTPError as error:
            unsigned_denied = error.code in {400, 401, 403, 404}

        with urlopen(signed_download_url(asset["object_key"]), timeout=60) as response:
            payload = response.read()

        checks = {
            "UNSIGNED_ACCESS_DENIED": unsigned_denied,
            "SIGNED_DOWNLOAD_SIZE_OK": len(payload) == asset["byte_size"],
            "SIGNED_DOWNLOAD_CHECKSUM_OK": (
                hashlib.sha256(payload).hexdigest() == asset["sha256"]
            ),
        }
        for name, passed in checks.items():
            print(f"{name}={passed}")
        return 0 if all(checks.values()) else 1
    except Exception as error:
        print(f"Private asset verification failed: {error}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
