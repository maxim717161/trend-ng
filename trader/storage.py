"""Хранилище состояния и логов: Яндекс Object Storage (S3) или локальная папка для разработки/тестов."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


class LocalStorage:
    def __init__(self, root: str):
        self.root = Path(root)

    def get(self, key: str) -> Optional[bytes]:
        p = self.root / key
        return p.read_bytes() if p.exists() else None

    def put(self, key: str, data: bytes) -> None:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file())


class S3Storage:
    def __init__(self, bucket: str, endpoint: str, access_key: str | None = None, secret_key: str | None = None):
        import boto3

        self.bucket = bucket
        self.s3 = boto3.client(
            "s3", endpoint_url=endpoint, region_name="ru-central1",
            aws_access_key_id=access_key or os.environ.get("S3_ACCESS_KEY_ID"),
            aws_secret_access_key=secret_key or os.environ.get("S3_SECRET_ACCESS_KEY"),
        )

    def get(self, key: str) -> Optional[bytes]:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def put(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)

    def list(self, prefix: str) -> list[str]:
        keys, token = [], None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            r = self.s3.list_objects_v2(**kw)
            keys += [o["Key"] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                return sorted(keys)
            token = r["NextContinuationToken"]


class Store:
    """Ключи с префиксом: state.json, control/halt|flatten, logs/YYYY-MM-DD/<время>_<n>.jsonl."""

    def __init__(self, backend, prefix: str):
        self.b, self.prefix = backend, prefix

    def load_state(self) -> dict:
        raw = self.b.get(self.prefix + "state.json")
        return json.loads(raw) if raw else {}

    def save_state(self, state: dict) -> None:
        self.b.put(self.prefix + "state.json", json.dumps(state, ensure_ascii=False, indent=1).encode())

    def control(self) -> Optional[str]:
        if self.b.get(self.prefix + "control/halt") is not None:
            return "halt"
        if self.b.get(self.prefix + "control/flatten") is not None:
            return "flatten"
        return None

    def write_events(self, ts_msk: str, events: list[dict]) -> None:
        if not events:
            return
        key = f"{self.prefix}logs/{ts_msk[:10]}/{ts_msk[11:19].replace(':', '')}_{len(events)}.jsonl"
        self.b.put(key, "\n".join(json.dumps(e, ensure_ascii=False) for e in events).encode())
