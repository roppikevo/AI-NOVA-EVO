from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ResourceSnapshot:
    timestamp: float
    cpu_count: int
    ram_total_bytes: int
    ram_available_bytes: int
    ram_used_bytes: int
    disk_free_bytes: int
    gpu_available: bool = False
    gpu_name: str | None = None
    gpu_vram_total_bytes: int | None = None
    gpu_vram_allocated_bytes: int | None = None
    gpu_vram_reserved_bytes: int | None = None
    gpu_vram_free_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_count": self.cpu_count,
            "ram_total_bytes": self.ram_total_bytes,
            "ram_available_bytes": self.ram_available_bytes,
            "ram_used_bytes": self.ram_used_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_vram_total_bytes": self.gpu_vram_total_bytes,
            "gpu_vram_allocated_bytes": self.gpu_vram_allocated_bytes,
            "gpu_vram_reserved_bytes": self.gpu_vram_reserved_bytes,
            "gpu_vram_free_bytes": self.gpu_vram_free_bytes,
        }


@dataclass
class ResourceLimits:
    max_vram_bytes: int | None = None
    max_ram_bytes: int | None = None
    max_cpu_threads: int | None = None
    max_runtime_seconds: int | None = None
    max_storage_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_vram_bytes": self.max_vram_bytes,
            "max_ram_bytes": self.max_ram_bytes,
            "max_cpu_threads": self.max_cpu_threads,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_storage_bytes": self.max_storage_bytes,
        }


class ResourceManager:
    """Hardware/resource monitoring and budget enforcement."""

    def __init__(
        self,
        project_root: str | Path = ".",
        policy_path: str | Path = "evo/engine/agent_policy.json",
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.policy_path = (
            self.project_root / policy_path
        ).resolve()

        self.limits = self._load_limits()

    def _load_policy(self) -> dict[str, Any]:
        if not self.policy_path.exists():
            return {}

        return json.loads(
            self.policy_path.read_text(
                encoding="utf-8"
            )
        )

    def _load_limits(self) -> ResourceLimits:
        policy = self._load_policy()
        limits = policy.get("limits", {})

        return ResourceLimits(
            max_vram_bytes=self._optional_int(
                limits.get("max_vram_bytes")
            ),
            max_ram_bytes=self._optional_int(
                limits.get("max_ram_bytes")
            ),
            max_cpu_threads=self._optional_int(
                limits.get("max_cpu_threads")
            ),
            max_runtime_seconds=self._optional_int(
                limits.get("max_candidate_runtime_seconds")
            ),
            max_storage_bytes=self._optional_int(
                limits.get("max_storage_bytes")
            ),
        )

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None

        return int(value)

    @staticmethod
    def _read_meminfo() -> dict[str, int]:
        result: dict[str, int] = {}

        try:
            text = Path(
                "/proc/meminfo"
            ).read_text(
                encoding="utf-8"
            )
        except OSError:
            return result

        for line in text.splitlines():
            parts = line.split()

            if len(parts) < 2:
                continue

            key = parts[0].rstrip(":")

            try:
                value = int(parts[1])
            except ValueError:
                continue

            # /proc/meminfo values are kB.
            result[key] = value * 1024

        return result

    def snapshot(self) -> ResourceSnapshot:
        meminfo = self._read_meminfo()

        total_ram = meminfo.get(
            "MemTotal",
            0,
        )

        available_ram = meminfo.get(
            "MemAvailable",
            meminfo.get("MemFree", 0),
        )

        used_ram = max(
            0,
            total_ram - available_ram,
        )

        disk = shutil.disk_usage(
            self.project_root
        )

        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_count=os.cpu_count() or 1,
            ram_total_bytes=total_ram,
            ram_available_bytes=available_ram,
            ram_used_bytes=used_ram,
            disk_free_bytes=disk.free,
        )

        self._add_gpu_metrics(snapshot)

        return snapshot

    @staticmethod
    def _add_gpu_metrics(
        snapshot: ResourceSnapshot,
    ) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                return

            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(
                device
            )

            total = props.total_memory
            allocated = torch.cuda.memory_allocated(
                device
            )
            reserved = torch.cuda.memory_reserved(
                device
            )

            snapshot.gpu_available = True
            snapshot.gpu_name = props.name
            snapshot.gpu_vram_total_bytes = total
            snapshot.gpu_vram_allocated_bytes = allocated
            snapshot.gpu_vram_reserved_bytes = reserved
            snapshot.gpu_vram_free_bytes = max(
                0,
                total - reserved,
            )

        except Exception:
            # GPU monitoring must never crash the experiment controller.
            snapshot.gpu_available = False

    def check_limits(
        self,
        snapshot: ResourceSnapshot | None = None,
    ) -> dict[str, Any]:

        if snapshot is None:
            snapshot = self.snapshot()

        violations: list[str] = []

        if (
            self.limits.max_ram_bytes is not None
            and snapshot.ram_used_bytes
            > self.limits.max_ram_bytes
        ):
            violations.append(
                "RAM usage exceeds configured limit"
            )

        if (
            self.limits.max_vram_bytes is not None
            and snapshot.gpu_vram_reserved_bytes is not None
            and snapshot.gpu_vram_reserved_bytes
            > self.limits.max_vram_bytes
        ):
            violations.append(
                "VRAM usage exceeds configured limit"
            )

        if (
            self.limits.max_cpu_threads is not None
            and snapshot.cpu_count
            > self.limits.max_cpu_threads
        ):
            violations.append(
                "CPU thread count exceeds configured limit"
            )

        if (
            self.limits.max_storage_bytes is not None
            and snapshot.disk_free_bytes
            < self.limits.max_storage_bytes
        ):
            violations.append(
                "Available storage is below configured limit"
            )

        return {
            "allowed": not violations,
            "violations": violations,
            "limits": self.limits.to_dict(),
            "snapshot": snapshot.to_dict(),
        }

    def can_start_experiment(
        self,
        estimated_vram_bytes: int | None = None,
        estimated_ram_bytes: int | None = None,
        estimated_cpu_threads: int | None = None,
    ) -> tuple[bool, list[str]]:

        snapshot = self.snapshot()
        reasons: list[str] = []

        if (
            estimated_vram_bytes is not None
            and snapshot.gpu_vram_free_bytes is not None
            and estimated_vram_bytes
            > snapshot.gpu_vram_free_bytes
        ):
            reasons.append(
                "Estimated VRAM requirement exceeds "
                "currently available VRAM"
            )

        if (
            estimated_ram_bytes is not None
            and estimated_ram_bytes
            > snapshot.ram_available_bytes
        ):
            reasons.append(
                "Estimated RAM requirement exceeds "
                "currently available RAM"
            )

        if (
            estimated_cpu_threads is not None
            and self.limits.max_cpu_threads is not None
            and estimated_cpu_threads
            > self.limits.max_cpu_threads
        ):
            reasons.append(
                "Requested CPU threads exceed configured limit"
            )

        return not reasons, reasons

    def runtime_allowed(
        self,
        elapsed_seconds: float,
    ) -> bool:

        if self.limits.max_runtime_seconds is None:
            return True

        return (
            elapsed_seconds
            <= self.limits.max_runtime_seconds
        )


def self_test() -> None:
    with __import__("tempfile").TemporaryDirectory() as tmp:
        root = Path(tmp)

        policy = {
            "limits": {
                "max_candidate_runtime_seconds": 300,
                "max_cpu_threads": 16,
            }
        }

        policy_path = (
            root / "evo/engine/agent_policy.json"
        )

        policy_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        policy_path.write_text(
            json.dumps(policy),
            encoding="utf-8",
        )

        manager = ResourceManager(
            project_root=root
        )

        snapshot = manager.snapshot()

        assert snapshot.cpu_count >= 1
        assert snapshot.ram_total_bytes >= 0
        assert snapshot.ram_available_bytes >= 0
        assert snapshot.disk_free_bytes >= 0

        result = manager.check_limits(
            snapshot
        )

        assert "allowed" in result
        assert "violations" in result
        assert "snapshot" in result

        allowed, reasons = (
            manager.can_start_experiment(
                estimated_cpu_threads=1
            )
        )

        assert allowed
        assert reasons == []

        assert manager.runtime_allowed(1.0)
        assert not manager.runtime_allowed(301.0)

    print("RESOURCE MANAGER SELFTEST: PASSED")


if __name__ == "__main__":
    self_test()
