from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_AUTHORITY_FILE = Path(
    "/etc/ai/nova-evo/authority.json"
)

DEFAULT_CHECKSUM_FILE = Path(
    "/etc/ai/nova-evo/authority.json.sha256"
)

DEFAULT_CREATOR_AUTH_FILE = Path(
    "/etc/ai/nova-evo/creator.auth"
)


class AuthorityError(PermissionError):
    """Raised when an authority-controlled action is denied."""


class AuthorityManager:
    """
    Runtime authority boundary for NOVA-EVO.

    The policy and creator credential file live outside the project.
    NOVA-EVO may read them, but the agent must not modify them.

    Creator authentication is intentionally external to the model:
        username + password
            -> credential verification
            -> CREATOR-PRIMARY identity
            -> authority decision
    """

    def __init__(
        self,
        authority_file: str | Path = DEFAULT_AUTHORITY_FILE,
        checksum_file: str | Path = DEFAULT_CHECKSUM_FILE,
        creator_auth_file: str | Path = DEFAULT_CREATOR_AUTH_FILE,
        actor_id: str = "NOVA-EVO",
    ) -> None:
        self.authority_file = Path(
            authority_file
        ).resolve()

        self.checksum_file = Path(
            checksum_file
        ).resolve()

        self.creator_auth_file = Path(
            creator_auth_file
        ).resolve()

        self.actor_id = actor_id.strip()

        if not self.actor_id:
            raise ValueError(
                "actor_id cannot be empty"
            )

    def load(self) -> dict[str, Any]:
        if not self.authority_file.exists():
            raise AuthorityError(
                f"Authority policy missing: "
                f"{self.authority_file}"
            )

        try:
            policy = json.loads(
                self.authority_file.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError as exc:
            raise AuthorityError(
                f"Invalid authority policy: {exc}"
            ) from exc

        if not isinstance(policy, dict):
            raise AuthorityError(
                "Authority policy must be a JSON object."
            )

        return policy

    def verify_integrity(self) -> bool:
        if not self.checksum_file.exists():
            raise AuthorityError(
                f"Authority checksum missing: "
                f"{self.checksum_file}"
            )

        data = self.authority_file.read_bytes()

        calculated = hashlib.sha256(
            data
        ).hexdigest()

        expected_line = (
            self.checksum_file
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        if not expected_line:
            raise AuthorityError(
                "Authority checksum is empty."
            )

        expected = expected_line.split()[0]

        if calculated != expected:
            raise AuthorityError(
                "Authority policy checksum mismatch."
            )

        return True

    def policy(self) -> dict[str, Any]:
        self.verify_integrity()
        return self.load()

    def _load_creator_auth(self) -> dict[str, str]:
        if not self.creator_auth_file.exists():
            raise AuthorityError(
                f"Creator credential file missing: "
                f"{self.creator_auth_file}"
            )

        values: dict[str, str] = {}

        for raw_line in self.creator_auth_file.read_text(
            encoding="utf-8"
        ).splitlines():
            line = raw_line.strip()

            if not line or "=" not in line:
                continue

            key, value = line.split(
                "=",
                1,
            )

            values[key.strip()] = value.strip()

        required = {
            "version",
            "username",
            "salt",
            "hash",
            "algorithm",
            "iterations",
        }

        missing = required - values.keys()

        if missing:
            raise AuthorityError(
                "Creator credential file missing fields: "
                + ", ".join(sorted(missing))
            )

        return values

    @staticmethod
    def _safe_compare(
        left: str,
        right: str,
    ) -> bool:
        return hashlib.sha256(
            left.encode("utf-8")
        ).digest() == hashlib.sha256(
            right.encode("utf-8")
        ).digest()

    def authenticate_creator(
        self,
        username: str,
        password: str,
    ) -> bool:
        auth = self._load_creator_auth()

        if not self._safe_compare(
            username.strip(),
            auth["username"],
        ):
            return False

        if auth["algorithm"] != (
            "pbkdf2-hmac-sha256"
        ):
            raise AuthorityError(
                "Unsupported creator authentication algorithm."
            )

        try:
            iterations = int(
                auth["iterations"]
            )
            salt = bytes.fromhex(
                auth["salt"]
            )
            expected = bytes.fromhex(
                auth["hash"]
            )
        except (
            ValueError,
            TypeError,
        ) as exc:
            raise AuthorityError(
                "Invalid creator credential format."
            ) from exc

        calculated = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )

        return hmac.compare_digest(
            calculated,
            expected,
        )

    def authenticate_and_identify(
        self,
        username: str,
        password: str,
    ) -> str | None:
        if not self.authenticate_creator(
            username,
            password,
        ):
            return None

        policy = self.policy()

        creator_id = str(
            policy.get(
                "creator",
                {},
            ).get(
                "id",
                "",
            )
        ).strip()

        if not creator_id:
            raise AuthorityError(
                "Creator identity missing from authority policy."
            )

        configured_username = (
            self._load_creator_auth()["username"]
        )

        if configured_username != (
            username.strip()
        ):
            return None

        return creator_id

    def creator_id(self) -> str:
        return str(
            self.policy()
            .get("creator", {})
            .get("id", "")
        )

    def operator_ids(self) -> set[str]:
        operators = self.policy().get(
            "operators",
            [],
        )

        result: set[str] = set()

        for operator in operators:
            if isinstance(operator, dict):
                identifier = str(
                    operator.get("id", "")
                ).strip()

                if identifier:
                    result.add(identifier)

        return result

    def is_creator(self) -> bool:
        return (
            self.actor_id
            == self.creator_id()
        )

    def is_operator(self) -> bool:
        return (
            self.actor_id
            in self.operator_ids()
        )

    def permission(
        self,
        permission: str,
    ) -> bool:
        if self.is_creator():
            return True

        for operator in self.policy().get(
            "operators",
            [],
        ):
            if not isinstance(operator, dict):
                continue

            if operator.get("id") != self.actor_id:
                continue

            permissions = operator.get(
                "permissions",
                [],
            )

            return permission in permissions

        return False

    def can_start(self) -> bool:
        return self.permission("start")

    def can_stop(self) -> bool:
        return self.permission("stop")

    def can_set_goal(self) -> bool:
        return self.permission("set_goal")

    def can_inspect(self) -> bool:
        return self.permission("inspect")

    def can_restore_checkpoint(self) -> bool:
        return self.permission(
            "restore_checkpoint"
        )

    def can_override_autonomy(self) -> bool:
        if not self.is_creator():
            return False

        return bool(
            self.policy()
            .get("creator", {})
            .get("can_override_autonomy", False)
        )

    def can_change_authority(self) -> bool:
        if not self.is_creator():
            return False

        return bool(
            self.policy()
            .get("creator", {})
            .get("can_change_authority_policy", False)
        )

    def emergency_stop_allowed(self) -> bool:
        if not self.is_creator():
            return False

        return bool(
            self.policy()
            .get("intervention", {})
            .get("emergency_stop", False)
        )

    def self_learning_allowed(self) -> bool:
        return bool(
            self.policy()
            .get("autonomy", {})
            .get("self_learning", False)
        )

    def self_evolution_allowed(self) -> bool:
        return bool(
            self.policy()
            .get("autonomy", {})
            .get("self_evolution", False)
        )

    def self_modification_allowed(self) -> bool:
        policy = self.policy()

        return bool(
            policy.get("autonomy", {}).get(
                "self_modification",
                False,
            )
            and policy.get(
                "self_modification",
                {},
            ).get(
                "allowed",
                False,
            )
        )

    def assert_runtime_action(
        self,
        action: str,
    ) -> None:
        checks = {
            "start": self.can_start,
            "stop": self.can_stop,
            "set_goal": self.can_set_goal,
            "inspect": self.can_inspect,
            "restore_checkpoint": self.can_restore_checkpoint,
            "override_autonomy": self.can_override_autonomy,
            "emergency_stop": self.emergency_stop_allowed,
            "change_authority": self.can_change_authority,
            "self_learning": self.self_learning_allowed,
            "self_evolution": self.self_evolution_allowed,
            "self_modification": self.self_modification_allowed,
        }

        check = checks.get(action)

        if check is None:
            raise AuthorityError(
                f"Unknown authority action: {action}"
            )

        if not check():
            raise AuthorityError(
                f"Authority denied for actor "
                f"{self.actor_id}: {action}"
            )

    def protected_path(
        self,
        path: str | Path,
    ) -> bool:
        target = Path(path).resolve()

        policy = self.policy()

        protected = {
            str(item).strip()
            for item in policy.get(
                "protected",
                [],
            )
        }

        if target in {
            self.authority_file,
            self.checksum_file,
            self.creator_auth_file,
        }:
            return True

        if target.name in {
            "authority.json",
            "authority.json.sha256",
            "creator.auth",
        }:
            return True

        if target.name in protected:
            return True

        return False

    def may_modify_path(
        self,
        path: str | Path,
    ) -> bool:
        if self.protected_path(path):
            return False

        policy = self.policy()

        modification = policy.get(
            "self_modification",
            {},
        )

        if not modification.get(
            "allowed",
            False,
        ):
            return False

        if modification.get(
            "workspace_only",
            True,
        ):
            target = Path(path).resolve()
            project_root = Path(
                "/opt/ai/work/nova-evo"
            ).resolve()

            try:
                target.relative_to(project_root)
            except ValueError:
                return False

        return True


def self_test() -> None:
    import tempfile
    import getpass  # noqa: F401

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        authority = root / "authority.json"
        checksum = root / "authority.json.sha256"
        creator_auth = root / "creator.auth"

        policy = {
            "creator": {
                "id": "CREATOR-PRIMARY",
                "can_override_autonomy": True,
                "can_change_authority_policy": True,
            },
            "operators": [
                {
                    "id": "OPERATOR-PRIMARY",
                    "permissions": [
                        "start",
                        "stop",
                        "inspect",
                    ],
                }
            ],
            "autonomy": {
                "enabled": True,
                "self_learning": True,
                "self_evolution": True,
                "self_modification": True,
            },
            "self_modification": {
                "allowed": True,
                "workspace_only": True,
            },
            "intervention": {
                "emergency_stop": True,
            },
            "protected": [
                "authority.json",
                "authority.json.sha256",
                "creator.auth",
                "final_holdout",
            ],
        }

        raw = json.dumps(
            policy,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        authority.write_bytes(raw)

        digest = hashlib.sha256(
            raw
        ).hexdigest()

        checksum.write_text(
            f"{digest}  {authority}\n",
            encoding="utf-8",
        )

        username = "creator-test"
        password = "creator-password-123"

        salt = os.urandom(16)
        iterations = 600_000

        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )

        creator_auth.write_text(
            "version=1\n"
            f"username={username}\n"
            f"salt={salt.hex()}\n"
            f"hash={password_hash.hex()}\n"
            "algorithm=pbkdf2-hmac-sha256\n"
            f"iterations={iterations}\n",
            encoding="utf-8",
        )

        creator = AuthorityManager(
            authority_file=authority,
            checksum_file=checksum,
            creator_auth_file=creator_auth,
            actor_id="CREATOR-PRIMARY",
        )

        assert creator.verify_integrity()

        assert creator.authenticate_creator(
            username,
            password,
        )

        assert not creator.authenticate_creator(
            username,
            "wrong-password",
        )

        assert (
            creator.authenticate_and_identify(
                username,
                password,
            )
            == "CREATOR-PRIMARY"
        )

        assert creator.is_creator()
        assert creator.can_change_authority()
        assert creator.can_override_autonomy()
        assert creator.self_modification_allowed()

        creator.assert_runtime_action(
            "emergency_stop"
        )

        operator = AuthorityManager(
            authority_file=authority,
            checksum_file=checksum,
            creator_auth_file=creator_auth,
            actor_id="OPERATOR-PRIMARY",
        )

        assert operator.is_operator()
        assert operator.can_start()
        assert not operator.can_override_autonomy()

        try:
            operator.assert_runtime_action(
                "override_autonomy"
            )
        except AuthorityError:
            pass
        else:
            raise AssertionError(
                "Operator override must be denied."
            )

        print(
            "AUTHORITY MANAGER SELFTEST: PASSED"
        )


if __name__ == "__main__":
    self_test()
