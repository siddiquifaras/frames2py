"""A fake macOS power backend for tests of ``benchmarks.power``."""

from __future__ import annotations

from typing import Any

from benchmarks import power


class FakeMac:
    """A macOS power backend with a chosen capability value and assertion behaviour."""

    def __init__(self, capabilities: int | None, create_fails: bool = False,
                 confirms: bool = True) -> None:
        self.caps, self.create_fails, self.confirms = capabilities, create_fails, confirms
        self.held: set[int] = set()
        self.created = 0

    def capabilities(self) -> int | None:
        return self.caps

    def create_assertion(self, name: str) -> int:
        if self.create_fails:
            raise power.PowerStateError("denied")
        self.created += 1
        self.held.add(self.created)
        return self.created

    def assertion_properties(self, assertion: int) -> dict[str, Any]:
        if not self.confirms or assertion not in self.held:
            return {}
        return {"AssertType": power.ASSERTION_TYPE, "AssertLevel": power.ASSERTION_LEVEL_ON, "AssertName": "x"}

    def release_assertion(self, assertion: int) -> None:
        self.held.discard(assertion)
