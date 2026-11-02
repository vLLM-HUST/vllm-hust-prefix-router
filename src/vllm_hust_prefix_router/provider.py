"""External Router host provider for vLLM-HUST Extension Manager."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from vllm_hust_prefix_router.service.provider_checks import (
    EXTENSION_ID,
    probe_deployment,
    validate_binding,
)


class PrefixRouterProvider:
    """Describe operator-owned routing and verify its required live services."""

    name = "prefix-router"

    def supports(self, manifest: Any) -> bool:
        return (
            manifest.host.provider == self.name
            and manifest.bundle_id == EXTENSION_ID
            and manifest.runtime.type == "external_service"
        )

    def plan(
        self, manifest: Any, configuration: dict[str, Any], *, enabled: bool
    ) -> Any:
        from vllm_hust_ext.providers.base import PlanAction, ProviderPlan

        actions = [
            PlanAction(
                "check_external_router",
                EXTENSION_ID,
                manifest.lifecycle_owner,
                details={
                    "router_url": configuration.get("router_url"),
                    "enabled": enabled,
                },
            )
        ]
        actions.extend(
            PlanAction(
                "check_service",
                service.service_id,
                manifest.lifecycle_owner,
                details={
                    "protocol": service.protocol,
                    "endpoint_config": service.endpoint_config,
                },
            )
            for service in manifest.requires_services
        )
        return ProviderPlan(
            manifest.bundle_id,
            self.name,
            tuple(actions),
            {
                "router_url": configuration.get("router_url"),
                "router_config_path": configuration.get("router_config_path"),
            },
            (
                "The operator starts, stops, upgrades, and rolls back the Router; "
                "Extension Manager only records intent and checks live services.",
            ),
        )

    def render(self, plan: Any) -> tuple[Any, ...]:
        from vllm_hust_ext.providers.base import RenderArtifact

        return (
            RenderArtifact(
                "prefix-router-operator-plan.json",
                "application/json",
                json.dumps(
                    {
                        "extension_id": plan.extension_id,
                        "provider": self.name,
                        "actions": [
                            {
                                "operation": action.operation,
                                "target": action.target,
                                "lifecycle_owner": action.lifecycle_owner,
                                "mutating": action.mutating,
                                "details": action.details,
                            }
                            for action in plan.actions
                        ],
                        "configuration": plan.generated_config,
                        "warnings": plan.warnings,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            ),
        )

    def check(self, manifest: Any, configuration: dict[str, Any]) -> Any:
        from vllm_hust_ext.providers.base import ProviderCheck, assess_compatibility

        try:
            installed_version = version("vllm-hust-prefix-router")
        except PackageNotFoundError:
            return ProviderCheck(
                False,
                False,
                degraded=True,
                evidence=("vllm-hust-prefix-router distribution is not installed",),
            )
        compatible, evidence = assess_compatibility(
            manifest, {}, detected_host_version=installed_version
        )
        if compatible is False:
            return ProviderCheck(False, False, degraded=True, evidence=evidence)
        try:
            binding = validate_binding(configuration)
        except (OSError, ValueError, TypeError) as error:
            return ProviderCheck(
                compatible,
                False,
                degraded=True,
                evidence=evidence + (f"Router configuration is not bound: {error}",),
            )
        probe = probe_deployment(binding)
        return ProviderCheck(
            compatible,
            True,
            reachable=probe.reachable,
            healthy=probe.healthy,
            degraded=probe.degraded,
            evidence=evidence + probe.evidence,
        )
