import json
import os
from typing import Any

from ..base import (
    GenericModelHandler,
    MarkupOutput,
    MarkupPoint,
    ModelHandler,
    OutputHandlerContext,
    OutputPlan,
    TableOutput,
)


class Grt123LungCancerHandler(ModelHandler):
    model_names = ("gc_grt123_lung_cancer",)
    output_suffix = "gc_grt123_lung_cancer_findings.json"

    def build_output_plan(self, context: OutputHandlerContext) -> OutputPlan:
        files = context.files_with_suffixes((".json",))
        files.sort(key=lambda path: os.path.basename(path) != self.output_suffix)
        selected = None
        payload = None
        for path in files:
            try:
                with open(path, encoding="utf-8") as stream:
                    candidate = json.load(stream)
                self._validate_payload(candidate)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if str(candidate["lungcad"].get("name", "")).strip().lower() != "grt123":
                continue
            selected = path
            payload = candidate
            break

        if selected is None or payload is None:
            plan = GenericModelHandler().build_output_plan(context)
            plan.warnings.append(
                f"No valid {self.output_suffix} output was found; used generic output handling."
            )
            return plan

        path = selected

        findings = payload["findings"]
        finding_keys = self._finding_keys(findings)
        lungcad = payload["lungcad"]
        image_info = payload["imageinfo"]
        cancer_info = payload.get("cancerinfo", {})

        plan = OutputPlan()
        plan.tables.append(
            TableOutput(
                name=f"{context.model_label} - Summary",
                columns=["Key", "Value"],
                rows=[
                    ["Case cancer probability", cancer_info.get("casecancerprobability", "")],
                    ["Number of findings", len(findings)],
                    ["Reference finding IDs", self._join(cancer_info.get("referencenoduleids", []))],
                    ["Model name", lungcad.get("name", "")],
                    ["Model revision", lungcad.get("revision", "")],
                    ["Execution time", lungcad.get("datetimeofexecution", "")],
                    ["Computation time (seconds)", lungcad.get("computationtimeinseconds", "")],
                    ["Coordinate system", lungcad.get("coordinatesystem", "")],
                ],
                source_file=path,
                identity="summary",
            )
        )

        finding_columns = [
            "ID", "X", "Y", "Z", "Detection probability", "Cancer probability"
        ]
        finding_rows = [
            [
                finding.get("id", ""),
                finding.get("x", ""),
                finding.get("y", ""),
                finding.get("z", ""),
                finding.get("probability", ""),
                finding.get("cancerprobability", ""),
            ]
            for finding in findings
        ]
        plan.tables.append(
            TableOutput(
                name=f"{context.model_label} - Findings",
                columns=finding_columns,
                rows=finding_rows,
                source_file=path,
                identity="findings",
                link_group="findings",
                row_keys=finding_keys,
            )
        )

        coordinate_system = str(lungcad.get("coordinatesystem", "")).strip().lower()
        if coordinate_system != "world":
            plan.warnings.append(
                f"Finding annotations were not created: unsupported coordinate system "
                f"{lungcad.get('coordinatesystem')!r}."
            )
            return plan

        points = []
        for finding_index, finding in enumerate(findings):
            if not self._has_numeric_position(finding):
                plan.warnings.append(
                    f"Finding {finding.get('id', '?')} has no valid world position and was not annotated."
                )
                continue
            finding_id = finding.get("id", len(points))
            description = (
                f"Detection probability: {finding.get('probability', '')}; "
                f"Cancer probability: {finding.get('cancerprobability', '')}"
            )
            points.append(
                MarkupPoint(
                    label=f"Finding {finding_id}",
                    position_lps=(float(finding["x"]), float(finding["y"]), float(finding["z"])),
                    description=description,
                    key=finding_keys[finding_index],
                )
            )

        if points:
            plan.markups.append(
                MarkupOutput(
                    name=f"{context.model_label} - Findings",
                    points=points,
                    image_geometry=image_info,
                    source_file=path,
                    identity="findings",
                    link_group="findings",
                )
            )
        return plan

    @staticmethod
    def _validate_payload(payload: Any) -> None:
        if not isinstance(payload, dict):
            raise ValueError("GRT123 output must be a JSON object.")
        for key in ("lungcad", "imageinfo", "findings"):
            if key not in payload:
                raise ValueError(f"GRT123 output is missing required key {key!r}.")
        if not isinstance(payload["lungcad"], dict):
            raise ValueError("GRT123 lungcad value must be an object.")
        if not isinstance(payload["imageinfo"], dict):
            raise ValueError("GRT123 imageinfo value must be an object.")
        if not isinstance(payload["findings"], list):
            raise ValueError("GRT123 findings value must be an array.")
        if not all(isinstance(finding, dict) for finding in payload["findings"]):
            raise ValueError("Every GRT123 finding must be an object.")

    @staticmethod
    def _has_numeric_position(finding: dict[str, Any]) -> bool:
        return all(
            isinstance(finding.get(axis), (int, float)) and not isinstance(finding.get(axis), bool)
            for axis in ("x", "y", "z")
        )

    @staticmethod
    def _finding_keys(findings: list[dict[str, Any]]) -> list[str]:
        occurrences: dict[str, int] = {}
        keys = []
        for index, finding in enumerate(findings):
            finding_id = finding.get("id")
            base_key = f"finding:{finding_id}" if finding_id is not None else f"row:{index}"
            occurrence = occurrences.get(base_key, 0)
            occurrences[base_key] = occurrence + 1
            keys.append(base_key if occurrence == 0 else f"{base_key}#{occurrence + 1}")
        return keys

    @staticmethod
    def _join(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        return str(value)
