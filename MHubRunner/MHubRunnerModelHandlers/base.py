import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InputHandlerContext:
    model_name: str
    selected_nodes: list[Any]
    selected_modality: str | None = None


@dataclass
class InputItem:
    node: Any
    modality: str | None = None
    target_subdirectory: str = ""


@dataclass
class InputPlan:
    items: list[InputItem]


@dataclass
class OutputHandlerContext:
    model_name: str
    model_label: str
    model_categories: list[str]
    output_directory: str


@dataclass
class TableOutput:
    name: str
    columns: list[str]
    rows: list[list[Any]]
    source_file: str
    identity: str = ""
    link_group: str = ""


@dataclass
class MarkupPoint:
    label: str
    position_lps: tuple[float, float, float]
    description: str = ""


@dataclass
class MarkupOutput:
    name: str
    points: list[MarkupPoint]
    image_geometry: dict[str, Any]
    source_file: str
    identity: str = ""
    link_group: str = ""


@dataclass
class OutputPlan:
    tables: list[TableOutput] = field(default_factory=list)
    markups: list[MarkupOutput] = field(default_factory=list)
    segmentation_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ModelHandler:
    model_names: tuple[str, ...] = ()

    def resolve_inputs(self, context: InputHandlerContext) -> InputPlan:
        if len(context.selected_nodes) != 1 or context.selected_nodes[0] is None:
            raise ValueError("This model requires exactly one selected input volume.")
        return InputPlan(
            items=[InputItem(context.selected_nodes[0], modality=context.selected_modality)]
        )

    def build_output_plan(self, context: OutputHandlerContext) -> OutputPlan:
        raise NotImplementedError

    @staticmethod
    def find_files(output_directory: str, suffixes: tuple[str, ...]) -> list[str]:
        matches = []
        for root, _, files in os.walk(output_directory):
            for filename in files:
                if filename == "mhubrunner-run.json":
                    continue
                if filename.lower().endswith(suffixes):
                    matches.append(os.path.join(root, filename))
        return sorted(matches)


class GenericModelHandler(ModelHandler):
    """Preserve the generic output behavior for models without an adapter."""

    def build_output_plan(self, context: OutputHandlerContext) -> OutputPlan:
        plan = OutputPlan(
            segmentation_files=self.find_files(context.output_directory, (".seg.dcm",))
        )
        json_files = self.find_files(context.output_directory, (".json",))
        csv_files = self.find_files(context.output_directory, (".csv",))

        # Keep the pre-handler behavior: automatically display one tabular output.
        if json_files:
            plan.tables.append(self._json_table(context, json_files[0]))
        elif csv_files:
            plan.tables.append(self._csv_table(context, csv_files[0]))
        return plan

    def _json_table(self, context: OutputHandlerContext, path: str) -> TableOutput:
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
        flattened: dict[str, Any] = {}

        def flatten(value: Any, prefix: str = "") -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    flatten(child, f"{prefix}.{key}" if prefix else str(key))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    flatten(child, f"{prefix}[{index}]")
            else:
                flattened[prefix] = value

        flatten(payload)
        return TableOutput(
            name=f"{context.model_label} - Output",
            columns=["Key", "Value"],
            rows=[[key, value] for key, value in flattened.items()],
            source_file=path,
        )

    def _csv_table(self, context: OutputHandlerContext, path: str) -> TableOutput:
        with open(path, encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            try:
                columns = next(reader)
            except StopIteration:
                columns = []
            rows = list(reader)
        return TableOutput(
            name=f"{context.model_label} - Output",
            columns=columns,
            rows=rows,
            source_file=path,
        )
