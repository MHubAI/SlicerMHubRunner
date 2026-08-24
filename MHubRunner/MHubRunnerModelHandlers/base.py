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
    output_files: list[str] | None = None

    def files_with_suffixes(self, suffixes: tuple[str, ...]) -> list[str]:
        # Use the manifest allowlist when supplied; otherwise scan without following symlinks.
        if self.output_files is not None:
            candidates = self.output_files
        else:
            candidates = ModelHandler.find_files(self.output_directory, suffixes)
        return sorted(
            path
            for path in candidates
            if os.path.basename(path) != "mhubrunner-run.json"
            and path.lower().endswith(suffixes)
        )


@dataclass
class TableOutput:
    name: str
    columns: list[str]
    rows: list[list[Any]]
    source_file: str
    identity: str = ""
    link_group: str = ""
    row_keys: list[str] = field(default_factory=list)


@dataclass
class MarkupPoint:
    label: str
    position_lps: tuple[float, float, float]
    description: str = ""
    key: str = ""


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
        # Keep legacy runs confined to regular files physically located below their run root.
        output_root = os.path.realpath(output_directory)
        matches = []
        for root, directories, files in os.walk(output_root, followlinks=False):
            directories[:] = [
                name
                for name in directories
                if not os.path.islink(os.path.join(root, name))
            ]
            for filename in files:
                if filename == "mhubrunner-run.json":
                    continue
                candidate = os.path.join(root, filename)
                if os.path.islink(candidate) or not filename.lower().endswith(suffixes):
                    continue
                resolved_candidate = os.path.realpath(candidate)
                try:
                    candidate_is_inside_output = os.path.commonpath(
                        [output_root, resolved_candidate]
                    ) == output_root
                except ValueError:
                    candidate_is_inside_output = False
                if candidate_is_inside_output and os.path.isfile(resolved_candidate):
                    matches.append(resolved_candidate)
        return sorted(matches)


class GenericModelHandler(ModelHandler):
    """Preserve the generic output behavior for models without an adapter."""

    def build_output_plan(self, context: OutputHandlerContext) -> OutputPlan:
        plan = OutputPlan(
            segmentation_files=context.files_with_suffixes((".seg.dcm",))
        )
        json_files = context.files_with_suffixes((".json",))
        csv_files = context.files_with_suffixes((".csv",))

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
