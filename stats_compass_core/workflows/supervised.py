"""
Shared skeleton for the supervised workflows.

classification.py and regression.py were near-duplicates maintained in
parallel: same sequence, same feature-engineering block, same training
dispatch, same evaluation, same plot loop, same result assembly. Every fix had
to be applied twice, and nothing enforced that it was. Three defects shipped
that way — a feature-column fix that matched classification's keyword call and
missed regression's positional one, a holdout fix that reached both evaluators
but only one set of plots, and a name referenced in regression without being
assigned.

The sequence lives here once. What genuinely differs between the two lives in a
SupervisedSpec: which trainers to dispatch to, which evaluator to call, and how
to turn a requested plot into either parameters or a reason for skipping.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from stats_compass_core.results import ToolWarning
from stats_compass_core.state import DataFrameState

from .feature_engineering import map_feature_columns, run_feature_engineering_steps
from .results import WorkflowArtifacts, WorkflowResult, WorkflowStepResult
from .utils import build_holdout_predictions, build_training_params, get_tool, run_step

# =============================================================================
# Plot dispatch
# =============================================================================

@dataclass
class PlotContext:
    """Everything a plot builder might need to decide what to do.

    dataframe_name is already the holdout view, so a builder cannot
    accidentally plot rows the model was fitted on.
    """

    plot_name: str
    schema: type
    dataframe_name: str | None
    target_column: str
    prediction_column: str | None
    model_id: str | None
    probability_columns: list[str] | None = None
    class_labels: list[Any] | None = None


@dataclass
class PlotDecision:
    """Either parameters to run the plot with, or a reason it was not run.

    Returning a reason rather than None is what keeps a skipped chart visible.
    Regression used to drop unsupported plots with a bare `continue`, so a
    config asking for residuals produced no chart, no step and no explanation.
    """

    params: Any | None = None
    skip_reason: str | None = None


@dataclass(frozen=True)
class SupervisedSpec:
    """What differs between classification and regression."""

    kind: str
    tool_map: dict[str, str]
    model_labels: dict[str, str]
    evaluator_tool: str
    plot_map: dict[str, tuple[str, str]]
    build_plot_params: Callable[[PlotContext], PlotDecision]
    default_config: Callable[[], Any]
    failure_suggestion: str


# =============================================================================
# Step recording
# =============================================================================

class StepRecorder:
    """Owns step numbering and the failure/skip boilerplate.

    The two workflows between them carried 34 hand-written `step_index += 1`
    lines and 29 hand-built failure blocks. Every new step re-implemented the
    same try/except and the same numbering, which is how they drifted apart.

    Indices are 1-based and contiguous, as ARCHITECTURE.md documents.
    """

    def __init__(self) -> None:
        self.steps: list[WorkflowStepResult] = []
        self._index = 0

    @property
    def index(self) -> int:
        """The last index issued. Handed to helpers that number their own steps."""
        return self._index

    def adopt(self, steps: list[WorkflowStepResult], next_index: int) -> None:
        """Absorb steps from a helper that numbered them itself."""
        self.steps.extend(steps)
        self._index = next_index

    def run(
        self,
        name: str,
        func: Callable,
        state: DataFrameState,
        params: Any,
        summary_template: str,
    ) -> WorkflowStepResult:
        self._index += 1
        result = run_step(
            step_name=name,
            step_index=self._index,
            func=func,
            state=state,
            params=params,
            summary_template=summary_template,
        )
        self.steps.append(result)
        return result

    def fail(self, name: str, summary: str, error: str) -> WorkflowStepResult:
        self._index += 1
        result = WorkflowStepResult(
            step_name=name,
            step_index=self._index,
            status="failed",
            duration_ms=0,
            summary=summary,
            error=error,
        )
        self.steps.append(result)
        return result

    def skip(self, name: str, reason: str) -> WorkflowStepResult:
        self._index += 1
        result = WorkflowStepResult(
            step_name=name,
            step_index=self._index,
            status="skipped",
            duration_ms=0,
            summary=f"{name} skipped: {reason}",
            skip_reason=reason,
        )
        self.steps.append(result)
        return result


# =============================================================================
# The workflow
# =============================================================================

def run_supervised_workflow(
    state: DataFrameState,
    params: Any,
    spec: SupervisedSpec,
) -> WorkflowResult:
    """
    Execute a supervised workflow: drop columns, feature-engineer, train,
    evaluate, plot, assemble.

    Args:
        state: The DataFrame state to operate on.
        params: RunClassificationInput or RunRegressionInput.
        spec: What differs between the two.
    """
    started_at = datetime.now()
    config = params.config or spec.default_config()

    source_name = params.dataframe_name or state.get_active_dataframe_name()
    current_df_name = source_name
    feature_columns = params.feature_columns  # May be renamed by encoding
    fe_mapping: dict[str, Any] = {}

    recorder = StepRecorder()
    charts_generated = 0
    dataframes_created: list[str] = []
    models_created: list[str] = []

    model_id: str | None = None
    predictions_df_name: str | None = None
    prediction_column: str | None = None
    probability_columns: list[str] | None = None
    class_labels: list[Any] | None = None

    # =========================================================================
    # Step 0a: Drop Columns (inline, no step recorded)
    # =========================================================================
    if config.drop_columns:
        cols_to_drop = [c for c in config.drop_columns if c != params.target_column]
        if cols_to_drop:
            df = state.get_dataframe(current_df_name)
            df = df.drop(columns=cols_to_drop, errors="ignore")
            state.set_dataframe(
                df, name=current_df_name, operation="drop_columns", set_active=True
            )

    # =========================================================================
    # Step 0b: Feature Engineering (optional)
    # =========================================================================
    if config.feature_engineering:
        fe_steps, fe_dfs, current_df_name, next_index, fe_mapping = (
            run_feature_engineering_steps(
                state=state,
                config=config.feature_engineering,
                source_name=source_name,
                target_column=params.target_column,
                start_step_index=recorder.index,
                feature_columns=params.feature_columns,
            )
        )
        recorder.adopt(fe_steps, next_index)
        dataframes_created.extend(fe_dfs)
        # Encoding renamed the columns it replaced.
        feature_columns = map_feature_columns(feature_columns, fe_mapping)

    # =========================================================================
    # Step 1: Train Model (registry-based dispatch)
    # =========================================================================
    tool_name = spec.tool_map.get(config.model_type)
    if tool_name is None:
        available = ", ".join(spec.tool_map.keys())
        recorder.fail(
            "train_model",
            summary=f"Unknown model type: {config.model_type}",
            error=f"Unknown model type '{config.model_type}'. Available: {available}",
        )
    else:
        try:
            train_func, TrainInputSchema = get_tool("ml", tool_name)
            train_params = build_training_params(
                input_schema=TrainInputSchema,
                source_name=current_df_name,  # Use FE'd DataFrame if available
                target_column=params.target_column,
                feature_columns=feature_columns,  # Translated through encoding
                config=config,
            )

            model_label = spec.model_labels.get(config.model_type, config.model_type)
            step_result = recorder.run(
                "train_model", train_func, state, train_params,
                f"Trained {model_label}",
            )

            if step_result.status == "success" and step_result.result:
                trained = step_result.result
                model_id = trained.get("model_id")
                predictions_df_name = trained.get("predictions_dataframe")
                # Read from the result rather than rebuilding "pred_{target}":
                # regression used to guess the name, which would break silently
                # the day a trainer named its output column differently.
                prediction_column = trained.get("prediction_column")
                probability_columns = trained.get("probability_columns")
                class_labels = trained.get("class_labels")

                if model_id:
                    models_created.append(model_id)
                if predictions_df_name:
                    dataframes_created.append(predictions_df_name)

        except Exception as e:
            recorder.fail("train_model", f"Failed to train model: {e}", str(e))

    # =========================================================================
    # Step 2: Evaluate Model
    # =========================================================================
    # One gate for both workflows. Evaluation needs a predictions frame and the
    # column holding the predictions; the model id is not enough.
    if predictions_df_name and prediction_column:
        try:
            eval_func, EvalInputSchema = get_tool("ml", spec.evaluator_tool)
            eval_params = EvalInputSchema(
                dataframe_name=predictions_df_name,
                target_column=params.target_column,
                prediction_column=prediction_column,
            )
            recorder.run(
                "evaluate_model", eval_func, state, eval_params,
                "Evaluated model performance",
            )
        except Exception as e:
            recorder.fail("evaluate_model", f"Failed to evaluate model: {e}", str(e))

    # =========================================================================
    # Step 3+: Generate Plots
    # =========================================================================
    if config.generate_plots and predictions_df_name and prediction_column:
        # Plot the holdout, not everything: a confusion matrix over rows the
        # model memorised contradicts the accuracy printed beside it.
        plot_df_name, _plotted_on = build_holdout_predictions(
            state, predictions_df_name, params.target_column
        )

        for plot_name in config.plots:
            entry = spec.plot_map.get(plot_name)
            if entry is None:
                # Configured but not registered. Recorded rather than dropped,
                # so a config asking for a chart that cannot be produced says so.
                recorder.skip(
                    plot_name,
                    f"no plot tool is registered for '{plot_name}' in the "
                    f"{spec.kind} workflow",
                )
                continue

            tool_name, chart_type = entry
            try:
                plot_func, PlotInputSchema = get_tool("plots", tool_name)
                decision = spec.build_plot_params(PlotContext(
                    plot_name=plot_name,
                    schema=PlotInputSchema,
                    dataframe_name=plot_df_name,
                    target_column=params.target_column,
                    prediction_column=prediction_column,
                    model_id=model_id,
                    probability_columns=probability_columns,
                    class_labels=class_labels,
                ))

                if decision.params is None:
                    recorder.skip(
                        chart_type,
                        decision.skip_reason or "no parameters could be built",
                    )
                    continue

                step_result = recorder.run(
                    chart_type, plot_func, state, decision.params,
                    f"Generated {chart_type.replace('_', ' ')}",
                )
                if step_result.status == "success":
                    charts_generated += 1

            except Exception as e:
                recorder.fail(
                    chart_type, f"Failed to generate {chart_type}: {e}", str(e)
                )

    # =========================================================================
    # Build Final Result
    # =========================================================================
    completed_at = datetime.now()
    total_duration_ms = int((completed_at - started_at).total_seconds() * 1000)

    steps = recorder.steps
    failed_steps = [s for s in steps if s.status == "failed"]
    success_steps = [s for s in steps if s.status == "success"]

    # Both workflows now explain themselves on failure. Classification used to
    # leave error_summary and suggestion unset, so an assistant handed a failed
    # classification had nothing to relay — the summariser forwards both fields.
    if not success_steps:
        overall_status = "failed"
        error_summary = "All steps failed"
        suggestion = spec.failure_suggestion
    elif failed_steps:
        overall_status = "partial_failure"
        failed_names = [s.step_name for s in failed_steps]
        error_summary = f"{len(failed_steps)} step(s) failed: {', '.join(failed_names)}"
        suggestion = "Review failed steps. The trained model may still be usable."
    else:
        overall_status = "success"
        error_summary = None
        suggestion = None

    notes = []
    if predictions_df_name and prediction_column:
        notes.append(
            f"Predictions are in '{predictions_df_name}' "
            f"(includes '{prediction_column}' column)"
        )
    if models_created:
        notes.append(
            f"Trained model ID: '{models_created[0]}' - "
            "use for feature_importance or predictions"
        )

    warnings = collect_step_warnings(steps, fe_mapping)

    artifacts = WorkflowArtifacts(
        dataframes_created=dataframes_created,
        models_created=models_created,
        charts_generated=charts_generated,
        final_dataframe=predictions_df_name,
    )

    return WorkflowResult(
        workflow_name=f"run_{spec.kind}",
        status=overall_status,
        started_at=started_at,
        completed_at=completed_at,
        total_duration_ms=total_duration_ms,
        input_dataframe=source_name,
        steps=steps,
        artifacts=artifacts,
        error_summary=error_summary,
        suggestion=suggestion,
        notes=notes,
        recoverable=True,
        warnings=warnings,
    )


def collect_step_warnings(
    steps: list[WorkflowStepResult], column_mapping: dict[str, Any]
) -> list[ToolWarning]:
    """Lift every step's warnings to the workflow, in the user's column names.

    The trainer sees 'cancellation_reason_encoded'; the user has never heard of
    it. A warning is only useful if it names the column they can go and drop.
    """
    original_of: dict[str, str] = {}
    for original, encoded in column_mapping.items():
        for name in encoded if isinstance(encoded, list) else [encoded]:
            original_of[name] = original

    warnings: list[ToolWarning] = []
    for step in steps:
        for raw in (step.result or {}).get("warnings") or []:
            warning = ToolWarning.model_validate(raw)
            columns = list(
                dict.fromkeys(original_of.get(c, c) for c in warning.columns)
            )
            message = warning.message
            for name in warning.columns:
                if name in original_of:
                    message = message.replace(
                        f"'{name}'", f"'{original_of[name]}' (encoded as '{name}')"
                    )
            warnings.append(
                warning.model_copy(update={"columns": columns, "message": message})
            )
    return warnings
