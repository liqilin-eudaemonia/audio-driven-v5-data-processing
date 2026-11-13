"""Fixed-set ArcFace/FID/FVD/SyncNet-C validation benchmark."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable

from ltx_trainer import logger


def validation_collective_timeout_seconds(config) -> int:
    """Allow long validation ranks to finish within the configured validation budget."""
    if config.interval or config.run_only:
        return max(600, int(config.metric_timeout_seconds))
    return 600


def validate_metric_results(
    identity: dict,
    visual: dict,
    syncnet: dict,
    expected_count: int,
    *,
    required_visual_metrics: tuple[str, ...] = ("fid", "fvd"),
) -> None:
    """Reject partial failed runs, without treating explicitly ineligible cases as errors."""
    errors = []
    ident = identity.get("id_similarity", {})
    if ident.get("count") != expected_count:
        errors.append(f"Identity coverage {ident.get('count')}/{expected_count}")
    sync = syncnet.get("sync_c", {})
    if sync.get("dataset_count") != expected_count:
        errors.append(f"SyncNet dataset_count {sync.get('dataset_count')} != {expected_count}")
    eligible = sync.get("eligible_count")
    success = sync.get("success_count")
    if (
        not isinstance(eligible, int)
        or eligible < 0
        or success != eligible
        or sync.get("count") != success
    ):
        errors.append(f"SyncNet eligible coverage {success}/{eligible}")
    failed = syncnet.get("status_counts", {}).get("error", 0)
    if failed:
        errors.append(f"SyncNet reported {failed} per-sample errors")
    values = {"id_similarity": ident.get("mean")}
    values.update(
        {metric: visual.get(metric) for metric in required_visual_metrics}
    )
    if eligible:
        values["sync_c"] = sync.get("mean")
    for name, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"{name} is missing or non-finite: {value!r}")
    if errors:
        raise ValueError("Incomplete validation metrics: " + "; ".join(errors))


class ValidationBenchmark:
    def __init__(
        self,
        config,
        output_root: Path,
        step: int,
        *,
        expected_sample_count: int,
        metadata_rows: list[dict] | None = None,
        label: str | None = None,
        metric_prefix: str | None = None,
    ) -> None:
        self.config = config
        self.step = int(step)
        if expected_sample_count < 1:
            raise ValueError("expected_sample_count must be positive")
        self.expected_sample_count = int(expected_sample_count)
        self.label = label
        self.metric_prefix = (
            metric_prefix
            if metric_prefix is not None
            else f"validation_{label}"
            if label
            else "validation"
        )
        self.metadata_rows = metadata_rows or []
        label_suffix = f"_{label}" if label else ""
        self.output_dir = output_root / f"step_{self.step:06d}{label_suffix}_benchmark"
        self.status_path = self.output_dir / "benchmark_status.json"

    @property
    def enabled(self) -> bool:
        return bool(self.config.metrics)

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def prepare(self, is_main_process: bool) -> None:
        if not self.enabled or not is_main_process:
            return
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
        self._atomic_json(self.status_path, {"status": "running", "error": None})

    @staticmethod
    def _run(command: list[str], cwd: Path, log_path: Path, timeout: float) -> None:
        env = os.environ.copy()
        current = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(cwd) if not current else str(cwd) + os.pathsep + current
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("COMMAND: " + " ".join(command) + "\n")
            handle.flush()
            timeout = max(1.0, timeout)
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                raise subprocess.TimeoutExpired(command, timeout) from exc
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

    @staticmethod
    def _extract_transcript(prompt: str, metadata: dict) -> str:
        words_pattern = re.escape("[WORDS]") + "(.*?)" + re.escape("[END_WORDS]")
        utterances = re.findall(words_pattern, prompt, flags=re.S | re.I)
        utterances.extend(
            re.findall(
                r"<words(?:[ 	]*:[^>]*)?>(.*?)</words[ 	]*>",
                prompt,
                flags=re.S | re.I,
            )
        )
        transcript = " ".join(" ".join(item.split()) for item in utterances).strip()
        if transcript:
            return transcript
        fallback = metadata.get("qwenasr")
        return " ".join(fallback.split()) if isinstance(fallback, str) else ""

    @staticmethod
    def _sync_eligibility(
        prompt: str, metadata: dict, transcript: str
    ) -> tuple[bool, str, str]:
        if metadata.get("_syncnet_applicable") is False:
            return False, "generated_plan_no_response", "none"
        category = metadata.get("_validation_category")
        visible_count = metadata.get("_visible_speaker_count")
        speaker_ids = metadata.get("_dialogue_speaker_ids")
        unique_speakers = {
            str(value).strip()
            for value in speaker_ids or []
            if isinstance(value, str) and value.strip()
        }

        has_explicit_metadata = (
            category is not None or visible_count is not None or speaker_ids is not None
        )
        if has_explicit_metadata:
            eligible = bool(transcript)
            if category is not None:
                eligible = eligible and category == "single_speaker"
            if visible_count is not None:
                eligible = eligible and visible_count == 1
            if speaker_ids is not None:
                eligible = eligible and len(unique_speakers) == 1
            if not transcript:
                reason = "missing_dialogue"
            elif category is not None and category != "single_speaker":
                reason = f"validation_category_{category}"
            elif visible_count is not None and visible_count != 1:
                reason = f"visible_speaker_count_{visible_count}"
            elif speaker_ids is not None and len(unique_speakers) != 1:
                reason = f"dialogue_speaker_count_{len(unique_speakers)}"
            else:
                reason = "single_visible_speaker_pending_generated_face_gate"
            return eligible, reason, "single" if eligible else "unknown"

        visible_match = re.search(
            re.escape("[VISIBLE_MAIN_CHARACTER]")
            + r".*?<count>[^0-9]*([0-9]+)[^<]*</count>",
            prompt,
            flags=re.S | re.I,
        )
        parsed_visible_count = int(visible_match.group(1)) if visible_match else None
        parsed_speakers = {
            " ".join(value.split())
            for value in re.findall(
                r"<character_id>(.*?)</character_id>", prompt, flags=re.S | re.I
            )
            if value.strip()
        }
        has_structured_metadata = parsed_visible_count is not None or bool(
            parsed_speakers
        )
        eligible = bool(transcript)
        if has_structured_metadata:
            eligible = (
                eligible and parsed_visible_count == 1 and len(parsed_speakers) == 1
            )
        reason = (
            "single_visible_speaker_pending_generated_face_gate"
            if eligible
            else "missing_or_non_single_dialogue"
        )
        return eligible, reason, "single" if eligible else "unknown"

    def _write_manifest(
        self, paths: list[Path], prompts: list[str], references: list[str]
    ) -> Path:
        expected = self.expected_sample_count
        if (
            len(paths) != expected
            or len(prompts) != expected
            or len(references) != expected
        ):
            raise RuntimeError(
                f"validation benchmark expected {expected} cases, got "
                f"paths={len(paths)} prompts={len(prompts)} references={len(references)}"
            )
        if self.metadata_rows and len(self.metadata_rows) != expected:
            raise RuntimeError(
                f"validation benchmark expected {expected} metadata rows, got {len(self.metadata_rows)}"
            )
        manifest_path = self.output_dir / "benchmark_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for index, (path, prompt, reference) in enumerate(
                zip(paths, prompts, references, strict=True)
            ):
                metadata = self.metadata_rows[index] if self.metadata_rows else {}
                transcript = self._extract_transcript(prompt, metadata)
                if not transcript and self.config.plain_prompt_is_transcript:
                    transcript = " ".join(prompt.split())
                sync_eligible, sync_reason, speech_type = self._sync_eligibility(
                    prompt, metadata, transcript
                )
                if not path.is_file():
                    raise FileNotFoundError(
                        f"generated validation video not found: {path}"
                    )
                if not Path(reference).is_file():
                    raise FileNotFoundError(
                        f"reference validation video not found: {reference}"
                    )
                row = {
                    "id": f"{self.label or 'train_val'}_{index:04d}",
                    "prompt": prompt,
                    "transcript": transcript,
                    "scene_type": "live",
                    "speech_type": speech_type,
                    "metric_applicability": {
                        "id_similarity": {
                            "eligible": True,
                            "reason": "reference_first_frame",
                        },
                        "fid": {
                            "eligible": True,
                            "reason": "fixed_training_validation_set",
                        },
                        "fvd": {
                            "eligible": True,
                            "reason": "fixed_training_validation_set",
                        },
                        "sync_c": {
                            "eligible": sync_eligible,
                            "reason": sync_reason,
                        },
                        "sync_d": {
                            "eligible": sync_eligible,
                            "reason": "required_by_syncnet_runner"
                            if sync_eligible
                            else sync_reason,
                        },
                    },
                    "reference_video": reference,
                    "generated_video": str(path),
                    "seed": int(self.config.seed),
                    "checkpoint_step": self.step,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return manifest_path

    def _score(
        self,
        paths: list[Path],
        prompts: list[str],
        references: list[str],
        log_metrics: Callable[[dict[str, float]], None],
    ) -> dict:
        manifest = self._write_manifest(paths, prompts, references)
        benchmark_root = Path(self.config.benchmark_root)
        benchmark_dir = self.output_dir / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        log_path = benchmark_dir / "benchmark_metrics.log"
        deadline = time.monotonic() + int(self.config.metric_timeout_seconds)

        def remaining() -> float:
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError(
                    "validation benchmark exceeded metric_timeout_seconds"
                )
            return value

        id_output = benchmark_dir / "arcface_id.json"
        id_script = (
            Path(__file__).resolve().parents[2] / "scripts" / "compute_arcface_id.py"
        )
        id_command = [
            self.config.id_python,
            str(id_script),
            "--manifest",
            str(manifest),
            "--output",
            str(id_output),
            "--model-root",
            self.config.id_model_root,
            "--model-name",
            self.config.id_model_name,
            "--sample-frames",
            str(self.config.id_sample_frames),
        ]
        if self.config.id_cpu:
            id_command.append("--cpu")
        self._run(id_command, benchmark_root, log_path, remaining())

        visual_output = benchmark_dir / "visual_distribution.json"
        single_sample_diagnostic = (
            self.expected_sample_count == 1
            and bool(self.config.allow_small_sample_metrics)
        )
        requested_visual_metrics = "fid" if single_sample_diagnostic else "fid,fvd"
        self._run(
            [
                self.config.avbench_python,
                str(benchmark_root / "metrics" / "visual_distribution.py"),
                "--manifest",
                str(manifest),
                "--output",
                str(visual_output),
                "--cache-dir",
                str(benchmark_dir / "visual_cache"),
                "--device",
                self.config.metric_device,
                "--fid-frames",
                str(self.config.fid_frames),
                "--fvd-frames",
                str(self.config.fvd_frames),
                "--metrics",
                requested_visual_metrics,
                "--force",
            ],
            benchmark_root,
            log_path,
            remaining(),
        )

        sync_output = benchmark_dir / "syncnet.json"
        self._run(
            [
                self.config.vocal_sync_python,
                str(benchmark_root / "metrics" / "syncnet_metrics.py"),
                "--manifest",
                str(manifest),
                "--output",
                str(sync_output),
                "--per-sample-output",
                str(benchmark_dir / "syncnet_per_sample.jsonl"),
                "--work-dir",
                str(benchmark_dir / "syncnet_work"),
                "--syncnet-root",
                self.config.syncnet_root,
                "--gpus",
                self.config.syncnet_gpus,
                "--speech-types",
                "single",
                "--single-face-only",
            ],
            benchmark_root,
            log_path,
            remaining(),
        )

        identity = json.loads(id_output.read_text(encoding="utf-8"))
        visual = json.loads(visual_output.read_text(encoding="utf-8"))
        if single_sample_diagnostic:
            visual["fvd"] = None
            visual["fvd_applicability"] = {
                "eligible_count": 1,
                "excluded_count": 0,
                "status": "not_computable",
                "reason": "Frechet distance requires at least two video samples",
            }
        syncnet = json.loads(sync_output.read_text(encoding="utf-8"))
        # Keep raw metric diagnostics on disk, but never publish a partial run as
        # a successful fixed-set result. run_or_wait applies the configured policy.
        validate_metric_results(
            identity,
            visual,
            syncnet,
            self.expected_sample_count,
            required_visual_metrics=("fid",)
            if single_sample_diagnostic
            else ("fid", "fvd"),
        )
        summary = {
            "step": self.step,
            "sample_count": len(paths),
            "id_similarity": identity["id_similarity"],
            "fid": float(visual["fid"]),
            "fvd": (
                float(visual["fvd"])
                if visual["fvd"] is not None
                else None
            ),
            "fvd_applicability": visual.get("fvd_applicability"),
            "sync_c": syncnet["sync_c"],
            "syncnet_coverage": {
                "dataset_count": syncnet["sync_c"].get("dataset_count"),
                "eligible_count": syncnet["sync_c"].get("eligible_count"),
                "success_count": syncnet["sync_c"].get("success_count"),
                "eligible_rate": (
                    syncnet["sync_c"].get("eligible_count", 0) / len(paths)
                    if paths
                    else 0.0
                ),
            },
        }
        self._atomic_json(self.output_dir / "validation_metrics.json", summary)
        scalars = {
            f"{self.metric_prefix}/arcface_id": summary["id_similarity"].get("mean"),
            f"{self.metric_prefix}/fid": summary["fid"],
            f"{self.metric_prefix}/fvd": summary["fvd"],
            f"{self.metric_prefix}/syncnet_c": summary["sync_c"].get("mean"),
        }
        log_metrics({key: value for key, value in scalars.items() if value is not None})
        logger.info(
            f"Validation {self.label or 'fixed8'} benchmark: "
            f"{json.dumps(summary, ensure_ascii=False)}"
        )
        return summary

    def run_or_wait(
        self,
        *,
        is_main_process: bool,
        paths: list[Path],
        prompts: list[str],
        references: list[str],
        log_metrics: Callable[[dict[str, float]], None],
    ) -> bool:
        if not self.enabled:
            return True
        if is_main_process:
            status = {"status": "success", "error": None}
            try:
                self._score(paths, prompts, references, log_metrics)
            except Exception as exc:
                status = {
                    "status": "error",
                    "error": f"validation benchmark failed: {type(exc).__name__}: {exc}",
                }
            self._atomic_json(self.status_path, status)

        deadline = time.monotonic() + int(self.config.metric_timeout_seconds) + 60
        status = None
        while time.monotonic() < deadline:
            try:
                candidate = json.loads(self.status_path.read_text(encoding="utf-8"))
                if candidate.get("status") != "running":
                    status = candidate
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(1.0)
        if status is None:
            status = {
                "status": "error",
                "error": f"timed out waiting for validation benchmark status: {self.status_path}",
            }

        if status.get("status") != "success":
            error = str(status.get("error") or "validation benchmark failed")
            if is_main_process:
                try:
                    self._atomic_json(
                        self.output_dir / "validation_metric_health.json",
                        {
                            "status": "error",
                            "error": error,
                            "step": self.step,
                            "failure_policy": self.config.metric_failure_policy,
                            "run_only": bool(self.config.run_only),
                        },
                    )
                except Exception as exc:
                    logger.warning(f"Failed to persist validation metric health: {exc}")
                try:
                    log_metrics({f"{self.metric_prefix}/failed": 1.0})
                except Exception as exc:
                    logger.warning(f"Failed to log validation failure scalar: {exc}")
            fatal = (
                bool(self.config.run_only)
                or self.config.metric_failure_policy == "error"
            )
            if is_main_process:
                action = "validation will fail" if fatal else "training will continue"
                logger.warning(f"Validation metrics failed; {action}: {error}")
            if fatal:
                raise RuntimeError(error)
            return False

        if is_main_process:
            try:
                self._atomic_json(
                    self.output_dir / "validation_metric_health.json",
                    {
                        "status": "success",
                        "error": None,
                        "step": self.step,
                        "failure_policy": self.config.metric_failure_policy,
                        "run_only": bool(self.config.run_only),
                    },
                )
            except Exception as exc:
                logger.warning(f"Failed to persist validation metric health: {exc}")
            try:
                log_metrics({f"{self.metric_prefix}/failed": 0.0})
            except Exception as exc:
                logger.warning(f"Failed to log validation health scalar: {exc}")
        return True
