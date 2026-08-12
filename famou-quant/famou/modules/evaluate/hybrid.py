"""
Hybrid Evaluation Module for Famou 2.0.

Evaluates programs using remote evaluator service via famou_sdk.
"""

import asyncio
import os
import subprocess
import tempfile
import time
import logging
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional

import famou_sdk
import httpx

from famou.core.data import Context, RolloutResult
from famou.config.settings import EvaluatorConfig
from famou.core.types import FatalRolloutError
from famou.modules.evaluate.base import EvaluateModule

logger = logging.getLogger("famou")


class HybridEvaluateModule(EvaluateModule):
    """
    Hybrid evaluation module using remote evaluator service.

    Inherits from EvaluateModule and overrides evaluation logic to use
    famou_sdk.FamouClient for remote evaluation.

    Features:
    - Dual-layer timeout protection (system + user)
    - Network retry with exponential backoff
    - Status polling until completion

    Args:
        config: EvaluatorConfig with service_url, api_key, etc.
        name: Module name (default: class name)
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        name: Optional[str] = None,
    ):
        # No evaluate_fn needed for hybrid mode
        super().__init__(name=name, evaluate_fn=None)
        self.evaluator_config = config
        self.client = self._create_client()

    def _resolve_timeout(self) -> int:
        """Resolve evaluation timeout from EnvConfig.timeout (via self.env.default_timeout)."""
        timeout = getattr(self.env, "default_timeout", None)
        if timeout is None:
            logger.warning("No timeout available from env, defaulting to 30s")
            timeout = 30
        return timeout

    def _create_client(self):
        """Create famou_sdk client."""
        return famou_sdk.FamouClient(
            self.evaluator_config.service_url,
            self.evaluator_config.api_key
        )

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """Validate that a program is available."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: No program to evaluate. "
                "Make sure GenerateModule has run first."
            )

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """Execute program using remote evaluator service."""
        print("[hybrid debug] start to evaluate program")
        program = result.generated_program

        # Perform security scan before evaluation (optional, non-blocking)
        try:
            scan_result = self._security_scan(program.code, program.id)
            if scan_result["has_issues"]:
                severity_counts = scan_result["severity_counts"]
                if severity_counts["ERROR"] > 0:
                    self.log_warning(
                        f"Security scan found {severity_counts['ERROR']} ERROR level issues "
                        f"for program {program.id}"
                    )
        except Exception as e:
            self.log_warning(f"Security scan failed, continuing with evaluation: {str(e)}")

        try:
            self.log_info(f"Evaluating {program.id} via HybridEvaluator")

            # Resolve timeout from EnvConfig.timeout (with env var override)
            timeout = self._resolve_timeout()
            print(f"HybridEvaluator resolved timeout: {timeout}s")

            # Run async evaluation in sync context
            eval_result = asyncio.run(self._evaluate_async(program.code, timeout))

            # Use parent class result processing
            self._process_result(program, eval_result)

        except Exception as e:
            if self._is_fatal_error(e):
                # Populate error_info before raising so the archive record reflects the failure
                self._handle_error(program, e)
                raise FatalRolloutError(
                    f"HybridEvaluator service error: {str(e)}. "
                    f"Please check service_url, api_key, experiment_id, and network connectivity."
                ) from e

            # Other errors: mark program as invalid but continue
            self._handle_error(program, e)

        return result

    async def _evaluate_async(self, program_code: str, timeout: int) -> Dict[str, Any]:
        """
        Async evaluation with dual-layer timeout polling.

        Args:
            program_code: The code to evaluate
            timeout: Evaluation timeout in seconds (from EnvConfig.timeout)

        Reference: famou HybridEvaluator implementation
        """
        evaluation_id = str(uuid.uuid4())

        # 1. Create evaluation task with retry
        await self._create_evaluation_with_retry(evaluation_id, program_code, timeout)

        # 2. Dual-layer timeout polling
        start_time = time.time()
        running_start_time = None
        prev_retry = 0
        polling_count = 0

        system_protection_timeout = self.evaluator_config.system_protection_timeout
        user_configured_timeout = timeout + self.evaluator_config.user_timeout_delta

        while True:
            current_time = time.time()

            # System protection timeout check
            if current_time - start_time > system_protection_timeout:
                raise TimeoutError(f"System timeout: pending over {system_protection_timeout}s")

            # User configured timeout check
            if running_start_time and current_time - running_start_time > user_configured_timeout:
                timeout_msg = f"User timeout: exceeded {user_configured_timeout}s"
                logger.warning(f"Evaluation {evaluation_id} {timeout_msg}")
                return {
                    "combined_score": 0.0,
                    "validity": 0.0,
                    "error_info": timeout_msg,
                }

            # Get evaluation status with retry
            resp_data = await self._get_evaluation_with_retry(evaluation_id)
            polling_count += 1

            # Log status periodically
            if polling_count % 30 == 0 or resp_data.get("status") != "running":
                logger.debug(f"Evaluation status: {resp_data}")

            current_status = resp_data.get("status", "unknown")

            # Detect status change to running
            if current_status == "running" and running_start_time is None:
                running_start_time = current_time
                logger.info(
                    f"Evaluation {evaluation_id} started running, "
                    f"user timeout: {user_configured_timeout}s"
                )

            # Reset timer when retry parameter changes
            if current_status == "running":
                current_retry = resp_data.get("retry", 0)
                if current_retry > prev_retry:
                    running_start_time = current_time
                    logger.info(
                        f"Retry parameter changed from {prev_retry} to {current_retry}, "
                        f"reset user timeout timer"
                    )
                    prev_retry = current_retry

            # Check completion status
            if current_status in {"finished", "failed"}:
                if current_status == "finished" and resp_data.get("result"):
                    result = json.loads(resp_data["result"])
                    if isinstance(result, dict) and "metrics" in result:
                        return result["metrics"]
                    return result
                elif current_status == "failed":
                    raise RuntimeError(f"Evaluation failed: {resp_data}")
                break

            await asyncio.sleep(self.evaluator_config.polling_interval)

        raise RuntimeError("Evaluation did not complete")

    async def _create_evaluation_with_retry(
        self,
        evaluation_id: str,
        program_code: str,
        timeout: int,
    ) -> None:
        """Create evaluation task with retry."""
        last_exception = None

        for attempt in range(self.evaluator_config.network_retry_attempts):
            try:
                logger.debug(
                    f"Attempting evaluation_create (attempt {attempt + 1}/"
                    f"{self.evaluator_config.network_retry_attempts})"
                )

                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.evaluation_create(
                            evaluation_id=evaluation_id,
                            exp_id=self.evaluator_config.experiment_id,
                            code=program_code,
                            timeout=timeout
                        )
                    ),
                    timeout=self.evaluator_config.network_timeout
                )

                logger.debug(f"evaluation_create succeeded on attempt {attempt + 1}")
                return

            except asyncio.TimeoutError:
                last_exception = TimeoutError(
                    f"evaluation_create timeout after {self.evaluator_config.network_timeout}s"
                )
                logger.warning(f"evaluation_create timeout on attempt {attempt + 1}")
            except famou_sdk.FamouAPIError as e:
                if e.code == "EvalutionIdExist":
                    logger.info(f"Evaluation {evaluation_id} already exists, ignore")
                    return
                last_exception = e
                logger.warning(f"evaluation_create failed on attempt {attempt + 1}: {str(e)}")
            except Exception as e:
                last_exception = e
                logger.warning(f"evaluation_create failed on attempt {attempt + 1}: {str(e)}")

            # Exponential backoff before retry
            if attempt < self.evaluator_config.network_retry_attempts - 1:
                await asyncio.sleep(
                    self.evaluator_config.network_retry_delay * (2 ** attempt)
                )

        # All retries failed
        error_msg = (
            f"evaluation_create failed after {self.evaluator_config.network_retry_attempts} "
            f"attempts: {str(last_exception)}"
        )
        logger.error(error_msg)
        if last_exception is None:
            raise RuntimeError("evaluation_create failed with unknown error")
        raise last_exception

    async def _get_evaluation_with_retry(self, evaluation_id: str) -> Dict:
        """Get evaluation status with retry."""
        last_exception = None

        for attempt in range(self.evaluator_config.network_retry_attempts):
            try:
                logger.debug(
                    f"Attempting evaluation_get (attempt {attempt + 1}/"
                    f"{self.evaluator_config.network_retry_attempts})"
                )

                resp = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.evaluation_get(
                            evaluation_id,
                            self.evaluator_config.experiment_id
                        )
                    ),
                    timeout=self.evaluator_config.network_timeout
                )

                logger.debug(f"evaluation_get succeeded on attempt {attempt + 1}")
                # Convert response to dict
                if hasattr(resp, 'data'):
                    return asdict(resp.data)
                return resp

            except asyncio.TimeoutError:
                last_exception = TimeoutError(
                    f"evaluation_get timeout after {self.evaluator_config.network_timeout}s"
                )
                logger.warning(f"evaluation_get timeout on attempt {attempt + 1}")
            except Exception as e:
                last_exception = e
                logger.warning(f"evaluation_get failed on attempt {attempt + 1}: {str(e)}")

            # Exponential backoff before retry
            if attempt < self.evaluator_config.network_retry_attempts - 1:
                await asyncio.sleep(
                    self.evaluator_config.network_retry_delay * (2 ** attempt)
                )

        # All retries failed
        error_msg = (
            f"evaluation_get failed after {self.evaluator_config.network_retry_attempts} "
            f"attempts: {str(last_exception)}"
        )
        logger.error(error_msg)
        raise last_exception

    def _is_fatal_error(self, error: Exception) -> bool:
        """Return True when the error means the experiment cannot continue."""
        return isinstance(
            error,
            (
                TimeoutError,
                ConnectionError,
                OSError,
                httpx.HTTPError,
                famou_sdk.FamouSDKError,
            ),
        )

    def _process_result(self, program, eval_result: Dict[str, Any]) -> None:
        """Process evaluation result and enrich program."""
        if "combined_score" not in eval_result:
            raise ValueError(
                f"Evaluator must return 'combined_score'. "
                f"Got keys: {list(eval_result.keys())}"
            )
        if "validity" not in eval_result:
            raise ValueError(
                f"Evaluator must return 'validity'. "
                f"Got keys: {list(eval_result.keys())}"
            )

        # Extract metrics
        reserved_keys = {"combined_score", "validity", "error_info"}
        program.metrics = {
            k: v for k, v in eval_result.items()
            if k not in reserved_keys
        }

        # Set results
        program.combined_score = eval_result["combined_score"]
        program.validity = eval_result["validity"]
        error_info = eval_result.get("error_info")
        if error_info is not None and not isinstance(error_info, str):
            error_info = json.dumps(error_info, ensure_ascii=False)
        program.error_info = error_info

        self.log_info(
            f"Evaluated {program.id}",
            program_id=program.id,
            combined_score=program.combined_score,
            validity=program.validity,
        )

    def _handle_error(self, program, error: Exception) -> None:
        """Handle evaluation error."""
        program.validity = 0.0
        program.error_info = f"HybridEvaluator error: {str(error)}"
        program.combined_score = 0.0

        self.log_error(
            f"HybridEvaluator failed for {program.id}: {error}",
            program_id=program.id,
        )

        if program.error_info and len(program.error_info) > 20000:
            program.error_info = program.error_info[:20000]

    # =========================================================================
    # Security Scan Methods
    # =========================================================================

    def _security_scan(self, program_code: str, program_id: str = "") -> Dict:
        """
        Perform security scan on program code using semgrep.

        Args:
            program_code: The code to scan
            program_id: Program identifier for logging

        Returns:
            Dictionary containing scan results with severity breakdown
        """
        scan_result = {
            "has_issues": False,
            "severity_counts": {"ERROR": 0, "WARNING": 0, "INFO": 0},
            "findings": []
        }

        # Check if semgrep is available
        try:
            result = subprocess.run(
                ["semgrep", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                if self.logger:
                    self.logger.debug("semgrep is not installed or not available in PATH")
                return scan_result
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            if self.logger:
                self.logger.debug(f"semgrep not available: {str(e)}")
            return scan_result

        # Find security rules directory
        security_rules_path = self._get_security_rules_path()
        if not security_rules_path:
            if self.logger:
                self.logger.debug("Security rules directory not found, skipping scan")
            return scan_result

        # Create temporary file for scanning
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(program_code)
            temp_file_path = temp_file.name

        try:
            if self.logger:
                self.logger.debug(f"Starting security scan for program {program_id}")

            # Use the entire security_rules directory as config
            # This scans all yaml/yml rule files in the directory
            result = subprocess.run(
                [
                    "semgrep", "scan",
                    "--config", str(security_rules_path),
                    "--json",
                    "--quiet",
                    "--no-git-ignore",
                    temp_file_path
                ],
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout for scanning
            )

            # Parse semgrep JSON output
            if result.stdout:
                self._parse_semgrep_output(result.stdout, scan_result)

            # Log security scan results
            self._log_security_report(scan_result, program_id)

        except subprocess.TimeoutExpired:
            if self.logger:
                self.logger.warning(f"Security scan timeout for program {program_id}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Security scan failed for program {program_id}: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

        return scan_result

    def _get_security_rules_path(self) -> Optional[Path]:
        """Find security rules directory."""
        # security_rules is in the same directory as hybrid.py
        security_rules_path = Path(__file__).parent / "security_rules"
        if security_rules_path.exists() and security_rules_path.is_dir():
            return security_rules_path

        return None

    def _parse_semgrep_output(self, stdout: str, scan_result: Dict) -> None:
        """Parse semgrep JSON output and update scan_result."""
        try:
            semgrep_output = json.loads(stdout)
            results = semgrep_output.get("results", [])

            for finding in results:
                check_result = finding.get("extra", {})
                severity = check_result.get("severity", "INFO").upper()

                finding_info = {
                    "rule_id": finding.get("check_id", "unknown"),
                    "message": check_result.get("message", ""),
                    "severity": severity,
                    "file": finding.get("path", ""),
                    "line": finding.get("start", {}).get("line", 0),
                    "code_snippet": check_result.get("lines", "")
                }

                scan_result["findings"].append(finding_info)

                # Update severity counts
                if severity in scan_result["severity_counts"]:
                    scan_result["severity_counts"][severity] += 1

            if results:
                scan_result["has_issues"] = True

        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.warning(f"Failed to parse semgrep output: {str(e)}")

    def _log_security_report(self, scan_result: Dict, program_id: str) -> None:
        """Log security scan results."""
        if not self.logger:
            return

        if not scan_result["has_issues"]:
            self.logger.info(f"Security scan for {program_id}: No issues found")
            return

        severity_counts = scan_result["severity_counts"]
        total_issues = sum(severity_counts.values())

        self.logger.info(
            f"Security scan for {program_id}: "
            f"{total_issues} issues found "
            f"(ERROR: {severity_counts['ERROR']}, "
            f"WARNING: {severity_counts['WARNING']}, "
            f"INFO: {severity_counts['INFO']})"
        )
