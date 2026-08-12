"""
Base storage protocol and interfaces.

Defines the interface for data persistence (programs, experiments, results).
"""

from typing import Any, Dict, List, Optional, Protocol

from famou.core.data import Experiment, Program, RolloutResult


class DataService(Protocol):
    """
    Protocol for data persistence.

    DataService handles storage and retrieval of:
    - Programs (individual solutions)
    - RolloutResults (iteration outcomes)
    - Experiments (overall state)
    - Context (experiment configuration)

    Implementations can use:
    - Local files (JSONL, JSON)
    - Databases (SQLite, PostgreSQL)
    - Cloud storage (S3, GCS)
    - In-memory (for testing)

    Example Implementation:
        >>> class MyDataService:
        ...     def save_program(self, program):
        ...         with open(f"{program.id}.json", "w") as f:
        ...             f.write(program.model_dump_json())
        ...
        ...     def load_program(self, program_id):
        ...         with open(f"{program_id}.json") as f:
            ...             return Program.model_validate_json(f.read())

    Example Usage:
        >>> storage: DataService = LocalStorage("./data")
        >>> storage.save_program(my_program)
        >>> storage.save_result(rollout_result)
        >>> loaded = storage.load_program(my_program.id)
    """

    def save_program(self, program: Program) -> None:
        """
        Save a program.

        Args:
            program: Program to save
        """
        ...

    def load_program(self, program_id: str) -> Program:
        """
        Load a program by ID.

        Args:
            program_id: ID of program to load

        Returns:
            Program instance

        Raises:
            FileNotFoundError: If program not found
        """
        ...

    def list_programs(
        self,
        experiment_id: str,
        iteration: Optional[int] = None,
        generation: Optional[int] = None,
    ) -> List[Program]:
        """
        List programs matching filters.

        Args:
            experiment_id: Experiment ID to filter by
            iteration: Optional iteration to filter by
            generation: Optional generation to filter by

        Returns:
            List of matching programs
        """
        ...

    def save_result(self, result: RolloutResult) -> None:
        """
        Save a rollout result.

        Args:
            result: RolloutResult to save
        """
        ...

    def load_result(self, rollout_id: str) -> RolloutResult:
        """
        Load a rollout result by ID.

        Args:
            rollout_id: ID of rollout to load

        Returns:
            RolloutResult instance

        Raises:
            FileNotFoundError: If result not found
        """
        ...

    def save_experiment(self, experiment: Experiment) -> None:
        """
        Save experiment state (checkpoint).

        Saves to experiment_checkpoint_{iteration}.json where iteration is
        taken from experiment.current_iteration.

        Args:
            experiment: Experiment to save
        """
        ...

    def load_experiment(self, experiment_id: str, iteration: int = -1) -> Experiment:
        """
        Load experiment state.

        Args:
            experiment_id: Experiment ID to load
            iteration: Iteration number to load (default: -1 for latest checkpoint)

        Returns:
            Experiment instance

        Raises:
            FileNotFoundError: If experiment not found
        """
        ...

    def save_json_artifact(
        self,
        experiment_id: str,
        filename: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        Save an auxiliary JSON artifact under the experiment directory.

        Args:
            experiment_id: Experiment ID
            filename: Artifact filename
            payload: JSON-serializable payload
        """
        ...

    llm_request_log_dir: Optional[str]

    """
    Dedicated directory for llm_requests.log.

    When set, llm_requests.log will be written to this directory instead of
    the default experiment directory.  Defaults to None, which means the log
    goes into the experiment directory derived from storage paths.
    """

