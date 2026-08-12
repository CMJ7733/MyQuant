from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, Any, Optional, List, Generic, TypeVar, Type, Union
from pydantic import BaseModel
from famou.core.data import Context, Rollout, RolloutResult
from abc import ABC, abstractmethod
from enum import StrEnum
import re

if TYPE_CHECKING:
    from famou.controller.engine import RolloutEngine
T = TypeVar('T')


@dataclass
class BackendTask:
    """Standard task definition compatible with Evolver._execute_rollout interface.
    
    Supports two task types:
    - "rollout": Standard evolution rollout (Select → Generate → Evaluate → Judge)
    - "enrichment": Enrich existing program (Evaluate → Judge only)
    """
    task_id: str
    experiment_id: str
    island_id: int
    iteration: int
    attempt: int
    context: Context
    rollout: Optional[Rollout]
    engine: "RolloutEngine"  # Use string annotation to avoid runtime import
    priority: int = 0
    resource_requirements: Dict[str, Any] = None
    
    # Enrichment support
    task_type: str = "rollout"  # "rollout" | "enrichment"
    program: Optional[Any] = None  # For enrichment: Program to enrich
    enrichment_modules: Optional[List] = field(default=None)  # For enrichment: modules to apply
    enrichment_rollout_id: Optional[str] = None  # For enrichment: unique rollout ID
    
    @classmethod
    def create(
        cls,
        experiment_id,
        context,
        rollout,
        engine,
        island_id: int,
        iteration: int,
        attempt: int = 1,
    ) -> "BackendTask":
        """Create a new rollout task from given parameters"""

        task_id = f"{experiment_id}-island_{island_id}-iter_{iteration}"
        if attempt > 1:
            task_id = f"{task_id}-attempt_{attempt}"
        
        return cls(
            task_id=task_id,
            experiment_id=experiment_id,
            island_id=island_id,
            iteration=iteration,
            attempt=attempt,
            context=context,
            rollout=rollout,
            engine=engine,
        )
    
    @classmethod
    def create_enrichment(
        cls,
        experiment_id: str,
        context: Context,
        engine: "RolloutEngine",
        program: Any,
        enrichment_modules: List,
        enrichment_rollout_id: str,
        island_id: int = 0,
    ) -> "BackendTask":
        """Create an enrichment task for enriching an existing program.
        
        Args:
            experiment_id: Experiment ID
            context: Experiment context
            engine: RolloutEngine instance
            program: Program to enrich
            enrichment_modules: List of modules to apply (e.g., EvaluateModule, JudgeModule)
            enrichment_rollout_id: Unique rollout ID for this enrichment
            island_id: Island ID (default 0)
        """
        task_id = f"enrichment-{enrichment_rollout_id}"
        
        return cls(
            task_id=task_id,
            experiment_id=experiment_id,
            island_id=island_id,
            iteration=0,
            attempt=1,
            context=context,
            rollout=None,
            engine=engine,
            task_type="enrichment",
            program=program,
            enrichment_modules=enrichment_modules,
            enrichment_rollout_id=enrichment_rollout_id,
        )
    
    @staticmethod
    def get_task_metadata_from_task_id(task_id: str):
        """Extract island ID, iteration number, and attempt from task ID."""
        pattern = r'-island_(\d+)-iter_(\d+)(?:-attempt_(\d+))?$'
        match = re.search(pattern, task_id)
        if not match:
            raise ValueError("task_id format is not as expected")
        island_id = int(match.group(1))
        iteration = int(match.group(2))
        attempt = int(match.group(3)) if match.group(3) else 1
        return island_id, iteration, attempt

    @staticmethod
    def get_island_and_iteration_from_task_id(task_id: str):
        """Backward-compatible helper for callers that only need island+iteration."""
        island_id, iteration, _ = BackendTask.get_task_metadata_from_task_id(task_id)
        return island_id, iteration

class ResultHandle(Generic[T], ABC):
    """Abstract base class for result handles across different backends"""
    
    @abstractmethod
    def result(self, timeout: Optional[float] = None) -> T:
        """Get task result with optional timeout
        
        Args:
            timeout: Maximum time to wait (in seconds), None means no timeout
            
        Returns:
            The task result
        
        Raises:
            Exception: If task execution failed
        """
        pass
    
    @abstractmethod
    def done(self) -> bool:
        """Check if task is completed
        
        Returns:
            bool: True if task is completed, False otherwise
        """
        pass
    
    @abstractmethod
    def cancel(self) -> bool:
        """Attempt to cancel the task
        
        Returns:
            bool: True if cancellation was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def exception(self, timeout: Optional[float] = None) -> Optional[Exception]:
        """Get exception raised by task if any
        
        Args:
            timeout: Maximum time to wait (in seconds), None means no timeout
            
        Returns:
            Exception: The exception raised by task, or None if no exception
        """
        pass

class BatchResultHandle:
    """Batch task result handle that replaces wait+FIRST_COMPLETED logic"""
    def __init__(self, handles: List[ResultHandle], task_ids: List[str]):
        self.handles = handles
        self.task_ids = task_ids
        self._handle_dict = dict(zip(task_ids, handles))
    
    def wait_first_completed(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for first completed task and return result
        
        Simulates behavior of concurrent.futures.wait(return_when=FIRST_COMPLETED)
        
        Args:
            timeout: Maximum time to wait (in seconds), None means no timeout
            
        Returns:
            dict: Contains keys:
                - completed_task: ID of completed task
                - result: Task result (if successful)
                - error: Exception (if task failed)
                - handle: ResultHandle for completed task
                - status: "timeout" if timeout occurred
        """
        import time
        from concurrent.futures import FIRST_COMPLETED
        
        start_time = time.time()
        
        while True:
            # Check for completed tasks
            for task_id, handle in self._handle_dict.items():
                if handle.done():
                    try:
                        result = handle.result(timeout=0)
                        return {"completed_task": task_id, "result": result, "handle": handle}
                    except Exception as e:
                        # FatalExperimentError should propagate to stop the experiment
                        from famou.core.types import FatalExperimentError
                        if isinstance(e, FatalExperimentError):
                            raise
                        return {"completed_task": task_id, "error": e, "handle": handle}
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return {"status": "timeout"}
            
            time.sleep(0.01)  # Avoid busy waiting
    
    def remove_completed(self, task_id: str):
        """Remove completed task from tracking
        
        Args:
            task_id: ID of task to remove
        """
        if task_id in self._handle_dict:
            handle = self._handle_dict.pop(task_id)
            if handle in self.handles:
                self.handles.remove(handle)
            if task_id in self.task_ids:
                self.task_ids.remove(task_id)
    
    def has_pending(self) -> bool:
        """Check if there are any pending tasks
        
        Returns:
            bool: True if there are pending tasks, False otherwise
        """
        return len(self.handles) > 0


class BackendExecutor(ABC):
    """Abstract base class for backend executors designed for Evolver"""
    _registry: Dict[str, Type["BackendExecutor"]] = {}
    
    @abstractmethod
    def submit(self, task: BackendTask) -> ResultHandle[RolloutResult]:
        """Submit single task for execution
        
        Args:
            task: BackendTask to execute
            
        Returns:
            ResultHandle: Handle to track task progress and results
        """
        pass
    
    @abstractmethod
    def submit_batch(self, tasks: List[BackendTask]) -> BatchResultHandle:
        """Submit batch of tasks for execution (initial batch)
        
        Args:
            tasks: List of BackendTasks to execute
            
        Returns:
            BatchResultHandle: Handle to track batch progress
        """
        pass
    
    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor
        
        Args:
            wait: If True, wait for pending tasks to complete
        """
        pass
    
    @abstractmethod
    def get_max_workers(self) -> int:
        """Get maximum number of workers/actors
        
        Returns:
            int: Maximum number of workers
        """
        pass
    
    @abstractmethod
    def start_workers(self, num_workers: int, env_factory=None, llm_client=None, embedding_client=None) -> None:
        """Start specified number of workers/processes/actors

        Args:
            num_workers: Number of workers to start
            env_factory: Optional factory to create per-worker env objects
            llm_client: Optional LLM client whose config is injected into each
                worker at startup, so it becomes a resident object on the worker
                rather than being rebuilt on every task call.
            embedding_client: Optional embedding client for feature extraction modules
        """
        pass

    @classmethod
    def register(cls, name: str):
        """Decorator to register subclasses
        
        Args:
            name: Name to register the subclass under
            
        Returns:
            The decorator function
        """
        def decorator(subclass: Type["BackendExecutor"]):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, config: Union[BaseModel, Any]):
        """Unified factory interface
        
        Args:
            config: Configuration as either dataclass instance or dict
            
        Returns:
            BackendExecutor: Instance of appropriate backend
            
        Raises:
            TypeError: If config is invalid type
            ValueError: If backend type not registered
        """
        # If Pydantic BaseModel is passed, convert to dict
        if isinstance(config, BaseModel):
            config_dict = config.dict()
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise TypeError("config must be a dataclass instance or dict")
        type_name = config_dict.get("mode")
        if type_name not in cls._registry:
            raise ValueError(f"Unregistered type: {type_name}")
        subclass = cls._registry[type_name]

        return subclass(config_dict)
