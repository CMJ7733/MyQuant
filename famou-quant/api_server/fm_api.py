"""
Famou API 异步队列框架
基于 FastAPI + asyncio.Queue + 单 Worker 协程
API 接口协议与 fm_api.py 完全对齐
异步任务执行调用 famou.main() 或 famou.controller.Evolver
"""

import asyncio
import logging
import os
import signal
import sys
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from report import build_report
from famou.controller import Evolver
from famou.infrastructure.factory import InfraFactory
from famou.controller.evaluator_dependency import prepare_evaluator_environment
from famou.main import (
    load_config,
    load_evaluator,
    load_initial_programs,
    persist_strategy_route_decision,
    persist_strategy_route_trace,
    resolve_strategy,
)
from famou.utils import generate_experiment_id

# ==================== 日志配置 ====================
api_details_logger = logging.getLogger("api_details")
api_details_logger.setLevel(logging.INFO)
api_details_logger.propagate = False

llm_logger = logging.getLogger("llm_requests")
llm_logger.setLevel(logging.INFO)
llm_logger.propagate = False

WORKSPACE = os.getenv("WORKSPACE", default="/workspace")
MAX_CONCURRENT_TASKS = 2

# ==================== Pydantic 请求模型 ====================
class BaseRequest(BaseModel):
    """基础 worker 请求模型"""
    job_id: str = Field(..., description="Job ID")
    worker_id: str = Field(..., description="Worker ID")

class InitRequest(BaseRequest):
    """初始化请求模型"""
    config_path: str = Field(..., description="配置文件路径")

class EvolveRequest(BaseRequest):
    """进化请求模型"""
    iterations: int = Field(default=None, description="本次要向后迭代的次数")
    config_path: Optional[str] = Field(default=None, description="配置文件路径，若提供则自动初始化（不运行init阶段）")

class ConfigUpdateRequest(BaseRequest):
    """配置更新请求模型"""
    check_id: str = Field(..., description="校验 ID")

# ==================== 创建 FastAPI 应用 ====================
app = FastAPI()

# ==================== 任务队列和 Worker ====================
task_queue: asyncio.Queue = None
worker_tasks: list[asyncio.Task] = []
is_running: bool = False
shutdown_event: asyncio.Event = None  # 用于优雅退出（在 startup 中初始化）

# ==================== 全局异常处理 ====================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """处理请求校验错误，返回统一的错误格式"""
    if exc.errors():
        error = exc.errors()[0]
        loc = error.get("loc", [])
        field = loc[-1] if loc else ""
        msg = error.get("msg", "Validation error")
        if field and "missing" in error.get("type", ""):
            error_msg = f"Missing {field}"
        else:
            error_msg = f"{field}: {msg}" if field else msg
    else:
        error_msg = "Validation error"

    # 记录请求校验错误日志
    if any(isinstance(h, logging.FileHandler) for h in api_details_logger.handlers):
        log_message = (
            f'{request.client.host}:{request.client.port} - '
            f'"{request.method} {request.url.path} HTTP/{request.scope["http_version"]}" '
            f'Validation Error: {error_msg}'
        )
        api_details_logger.warning(log_message)

    return JSONResponse(
        status_code=200,
        content={"code": "-1", "msg": error_msg},
    )

# ==================== 异常处理中间件 ====================
@app.middleware("http")
async def exception_middleware(request: Request, call_next):
    """全局异常捕获中间件"""
    try:
        response = await call_next(request)
        return response
    except ValueError as e:
        return JSONResponse(
            status_code=200,
            content={"code": "-1", "msg": "Value error: " + str(e)},
        )
    except Exception as e:
        print(e)
        return Response(
            content='{"code": "-1", "msg": "Internal server error: ' + str(e) + '"}',
            status_code=500,
            media_type="application/json",
        )

# ==================== 日志中间件 ====================
@app.middleware("http")
async def log_api_details_middleware(request: Request, call_next):
    """日志中间件，记录请求和响应（外层中间件，在异常中间件之上，确保异常响应也能被记录）"""
    start_time = time.monotonic()
    request_body_bytes = await request.body()
    request_body_str = request_body_bytes.decode('utf-8', errors='ignore')

    response = await call_next(request)

    duration_ms = (time.monotonic() - start_time) * 1000
    response_body_bytes = b""
    async for chunk in response.body_iterator:
        response_body_bytes += chunk

    final_response = Response(
        content=response_body_bytes,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )

    # 控制台输出带耗时（替代 uvicorn access log）
    print(
        f'INFO:     {request.client.host}:{request.client.port} - '
        f'"{request.method} {request.url.path} HTTP/{request.scope["http_version"]}" '
        f'{response.status_code} [{duration_ms:.0f}ms]'
    )

    if any(isinstance(h, logging.FileHandler) for h in api_details_logger.handlers):
        response_body_str = response_body_bytes.decode('utf-8', errors='ignore')
        log_message = (
            f'{request.client.host}:{request.client.port} - '
            f'"{request.method} {request.url.path} HTTP/{request.scope["http_version"]}" '
            f'Request: {request_body_str} '
            f'{response.status_code} '
            f'Response: {response_body_str}'
        )
        api_details_logger.info(log_message)

    return final_response

# ==================== CORS 中间件 ====================
if os.environ.get("DEBUG", "False") == "True":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ==================== 日志设置函数 ====================
def set_logger(experiment_root: str):
    """根据 experiment_root 配置日志"""
    for handler in api_details_logger.handlers[:]:
        handler.close()
        api_details_logger.removeHandler(handler)

    job_log_dir = os.path.join(experiment_root, "logs")
    os.makedirs(job_log_dir, exist_ok=True)
    new_log_path = os.path.join(job_log_dir, "api.log")

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(new_log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    api_details_logger.addHandler(file_handler)
    api_details_logger.info(f"API details logger is now writing to: {new_log_path}")


# ==================== 任务状态和信息类 ====================
class TaskInfo:
    """用于存储任务相关信息的容器类"""
    job_id: str = ""
    worker_id: str = ""
    task: asyncio.Task = None
    stage: str = ""
    status: str = ""
    hero_task: asyncio.Task = None
    front_task: asyncio.Task = None
    cancel_task: asyncio.Task = None
    is_canceling: bool = False
    init_program: Any = None
    init_database: Any = None
    result: Any = None
    error_msg: str = ""

    # Famou 2.0 相关对象
    evolver: Any = None
    infra: Any = None
    config: Any = None
    strategy: Any = None
    experiment_id: str = ""
    config_path: str = ""
    _diversity_cache: float = 0.0       # 多样性缓存值
    _diversity_archive_size: int = 0    # 上次计算时的 archive 大小

    @property
    def experiment_root(self) -> str:
        """从 config_path 提取实验根路径，兜底返回 WORKSPACE。"""
        if not self.config_path:
            return WORKSPACE
        root = _extract_experiment_root(self.config_path)
        return root if root else WORKSPACE

TASKS: dict[str, TaskInfo] = {}
LOGGER_FLAG: bool = False


# ==================== 状态枚举 ====================
class Stage(StrEnum):
    """阶段枚举"""
    INIT = "init"
    COLDSTART = "coldstart"
    EVOLVE = "evolve"
    REPORT = "report"

class Status(StrEnum):
    """状态枚举"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    CANCELING = "canceling"


# ==================== 任务队列处理 ====================
async def process_task(task_data: dict):
    """
    处理队列中的任务，调用 famou 2.0 实际逻辑
    """
    task_type = task_data.get("type")
    worker_id = task_data.get("worker_id")

    task_info = TASKS.get(worker_id)
    if not task_info:
        return

    try:
        if task_type == "init":
            await _run_init_task(task_data, task_info)
        elif task_type == "coldstart":
            await _run_coldstart_task(task_data, task_info)
        elif task_type == "evolve":
            await _run_evolve_task(task_data, task_info)
            # evolve 正常完成后自动入队 report，由 worker_loop 统一管理生命周期
            if task_info.status == Status.COMPLETED:
                llm_config = None
                try:
                    llm_config = task_info.config.infrastructure.llm
                except AttributeError:
                    pass
                if llm_config:
                    await task_queue.put({"type": "report", "worker_id": worker_id})
                    print(f"Report task queued for worker {worker_id}")
        elif task_type == "front":
            await _run_front_task(task_data, task_info)
        elif task_type == "report":
            await _run_report_task(task_data, task_info)
        else:
            print(f"Unknown task type: {task_type}")
            task_info.status = Status.FAILED
    except Exception as e:
        import traceback
        print(f"Task {task_type} failed for worker {worker_id}: {e}")
        print(traceback.format_exc())
        task_info.status = Status.FAILED
        task_info.error_msg = str(e)


def _resolve_relative_path(path: str, base_dir: str) -> str:
    """将相对路径基于 base_dir 解析为绝对路径，已是绝对路径则原样返回"""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _extract_experiment_root(config_path: str) -> str:
    """
    从 config_path 提取实验根路径。

    路径格式: {root}/system/{worker_id}/input/config_*.yaml
    返回: {root}，即 {WORKSPACE}/{user_id}/{job_id}

    示例:
        config_path = /workspace/mock_user/exp_id/system/v1/input/config.yaml
        返回         = /workspace/mock_user/exp_id
    """
    parts = Path(config_path).parts
    for i, part in enumerate(parts):
        if part == "system":
            return str(Path(*parts[:i]))
    # fallback: 3 级父目录
    return str(Path(config_path).parent.parent.parent)


def _build_output_dir(experiment_root: str, worker_id: str, mode: str = "system") -> str:
    """
    构建输出目录路径

    Args:
        experiment_root: 实验根路径，即 {WORKSPACE}/{user_id}/{job_id}
        worker_id: 工作节点ID
        mode: 输出模式，默认为"system"

    Returns:
        格式化后的输出目录路径
    """
    return os.path.join(experiment_root, mode, worker_id, "output")


def _setup_init_task(config_path: str) -> tuple:
    """同步准备 init 所需的所有对象（在线程池中执行，避免阻塞事件循环）"""
    config = load_config(config_path)

    # 将 config_dir 作为基目录，把配置中的所有相对路径解析为绝对路径
    # 这样下游组件（LocalStorage、DualWriteLocalStorage、Logger 等）不再依赖进程 cwd
    config_dir = os.path.dirname(os.path.abspath(config_path))

    # 预解析基础设施中的相对路径
    config.infrastructure.storage.base_path = _resolve_relative_path(
        config.infrastructure.storage.base_path, config_dir
    )
    if config.infrastructure.storage.user_path:
        config.infrastructure.storage.user_path = _resolve_relative_path(
            config.infrastructure.storage.user_path, config_dir
        )
    if config.infrastructure.storage.precise_path:
        config.infrastructure.storage.precise_path = _resolve_relative_path(
            config.infrastructure.storage.precise_path, config_dir
        )
    if config.infrastructure.storage.precise_user_path:
        config.infrastructure.storage.precise_user_path = _resolve_relative_path(
            config.infrastructure.storage.precise_user_path, config_dir
        )
    if config.infrastructure.storage.llm_request_log_dir:
        config.infrastructure.storage.llm_request_log_dir = _resolve_relative_path(
            config.infrastructure.storage.llm_request_log_dir, config_dir
        )
    if config.infrastructure.logger.log_dir:
        config.infrastructure.logger.log_dir = _resolve_relative_path(
            config.infrastructure.logger.log_dir, config_dir
        )

    # 预解析实验配置中的相对路径
    if config.experiment.programs:
        resolved = []
        for p in config.experiment.programs.split(","):
            p = p.strip()
            if p:
                resolved.append(_resolve_relative_path(p, config_dir))
        config.experiment.programs = ",".join(resolved)

    if config.experiment.evaluator:
        config.experiment.evaluator = _resolve_relative_path(
            config.experiment.evaluator, config_dir
        )

    experiment_id = generate_experiment_id(config.experiment.name)

    infra = InfraFactory.create_all(
        config.infrastructure,
        experiment_name=config.experiment.name,
        experiment_id=experiment_id,
    )
    infra.storage.save_config(experiment_id, config)

    # Resolve programs: config.programs > default "init.py"
    if config.experiment.programs:
        program_paths = [p.strip() for p in config.experiment.programs.split(",") if p.strip()]
    else:
        program_paths = [os.path.join(config_dir, "init.py")]

    initial_programs = load_initial_programs(
        program_paths, experiment_id, config.experiment.language
    )

    # Resolve evaluator: config.evaluator > default "evaluator.py"
    if config.experiment.evaluator:
        evaluator_path = config.experiment.evaluator
    else:
        evaluator_path = os.path.join(config_dir, "evaluator.py")

    use_hybrid_evaluator = (
        config.infrastructure.evaluator is not None
        and config.infrastructure.evaluator.type == "hybrid"
    )

    if use_hybrid_evaluator:
        # hybrid 模式：evaluation 在远端服务执行，无需本地依赖安装
        evaluate_fn = None
    else:
        trace = prepare_evaluator_environment(
            evaluator_path=evaluator_path,
            llm_client=infra.llm,
            env=infra.env,
            storage=infra.storage,
            experiment_id=experiment_id,
            logger=infra.logger,
            prefer_env_install=str(config.infrastructure.env.type).lower() in {"local", "process"},
        )
        evaluate_fn = load_evaluator(
            evaluator_path,
            required_packages=trace["decision"]["final_packages"],
        )

    strategy, route_decision = resolve_strategy(
        config,
        infra,
        evaluate_fn,
    )

    evolver = Evolver(
        config=config.experiment,
        population_module=strategy.population_module,
        initial_programs=initial_programs,
        logger=infra.logger,
        llm_client=infra.llm,
        env=infra.env,
        data_service=infra.storage,
        experiment_id=experiment_id,
        embedding_client=infra.embedding,
        monitor=infra.monitor,
        langfuse=infra.langfuse,
        backend=infra.backend,
        evaluator_path=evaluator_path,
    )

    trace_filename = persist_strategy_route_trace(
        infra.storage,
        evolver.experiment.id,
        route_decision,
        logger=infra.logger,
    )
    persist_strategy_route_decision(
        evolver.experiment,
        route_decision,
        logger=infra.logger,
        trace_filename=trace_filename,
    )

    return config, infra, strategy, evolver, experiment_id, evaluate_fn


async def _run_init_task(task_data: dict, task_info: "TaskInfo"):
    """执行初始化任务"""
    config_path = task_data.get("config_path")
    worker_id = task_info.worker_id

    # 准备阶段放入线程池，避免阻塞事件循环
    try:
        config, infra, strategy, evolver, experiment_id, evaluate_fn = await asyncio.to_thread(
            _setup_init_task, config_path
        )
    except asyncio.CancelledError:
        task_info.status = Status.CANCELLED
        print(f"Init setup cancelled for worker {worker_id}")
        raise

    task_info.evolver = evolver
    task_info.infra = infra
    task_info.config = config
    task_info.strategy = strategy
    task_info.experiment_id = experiment_id
    task_info.evaluate_fn = evaluate_fn

    # 运行 init（仅 enrichment，不运行 evolution）
    try:
        await asyncio.to_thread(evolver.run, strategy, "init")
    except asyncio.CancelledError:
        task_info.status = Status.CANCELLED
        print(f"Init task cancelled for worker {worker_id}")
        raise

    # 续跑任务（初始解来自历史实验），跳过 error_info 校验
    is_historical = config.experiment.production_info.get("init_source") == "historical"

    # 检查 archive 中所有初始程序是否全部评估失败
    # EvaluateModule 会捕获异常并将 error_info 置为错误信息，而不重新抛出
    # 若所有程序均有 error_info，说明 evaluator 存在系统性故障（如缺少依赖）
    archive = evolver.experiment.archive

    def _has_error(error_info):
        return error_info is not None and error_info != "" and error_info != "{}"

    if archive and not is_historical:
        import math

        programs = list(archive.values())

        # 1. 所有程序均有 error_info → evaluator 系统性故障
        failed_programs = [p for p in programs if _has_error(p.error_info)]
        if len(failed_programs) == len(programs):
            first_error = failed_programs[0].error_info.split('\n')[0]
            task_info.status = Status.FAILED
            task_info.error_msg = f"All initial programs failed evaluation: {first_error}"
            print(f"Init task failed for worker {worker_id}: {task_info.error_msg}")
            return

        # 2. 所有程序 validity == 0.0 → 无可行初始解
        if all(p.validity is not None and p.validity == 0.0 for p in programs):
            task_info.status = Status.FAILED
            task_info.error_msg = "All initial programs have validity=0.0 (invalid solutions)"
            print(f"Init task failed for worker {worker_id}: {task_info.error_msg}")
            return

        # 3. 所有程序 combined_score 无效（None / inf / nan）→ 无可用评分
        def _is_invalid_score(score):
            return (score is None
                    or isinstance(score, (list, tuple, set, dict, str, bytes, bytearray))
                    or (isinstance(score, float) and (math.isinf(score) or math.isnan(score))))

        if all(_is_invalid_score(p.combined_score) for p in programs):
            task_info.status = Status.FAILED
            task_info.error_msg = "All initial programs have invalid scores"
            print(f"Init task failed for worker {worker_id}: {task_info.error_msg}")
            return

    task_info.status = Status.COMPLETED
    print(f"Init task completed for worker {worker_id}, experiment_id: {experiment_id}")


async def _run_coldstart_task(task_data: dict, task_info: "TaskInfo"):
    """执行 coldstart 任务"""
    # Coldstart 仅运行 evolution 循环（跳过 init/enrichment）
    if hasattr(task_info, 'evolver') and task_info.evolver:
        try:
            final_experiment = await asyncio.to_thread(task_info.evolver.run, task_info.strategy, "evolution")
        except asyncio.CancelledError:
            task_info.status = Status.CANCELLED
            print(f"Coldstart task cancelled for worker {task_info.worker_id}")
            raise
        task_info.status = Status.COMPLETED
        task_info.result = list(final_experiment.archive.values())
    else:
        # 如果没有 evolver，需要重新初始化
        task_info.status = Status.FAILED
        print(f"Coldstart task failed: no evolver found for worker {task_info.worker_id}")


async def _run_evolve_task(task_data: dict, task_info: "TaskInfo"):
    """执行 evolve 任务"""
    config_path = task_data.get("config_path")

    # 若提供 config_path，仿照 init 加载配置和创建对象，但不运行 evolver.run("init")
    if config_path:
        try:
            config, infra, strategy, evolver, experiment_id, evaluate_fn = await asyncio.to_thread(
                _setup_init_task, config_path
            )
        except asyncio.CancelledError:
            task_info.status = Status.CANCELLED
            print(f"Evolve setup cancelled for worker {task_info.worker_id}")
            raise
        except Exception as e:
            task_info.status = Status.FAILED
            task_info.error_msg = f"failed to setup from config_path: {e}"
            print(f"Evolve setup failed for worker {task_info.worker_id}: {e}")
            return

        task_info.evolver = evolver
        task_info.infra = infra
        task_info.config = config
        task_info.strategy = strategy
        task_info.experiment_id = experiment_id
        task_info.evaluate_fn = evaluate_fn

    is_resume = task_data.get("resume", False)

    if is_resume:
        # 仿照 main.py resume 模式：从 checkpoint 加载 experiment，重建 Evolver
        try:
            experiment = await asyncio.to_thread(
                task_info.infra.storage.load_experiment,
                task_info.experiment_id,
            )
        except Exception as e:
            task_info.status = Status.FAILED
            task_info.error_msg = f"failed to load checkpoint: {e}"
            print(f"Evolve resume failed for worker {task_info.worker_id}: {e}")
            return

        # 重新创建 backend 对象，避免旧线程的 finally 块调用 shutdown() 关闭新 executor
        # （旧 Evolver 持有旧 backend 引用，其 finally 块的 shutdown 仅影响旧 backend）
        task_info.infra.backend = InfraFactory.create_backend(
            task_info.config.infrastructure.backend
        )

        # Resolve evaluator_path for resume Evolver
        evaluator_path = task_info.config.experiment.evaluator
        if not evaluator_path:
            evaluator_path = os.path.join(
                os.path.dirname(os.path.abspath(task_data.get("config_path", ""))),
                "evaluator.py",
            )

        # Re-resolve strategy with persisted decision (aligned with main.py resume)
        persisted_route = experiment.strategy_router_decision
        strategy, route_decision = resolve_strategy(
            task_info.config,
            task_info.infra,
            task_info.evaluate_fn,
            persisted_decision=persisted_route,
        )
        task_info.strategy = strategy

        task_info.evolver = Evolver(
            config=task_info.config.experiment,
            population_module=strategy.population_module,
            experiment=experiment,
            logger=task_info.infra.logger,
            llm_client=task_info.infra.llm,
            env=task_info.infra.env,
            data_service=task_info.infra.storage,
            embedding_client=task_info.infra.embedding,
            monitor=task_info.infra.monitor,
            langfuse=task_info.infra.langfuse,
            backend=task_info.infra.backend,
            evaluator_path=evaluator_path,
        )

        trace_filename = persist_strategy_route_trace(
            task_info.infra.storage,
            task_info.evolver.experiment.id,
            route_decision,
            logger=task_info.infra.logger,
        )
        persist_strategy_route_decision(
            task_info.evolver.experiment,
            route_decision,
            logger=task_info.infra.logger,
            trace_filename=trace_filename,
        )
        print(f"Resuming evolve for worker {task_info.worker_id} "
              f"from iteration {experiment.current_iteration}")

    if not (hasattr(task_info, 'evolver') and task_info.evolver):
        task_info.status = Status.FAILED
        print(f"Evolve task failed: no evolver found for worker {task_info.worker_id}")
        return

    # 若指定了 iterations，在当前进度基础上叠加（仿照 fm_api.py run_evolve(iterations=N)）
    iterations = task_data.get("iterations")
    if iterations:
        current = task_info.evolver.experiment.current_iteration
        task_info.evolver.config.max_iterations = current + iterations
        # 持久化更新后的 max_iterations 到 config.yaml
        task_info.infra.storage.save_config(task_info.experiment_id, task_info.config)
        print(f"Worker {task_info.worker_id}: continue evolve for {iterations} iterations "
              f"(current={current}, new max={current + iterations})")

    # 指定 config_path 时直接运行全流程（enrichment + evolution），否则仅运行 evolution
    run_mode = None if config_path else "evolution"
    try:
        final_experiment = await asyncio.to_thread(task_info.evolver.run, task_info.strategy, run_mode)
    except asyncio.CancelledError:
        # 若外部已设置为 PAUSED/STOPPED，不覆盖；否则标记为 CANCELLED
        if task_info.status not in {Status.PAUSED, Status.STOPPED}:
            task_info.status = Status.CANCELLED
        # pause 时保存 checkpoint（仿照 main.py KeyboardInterrupt 处理）
        # 注意：此处为同步调用，会短暂阻塞事件循环；
        # CancelledError handler 内使用 await to_thread 有被二次取消的风险，
        # save_experiment 通常是快速文件写入，可接受
        if task_info.status == Status.PAUSED and task_info.infra and task_info.evolver:
            try:
                task_info.infra.storage.save_experiment(task_info.evolver.experiment)
                print(f"Checkpoint saved for worker {task_info.worker_id}, "
                      f"iteration {task_info.evolver.experiment.current_iteration}")
            except Exception as e:
                print(f"Warning: failed to save checkpoint on pause: {e}")
        print(f"Evolve task cancelled for worker {task_info.worker_id}")
        raise
    task_info.status = Status.COMPLETED
    task_info.result = list(final_experiment.archive.values())


async def _run_report_task(task_data: dict, task_info: "TaskInfo"):
    """执行 report 任务（由 evolve 完成后自动入队，经 worker_loop 统一管理）"""
    worker_id = task_info.worker_id
    llm_config = None
    try:
        llm_config = task_info.config.infrastructure.llm
    except AttributeError:
        pass

    if not llm_config:
        task_info.status = Status.FAILED
        task_info.error_msg = "no llm_config available for report"
        print(f"Report task failed: no llm_config for worker {worker_id}")
        return

    task_info.stage = Stage.REPORT
    task_info.status = Status.RUNNING

    try:
        await build_report(
            _build_output_dir(task_info.experiment_root, worker_id, "system"),
            llm_config,
            _build_output_dir(task_info.experiment_root, worker_id, "user"),
        )
    except asyncio.CancelledError:
        task_info.status = Status.CANCELLED
        print(f"Report task cancelled for worker {worker_id}")
        raise
    except Exception as e:
        task_info.status = Status.FAILED
        task_info.error_msg = str(e)
        print(f"Report task failed for worker {worker_id}: {e}")
        return

    task_info.status = Status.COMPLETED
    print(f"Report task completed for worker {worker_id}")


async def _run_front_task(task_data: dict, task_info: "TaskInfo"):
    """执行 front agent 任务"""
    # Front agent 仅运行 evolution 循环（跳过 init/enrichment）
    if hasattr(task_info, 'evolver') and task_info.evolver:
        try:
            final_experiment = await asyncio.to_thread(task_info.evolver.run, task_info.strategy, "evolution")
        except asyncio.CancelledError:
            task_info.status = Status.CANCELLED
            print(f"Front task cancelled for worker {task_info.worker_id}")
            raise
        task_info.status = Status.COMPLETED
        task_info.result = final_experiment
    else:
        task_info.status = Status.FAILED
        print(f"Front task failed: no evolver found for worker {task_info.worker_id}")


async def worker_loop():
    """单 Worker 协程，从队列中获取任务并处理"""
    global is_running
    print("Worker started, waiting for tasks...")
    current_task: asyncio.Task = None

    while is_running:
        try:
            task_data = await asyncio.wait_for(task_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        print(f"Worker received task: {task_data}")
        current_task = asyncio.create_task(process_task(task_data))
        # 将 task 引用存入 task_info，供 pause/stop 接口取消
        _task_info = TASKS.get(task_data.get("worker_id"))
        if _task_info:
            _task_info.task = current_task

        # 同时等待任务完成或者 shutdown 事件触发
        shutdown_waiter = asyncio.create_task(shutdown_event.wait())
        done, _ = await asyncio.wait(
            {current_task, shutdown_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if shutdown_waiter in done:
            # 收到终止信号，取消当前任务后退出
            if not current_task.done():
                print("Shutdown: cancelling current running task...")
                current_task.cancel()
                try:
                    await current_task
                except (asyncio.CancelledError, Exception):
                    pass
            shutdown_waiter.cancel()
            task_queue.task_done()
            break

        # 任务正常结束
        shutdown_waiter.cancel()
        task_queue.task_done()
        current_task = None

    print("Worker stopped")


def get_task_info(worker_id: str) -> TaskInfo:
    """获取任务信息"""
    global TASKS
    task_info = TASKS.get(worker_id)
    if task_info is None or not task_info.worker_id:
        raise ValueError(f"Task not initialized for worker_id {worker_id}")
    return task_info


# ==================== API 端点 ====================
@app.get("/")
async def root():
    """root"""
    return {"message": "欢迎使用 Famou 任务 API", "version": "v2"}


@app.post("/init")
async def init(request: InitRequest):
    """init - 初始化任务"""
    global TASKS, LOGGER_FLAG

    job_id = request.job_id
    worker_id = request.worker_id
    config_path = request.config_path

    if worker_id in TASKS and TASKS[worker_id].stage == Stage.INIT:
        return {
            "code": "0",
            "msg": "worker_id already exists and in init stage",
            "output_dir": _build_output_dir(TASKS[worker_id].experiment_root, worker_id)
        }
    if worker_id in TASKS:
        return {"code": "0", "msg": "worker_id already exists", "output_dir": f""}

    try:
        task_info = TaskInfo()
        task_info.job_id = job_id
        task_info.worker_id = worker_id
        task_info.config_path = config_path
        task_info.stage = Stage.INIT
        task_info.status = Status.RUNNING

        if not LOGGER_FLAG:
            set_logger(task_info.experiment_root)
            LOGGER_FLAG = True

        TASKS[worker_id] = task_info

        # 将任务放入队列
        await task_queue.put({
            "type": "init",
            "worker_id": worker_id,
            "job_id": job_id,
            "config_path": config_path
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"code": "-2", "msg": f"{e}", "output_dir": f""}

    return {
        "code": "0",
        "msg": "success",
        "output_dir": _build_output_dir(task_info.experiment_root, worker_id),
    }


@app.post("/cancel")
async def cancel(request: BaseRequest):
    """cancel - 取消任务"""
    worker_id = request.worker_id
    task_info = get_task_info(worker_id)

    if task_info.is_canceling:
        return {"code": "0", "msg": "task is canceling"}
    if task_info.status == "cancelled":
        return {"code": "0", "msg": "task has already cancelled"}

    task_info.is_canceling = True
    task_info.status = Status.CANCELLED

    # 通知 evolver 在迭代边界停止（线程安全）
    if task_info.evolver is not None:
        task_info.evolver.stop_evolve()
    # 取消 asyncio Task，触发 CancelledError 
    if task_info.task and not task_info.task.done():
        task_info.task.cancel()

    return {"code": "0", "msg": "success"}


@app.post("/front_agent/start")
async def start_front_agent(request: BaseRequest):
    """start frontAgent"""
    worker_id = request.worker_id
    task_info = get_task_info(worker_id)

    if task_info.stage != Stage.INIT:
        return {"code": "-1", "msg": f"current stage is not init, but {task_info.stage}"}
    if task_info.status != Status.COMPLETED:
        return {"code": "-1", "msg": "init not completed yet"}
    if task_info.front_task:
        return {"code": "-1", "msg": "frontAgent is already running"}

    # 放入队列
    await task_queue.put({
        "type": "front",
        "worker_id": worker_id,
    })

    task_info.front_task = asyncio.create_task(asyncio.sleep(0))  # Mock
    return {"code": "0", "msg": "success"}


@app.post("/front_agent/check")
async def check_front_agent(request: BaseRequest):
    """Check front agent status and return result"""
    worker_id = request.worker_id
    task_info = get_task_info(worker_id)

    if task_info.stage != Stage.INIT:
        return {
            "code": "-1",
            "msg": f"Current stage is not init, but {task_info.stage}"
        }

    if task_info.front_task is None:
        return {"code": "-1", "msg": "FrontAgent not started yet"}

    if task_info.front_task.done():
        error = task_info.front_task.exception()
        if error is not None:
            return {
                "code": "0",
                "msg": "success",
                "status": "failed",
                "result": str(error)
            }

        result = task_info.front_task.result()
        return {
            "code": "0",
            "msg": "success",
            "status": "completed",
            "result": result
        }

    return {
        "code": "0",
        "msg": "success",
        "status": "running",
        "result": ""
    }


# ==================== Coldstart 接口 ====================
@app.post("/coldstart/start")
async def start_coldstart(request: BaseRequest):
    """start coldstart"""
    job_id = request.job_id
    worker_id = request.worker_id
    task_info = get_task_info(worker_id)

    output_dir = _build_output_dir(task_info.experiment_root, worker_id, "coldstart")

    if task_info.stage == Stage.COLDSTART:
        return {"code": "0", "msg": "already in coldstart stage", "output_dir": output_dir}
    if task_info.stage != Stage.INIT:
        return {"code": "-1", "msg": f"current stage is not init, but {task_info.stage}"}
    if task_info.status != Status.COMPLETED:
        return {"code": "-1", "msg": "init not completed yet"}

    task_info.stage = Stage.COLDSTART
    task_info.status = Status.RUNNING

    # 放入队列
    await task_queue.put({
        "type": "coldstart",
        "worker_id": worker_id,
        "job_id": job_id,
    })

    return {
        "code": "0",
        "msg": "success",
        "output_dir": output_dir,
    }


@app.post("/coldstart/stop")
async def stop_coldstart(request: BaseRequest):
    """stop coldstart"""
    worker_id = request.worker_id
    task_info = get_task_info(worker_id)

    if task_info.stage != Stage.COLDSTART:
        return {"code": "-1", "msg": f"current stage is not coldstart, but {task_info.stage}"}
    if task_info.task is None:
        return {"code": "-1", "msg": "coldstart not running"}
    if task_info.task.done():
        return {"code": "-1", "msg": "coldstart not running"}

    task_info.status = Status.STOPPED
    return {"code": "0", "msg": "success"}


@app.post("/coldstart/pause")
async def pause_coldstart(request: BaseRequest):
    """pause coldstart"""
    worker_id = request.worker_id
    task_info = get_task_info(worker_id)

    if task_info.stage != Stage.COLDSTART:
        return {"code": "-1", "msg": f"current stage is not coldstart, but {task_info.stage}"}
    if task_info.task is None:
        return {"code": "-1", "msg": "coldstart not running"}
    if task_info.task.done():
        return {"code": "-1", "msg": "coldstart not running"}

    task_info.status = Status.PAUSED
    return {"code": "0", "msg": "success"}


@app.post("/coldstart/continue")
async def resume_coldstart(request: BaseRequest):
    """resume coldstart"""
    worker_id = request.worker_id
    task_info = get_task_info(worker_id)

    if task_info.stage != Stage.COLDSTART:
        return {"code": "-1", "msg": f"current stage is not coldstart, but {task_info.stage}"}
    if task_info.task is None:
        return {"code": "-1", "msg": "Coldstart not started"}

    task_info.status = Status.RUNNING
    return {"code": "0", "msg": "success"}


@app.post("/coldstart/restart")
async def restart_coldstart(request: BaseRequest):
    """restart coldstart"""
    job_id = request.job_id
    worker_id = request.worker_id
    task_info = get_task_info(worker_id)

    if task_info.stage != Stage.COLDSTART:
        return {"code": "-1", "msg": f"current stage is not coldstart, but {task_info.stage}"}
    if task_info.status != Status.COMPLETED:
        return {"code": "-1", "msg": "coldstart not completed yet"}

    output_dir = _build_output_dir(task_info.experiment_root, worker_id, "coldstart")
    task_info.status = Status.RUNNING
    task_info.stage = Stage.COLDSTART

    # 放入队列
    await task_queue.put({
        "type": "coldstart",
        "worker_id": worker_id,
        "job_id": job_id,
    })

    return {
        "code": "0",
        "msg": "success",
        "output_dir": output_dir,
    }


# ==================== Evolve 接口 ====================
@app.post("/evolve/start")
async def start_evolve(request: EvolveRequest):
    """start evolve"""
    global LOGGER_FLAG
    job_id = request.job_id
    worker_id = request.worker_id
    iterations = request.iterations
    config_path = request.config_path

    # 若提供 config_path 且 task 不存在，仿照 /init 创建 task_info
    if config_path and worker_id not in TASKS:
        try:
            task_info = TaskInfo()
            task_info.job_id = job_id
            task_info.worker_id = worker_id
            task_info.config_path = config_path
            TASKS[worker_id] = task_info

            if not LOGGER_FLAG:
                set_logger(task_info.experiment_root)
                LOGGER_FLAG = True
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return {"code": "-2", "msg": f"{e}", "output_dir": ""}
    else:
        task_info = get_task_info(worker_id)

    output_dir = _build_output_dir(task_info.experiment_root, worker_id)

    if config_path:
        # 提供 config_path 时跳过阶段限制，直接进入 evolve
        if task_info.task and not task_info.task.done():
            return {"code": "-1", "msg": f"stage {task_info.stage} still running"}
    else:
        if task_info.stage not in {Stage.COLDSTART, Stage.INIT, Stage.REPORT}:
            return {"code": "-1", "msg": f"current stage is not coldstart, init, or report, but {task_info.stage}"}

        # 前一个任务还在运行中，拒绝重复入队（参考 fm_api.py 的防重检查）
        if task_info.task and not task_info.task.done():
            return {"code": "-1", "msg": f"stage {task_info.stage} still running"}

        # 从 coldstart/init 阶段开始时不限制轮数（全量运行）
        if task_info.stage in {Stage.COLDSTART, Stage.INIT}:
            iterations = None

        # 从 report 阶段继续演化时，必须指定 iterations（参考 fm_api.py）
        if task_info.stage == Stage.REPORT and not iterations:
            return {"code": "-1", "msg": "iterations must be provided for continual running evolve"}

    task_info.stage = Stage.EVOLVE
    task_info.status = Status.RUNNING
    task_info.error_msg = ""  # 清除上一次的残留错误信息

    await task_queue.put({
        "type": "evolve",
        "worker_id": worker_id,
        "job_id": job_id,
        "iterations": iterations,
        "config_path": config_path,
    })

    return {
        "code": "0",
        "msg": "success",
        "output_dir": output_dir,
    }


@app.post("/evolve/stop")
async def stop_evolve(request: BaseRequest):
    """stop evolve"""
    worker_id = request.worker_id
    task_info = get_task_info(worker_id)

    if task_info.stage != Stage.EVOLVE:
        return {"code": "-1", "msg": f"current state is not evolve, but {task_info.stage}"}
    if task_info.status != Status.RUNNING:
        return {"code": "-1", "msg": f"evolve is not running, status: {task_info.status}"}

    task_info.status = Status.STOPPED
    # 先通知 evolver 在下一个迭代边界停止（线程安全），再取消 asyncio.Task
    if task_info.evolver is not None:
        task_info.evolver.stop_evolve()
    if task_info.task and not task_info.task.done():
        task_info.task.cancel()

    return {"code": "0", "msg": "success"}


@app.post("/evolve/pause")
async def pause_evolve(request: BaseRequest):
    """pause evolve - 设置 PAUSED 状态并取消任务，checkpoint 由 _run_evolve_task 的 CancelledError 处理保存"""
    worker_id = request.worker_id
    task_info = get_task_info(worker_id)

    if task_info.stage != Stage.EVOLVE:
        return {"code": "-1", "msg": f"current state is not evolve, but {task_info.stage}"}
    if task_info.status != Status.RUNNING:
        return {"code": "-1", "msg": f"evolve is not running, status: {task_info.status}"}

    # 先设置状态，再 cancel，避免被 CancelledError 处理覆盖
    task_info.status = Status.PAUSED
    # 通知 evolver 在下一个迭代边界停止（线程安全）
    if task_info.evolver is not None:
        task_info.evolver.stop_evolve()
    if task_info.task and not task_info.task.done():
        task_info.task.cancel()

    return {"code": "0", "msg": "success"}


@app.post("/evolve/continue")
async def resume_evolve(request: BaseRequest):
    """resume evolve - 入队带 resume 标记的任务，由 _run_evolve_task 完成 checkpoint 加载和 Evolver 重建"""
    job_id = request.job_id
    worker_id = request.worker_id
    task_info = get_task_info(worker_id)

    if task_info.stage != Stage.EVOLVE:
        return {"code": "-1", "msg": f"current state is not evolve, but {task_info.stage}"}
    if task_info.status != Status.PAUSED:
        return {"code": "-1", "msg": f"evolve is not paused, status: {task_info.status}"}

    task_info.status = Status.RUNNING
    await task_queue.put({
        "type": "evolve",
        "worker_id": worker_id,
        "job_id": job_id,
        "iterations": None,
        "resume": True,
    })

    return {"code": "0", "msg": "success"}


@app.post("/update_config")
async def update_config(request: ConfigUpdateRequest):
    """update_config - 用 check_id 的配置信息更新 worker_id 的运行任务"""
    worker_id = request.worker_id
    check_id = request.check_id

    task_info = get_task_info(worker_id)

    # 从 TASKS 获取 check_id 的配置信息（预先通过 init 接口初始化）
    check_task_info = TASKS.get(check_id)
    if check_task_info is None or not check_task_info.worker_id:
        return {"code": "-1", "msg": f"check_id {check_id} not initialized"}
    if not check_task_info.config:
        return {"code": "-1", "msg": f"No config stored for check_id {check_id}"}

    if task_info.stage != Stage.EVOLVE:
        return {"code": "-1", "msg": f"current stage is not evolve, but {task_info.stage}"}
    if task_info.status != Status.RUNNING:
        return {"code": "-1", "msg": f"evolve is not running, status: {task_info.status}"}

    # 1. 暂停当前任务
    task_info.status = Status.PAUSED
    if task_info.evolver is not None:
        task_info.evolver.stop_evolve()
    if task_info.task and not task_info.task.done():
        task_info.task.cancel()
        try:
            await task_info.task
        except (asyncio.CancelledError, Exception):
            pass

    # 等待 checkpoint 保存（_run_evolve_task 的 CancelledError 处理会保存）
    await asyncio.sleep(0.1)

    # 2. 直接使用 TASKS 中 check_id 的配置信息（init 时已加载并解析路径，无需重新读取文件）
    new_config = check_task_info.config.model_copy(deep=True)

    # 逐字段合并覆盖（排除 run_user_id，保留当前任务的 user）
    src_experiment = new_config.experiment
    dst_experiment = task_info.config.experiment

    for field_name, field_value in src_experiment.model_dump(exclude_unset=True).items():
        if field_value is not None and field_value != "" and field_name != "run_user_id":
            # model_dump() 会将嵌套 BaseModel 转为 dict，需还原为模型对象
            if isinstance(field_value, dict):
                original_value = getattr(src_experiment, field_name, None)
                if isinstance(original_value, BaseModel):
                    field_value = original_value
            setattr(dst_experiment, field_name, field_value)

    # 持久化更新后的 config 到实验产出目录，确保报告等下游读取到最新参数
    task_info.infra.storage.save_config(task_info.experiment_id, task_info.config)

    # 3. 继续运行任务（入队 resume）
    task_info.status = Status.RUNNING
    await task_queue.put({
        "type": "evolve",
        "worker_id": worker_id,
        "job_id": task_info.job_id,
        "iterations": None,
        "resume": True,
    })

    return {"code": "0", "msg": "success"}


@app.post("/check")
async def check(request: BaseRequest):
    """check - 检查任务状态"""
    job_id = request.job_id
    worker_id = request.worker_id

    task_info = get_task_info(worker_id)

    # 从 evolver 获取实际进度
    max_iterations = 0
    cur_iterations = 0

    # init 阶段不涉及 evolution，max_iterations 和 cur_iterations 都为 0
    if task_info.stage != Stage.INIT:
        if hasattr(task_info, 'evolver') and task_info.evolver:
            evolver = task_info.evolver
            max_iterations = evolver.config.max_iterations
            cur_iterations = evolver.experiment.current_iteration

    if task_info.is_canceling:
        task_info.status = Status.CANCELLED
        task_info.is_canceling = False

    # 计算多样性（archive 大小不变时复用缓存，避免每次 O(n²) 运算）
    diversity = task_info._diversity_cache
    if hasattr(task_info, 'evolver') and task_info.evolver:
        try:
            all_programs = list(task_info.evolver.experiment.archive.values())
            current_size = len(all_programs)
            if current_size != task_info._diversity_archive_size and current_size >= 2:
                vectors = [p.feature_vector for p in all_programs if p.feature_vector]
                if len(vectors) >= 2:
                    from famou.utils.math import cosine_distance
                    total_dist = 0.0
                    count = 0
                    for i in range(len(vectors)):
                        for j in range(i + 1, len(vectors)):
                            total_dist += cosine_distance(vectors[i], vectors[j])
                            count += 1
                    diversity = total_dist / count if count > 0 else 0.0
                    task_info._diversity_cache = diversity
                    task_info._diversity_archive_size = current_size
        except Exception:
            pass

    # 取 error_msg 放在所有状态检查之后，确保 report 失败等场景的 error_msg 能被捕获
    error_message = task_info.error_msg if hasattr(task_info, 'error_msg') else ""

    output = {
        "code": "0",
        "msg": "success",
        "stage": task_info.stage,
        "max_iterations": max_iterations,
        "cur_iterations": cur_iterations,
        "progress": cur_iterations / max_iterations if max_iterations > 0 else 0.0,
        "status": task_info.status,
        "error_msg": error_message,
        "diversity": diversity,
    }

    return output


@app.post("/check_all")
async def check_all():
    """check all - 检查所有任务状态"""
    finished = []
    unfinished = []

    for worker_id, task_info in TASKS.items():
        if task_info.status in {Status.COMPLETED, Status.FAILED, Status.CANCELLED}:
            finished.append(worker_id)
        else:
            unfinished.append(worker_id)

    return {"code": "0", "msg": "success", "finished": finished, "unfinished": unfinished}


# ==================== 生命周期管理 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动时创建队列和启动 Worker"""
    global task_queue, worker_tasks, is_running, shutdown_event

    shutdown_event = asyncio.Event()
    task_queue = asyncio.Queue()
    is_running = True
    for i in range(MAX_CONCURRENT_TASKS):
        t = asyncio.create_task(worker_loop(), name=f"worker-{i}")
        worker_tasks.append(t)
    print(f"Application started, {MAX_CONCURRENT_TASKS} workers initialized")

    # 为 api_details_logger 配置默认 FileHandler，确保所有请求都能落盘
    default_log_dir = f"{WORKSPACE}/logs"
    os.makedirs(default_log_dir, exist_ok=True)
    default_log_path = os.path.join(default_log_dir, "api.log")
    _default_handler = logging.FileHandler(default_log_path, encoding="utf-8")
    _default_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    api_details_logger.addHandler(_default_handler)
    api_details_logger.info(f"API details logger initialized with default handler: {default_log_path}")

    # 用 asyncio 的信号接口注册，保证在事件循环线程内执行，线程安全
    loop = asyncio.get_running_loop()

    def _on_signal():
        global is_running
        print("\nReceived shutdown signal, initiating graceful shutdown...")
        is_running = False
        shutdown_event.set()
        for wt in worker_tasks:
            if not wt.done():
                wt.cancel()
        # 若 5 秒内仍未退出则强制退出（托管线程无法被 cancel）
        loop.call_later(5, os._exit, 0)

    try:
        loop.add_signal_handler(signal.SIGINT, _on_signal)
        loop.add_signal_handler(signal.SIGTERM, _on_signal)
    except (NotImplementedError, OSError):
        # Windows 不支持 add_signal_handler
        pass


@app.on_event("shutdown")
async def shutdown_handler():
    """应用关闭时停止 Worker"""
    global worker_tasks, is_running, shutdown_event
    is_running = False
    if shutdown_event:
        shutdown_event.set()
    for wt in worker_tasks:
        if not wt.done():
            wt.cancel()
            try:
                await wt
            except (asyncio.CancelledError, Exception):
                pass
    print("Application shutdown, all workers stopped")


if __name__ == "__main__":
    print("服务即将启动于 http://127.0.0.1:8090")
    uvicorn.run(app, host="0.0.0.0", port=8090, log_config={
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
            },
        },
        "handlers": {
            "default": {"class": "logging.StreamHandler", "formatter": "default", "stream": "ext://sys.stderr"},
        },
        "loggers": {
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": [], "level": "INFO", "propagate": False},
        },
    })
