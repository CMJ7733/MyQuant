#!/usr/bin/env python3
"""
evaluator_mlebench.py ─ 支持自动运行 user_code.py 生成 submission.csv 并评测
────────────────────────────────────
"""
from ctypes import PyDLL
import logging
import re
import os
import time
import subprocess
import sys
import time
import json
import traceback
import numpy as np
import threading
import pynvml
import shutil
import glob
import pandas as pd
import ast
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger("famou")


# ============================================================================
# 配置类和常量定义
# ============================================================================

class ReturnCode(IntEnum):
    """进程返回码枚举"""
    SUCCESS = 0
    GENERAL_ERROR = -1
    TIMEOUT = -9
    INVALID_SUBMISSION = -10


# ============================================================================
# 配置类
# ============================================================================

class EvaluationThresholds(object):
    """评估阈值配置"""
    INVALID_SUBMISSION_VALIDITY: float = 0.3
    GOLD_MEDAL_SCORE: float = 3.0
    SILVER_MEDAL_SCORE: float = 2.0
    BRONZE_MEDAL_SCORE: float = 1.0
    FILE_CHECK_INTERVAL: float = 2.0
    GPU_MONITOR_INTERVAL: float = 0.5
    GPU_MONITOR_PRINT_INTERVAL: int = 20  # 分钟

    # 早期终止配置
    MAX_INVALID_FILES: int = 3  # 最多容忍的连续无效文件数


@dataclass
class PathConfig:
    """路径配置"""
    forced_submission_dir: str = "/mnt/cfs_bj_mt/workspace/gezengle/fm/agent/workspace/baidu/acgbenchmark/alpha_evolve"
    data_dir: str = "/mnt/cfs_bj_mt/workspace/caolizhe/public/mlebench-competitions"
    # 备用数据目录（typo修复）
    data_dir_backup: str = "/mnt/cfs_bj_mt/workspace/caolizhe/pubic/mlebench-competitions"


@dataclass
class ParallelConfig:
    """并行执行配置"""
    # 线程池大小，None 表示自动计算
    max_workers: Optional[int] = None

    # 单个 grader 超时时间（秒）
    grader_timeout: int = 8 * 60 * 60  # 8小时

    # 是否启用并行（可关闭用于调试）
    enable_parallel: bool = True

    def __post_init__(self):
        """初始化后处理，自动计算线程池大小"""
        if self.max_workers is None:
            # I/O 密集型任务，可以使用较大的线程池
            cpu_count = os.cpu_count() or 1
            self.max_workers = min(32, cpu_count * 4)

            print(
                f"[ParallelConfig] 自动设置线程池大小: {self.max_workers} "
                f"(CPU核心数: {cpu_count})"
            )


@dataclass
class EvaluatorConfig:
    """评估器配置"""
    # 路径配置
    paths: PathConfig = field(default_factory=PathConfig)

    # 评估阈值
    thresholds: EvaluationThresholds = field(default_factory=EvaluationThresholds)

    # 并行执行配置
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    # 超时配置（秒）
    default_timeout: int = 6 * 60 * 60  # 6小时

    # 指标方向（根据比赛类型动态设置）
    lower_is_better: bool = True

    # GPU 配置
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    enable_gpu_monitoring: bool = True


# ============================================================================
# 辅助函数
# ============================================================================


def calculate_gpu_memory(peak_memory_mb: float, total_memory_mb: float, free_memory_mb: float) -> float:
    """
    计算实际使用的GPU显存

    Args:
        peak_memory_mb: 峰值显存占用
        total_memory_mb: GPU总显存
        free_memory_mb: 初始可用显存

    Returns:
        实际使用的显存（MB）
    """
    return peak_memory_mb - (total_memory_mb - free_memory_mb)


def calculate_gpu_utilization(used_memory_mb: float, free_memory_mb: float) -> float:
    """
    计算GPU显存利用率

    Args:
        used_memory_mb: 使用的显存
        free_memory_mb: 初始可用显存

    Returns:
        显存利用率（0-1之间）
    """
    if free_memory_mb <= 0:
        return 0.0
    return round(used_memory_mb / free_memory_mb, 4)


def build_evaluation_result(
    success: bool,
    error_info: dict,
    wall_time: float,
    submission_csv: Optional[List[str]] = None,
    peak_gpu_memory_mb: Optional[float] = None,
    gpu_utilization: Optional[float] = None,
    **kwargs
) -> dict:
    """
    构建统一的评估结果字典

    Args:
        success: 是否成功
        error_info: 错误信息字典
        wall_time: 执行时间
        submission_csv: 提交文件路径列表
        peak_gpu_memory_mb: 峰值GPU显存
        gpu_utilization: GPU利用率
        **kwargs: 其他字段

    Returns:
        标准化的结果字典
    """
    result = {
        "success": success,
        "error_info": error_info,
        "wall_time": wall_time,
        "submission_csv": submission_csv,
    }

    # 添加GPU信息（如果提供）
    if peak_gpu_memory_mb is not None:
        result["peak_gpu_memory_mb"] = peak_gpu_memory_mb
    if gpu_utilization is not None:
        result["gpu_utilization"] = gpu_utilization

    # 添加其他字段
    result.update(kwargs)

    return result


def get_error_info_from_exception(ex: Exception, traceback_str: Optional[str] = None) -> dict:
    """
    从异常对象构建错误信息字典

    Args:
        ex: 异常对象
        traceback_str: 追踪信息字符串

    Returns:
        标准化的错误信息字典
    """
    error_info = {
        "exception": str(ex),
        "exception_type": type(ex).__name__,
    }
    if traceback_str:
        error_info["traceback"] = traceback_str
    return error_info


# ============================================================================
# 资源管理类
# ============================================================================

class ResourceManager(object):
    """
    资源管理器 - 负责管理进程、线程、队列等资源的清理

    职责：
    - 管理子进程的生命周期
    - 管理输出读取线程
    - 管理 GPU 监控线程
    - 统一清理所有资源
    """

    def __init__(self, enable_gpu_monitoring: bool = True, gpu_ids: List[int] = None):
        """
        初始化资源管理器

        Args:
            enable_gpu_monitoring: 是否启用GPU监控
            gpu_ids: GPU编号列表，例如 [0] 或 [0, 1]
        """
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.gpu_ids = gpu_ids or [0]

        # 进程资源
        self.proc = None
        self.stdout_thread = None
        self.stderr_thread = None

        # GPU监控资源
        self.stop_monitor_event = None
        self.monitor_thread = None
        self.result_queue = None

        # GPU信息
        self.total_mb = None
        self.free_mb = None

        # 输出收集
        self.stdout_lines = []
        self.stderr_lines = []
        self.stdout_lines_lock = None
        self.stderr_lines_lock = None

    def initialize_gpu_monitoring(self):
        """初始化GPU监控"""
        if not self.enable_gpu_monitoring:
            return

        self.total_mb, self.free_mb = get_gpu_memory_info(self.gpu_ids)
        if self.total_mb and self.free_mb:
            print(
                f"GPU {self.gpu_ids} 总显存: {self.total_mb:.2f} MB\n"
                f"GPU {self.gpu_ids} 可用显存: {self.free_mb:.2f} MB"
            )

            from queue import Queue
            self.stop_monitor_event = threading.Event()
            self.result_queue = Queue()
            self.monitor_thread = threading.Thread(
                target=monitor_gpu_memory,
                args=(self.gpu_ids, self.stop_monitor_event, self.result_queue),
                daemon=True,
            )
            self.monitor_thread.start()

    def initialize_output_containers(self):
        """初始化输出收集容器"""
        self.stdout_lines = []
        self.stderr_lines = []
        self.stdout_lines_lock = threading.Lock()
        self.stderr_lines_lock = threading.Lock()

    def safe_terminate_process(self, process, process_name="用户脚本") -> bool:
        """
        安全终止进程，先尝试温和终止，再强制杀死

        Args:
            process: 要终止的进程对象
            process_name: 进程名称（用于日志）

        Returns:
            是否成功终止
        """
        if process is None or process.poll() is not None:
            return True  # 进程已结束

        logger.warning(f"正在终止{process_name}进程...")
        try:
            # 1. 首先尝试温和终止
            process.terminate()
            time.sleep(2)  # 给进程清理时间

            # 2. 如果还在运行，强制杀死
            if process.poll() is None:
                logger.warning(f"{process_name}进程未响应terminate，发送kill信号")
                process.kill()
                time.sleep(1)

            # 3. 等待进程结束
            process.wait(timeout=5.0)
            print(f"{process_name}进程已成功终止")
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"{process_name}进程终止超时")
            return False
        except Exception as e:
            logger.error(f"终止{process_name}进程时出错: {e}")
            return False

    def cleanup_all_resources(self):
        """清理所有资源：停止监控线程、输出线程等"""
        errors = []

        # 1. 停止GPU监控线程
        if self.enable_gpu_monitoring and self.total_mb and self.free_mb:
            if self.stop_monitor_event:
                self.stop_monitor_event.set()
            if self.monitor_thread:
                self.monitor_thread.join(timeout=3.0)
                if self.monitor_thread.is_alive():
                    logger.warning("GPU监控线程未正常退出")
                    errors.append("GPU监控线程未正常退出")

        # 2. 清空结果队列
        if self.enable_gpu_monitoring and self.total_mb and self.free_mb:
            if self.result_queue:
                try:
                    while not self.result_queue.empty():
                        self.result_queue.get_nowait()
                except Exception as e:
                    errors.append(f"清空队列失败: {e}")

        # 3. 等待输出线程结束
        thread_timeout = 2.0
        if self.stdout_thread and self.stdout_thread.is_alive():
            self.stdout_thread.join(timeout=thread_timeout)
            if self.stdout_thread.is_alive():
                logger.warning("stdout线程仍在运行")
                errors.append("stdout线程仍在运行")

        if self.stderr_thread and self.stderr_thread.is_alive():
            self.stderr_thread.join(timeout=thread_timeout)
            if self.stderr_thread.is_alive():
                logger.warning("stderr线程仍在运行")
                errors.append("stderr线程仍在运行")

        # 4. 确保进程已终止（最终保障）
        if self.proc and self.proc.poll() is None:
            if not self.safe_terminate_process(self.proc, "最终清理"):
                errors.append("进程终止失败")

        if errors:
            logger.warning(f"[ResourceManager] 清理时发生错误: {errors}")

    def get_peak_gpu_memory(self) -> float:
        """
        获取峰值GPU显存

        Returns:
            峰值显存（MB）
        """
        if not self.enable_gpu_monitoring or not self.total_mb or not self.free_mb:
            return 0.0

        peak_memory_mb = 0
        if self.result_queue:
            try:
                while not self.result_queue.empty():
                    peak_memory_mb = self.result_queue.get_nowait()
            except Exception:
                pass

        return peak_memory_mb


class SubmissionMonitor(object):
    """
    提交文件监控器 - 负责实时监控和验证submission文件

    职责：
    - 实时监控新提交文件的生成
    - 调用 grader 验证文件有效性
    - 缓存验证结果
    - 检测到无效提交时触发早期终止
    """

    def __init__(
        self,
        competition_name: str,
        submission_dir: str,
        invalidity_threshold: float = 0.3,
    ):
        """
        初始化提交文件监控器

        Args:
            competition_name: 比赛名称
            submission_dir: 提交文件目录
            invalidity_threshold: 无效提交阈值
        """
        self.competition_name = competition_name
        self.submission_dir = submission_dir
        self.invalidity_threshold = invalidity_threshold

        # 缓存已测试的文件
        self.tested_files: Dict[str, dict] = {}

        # 回调函数（发现无效提交时调用）
        self.on_invalid_submission = None

    def check_new_files(self, previous_files: set) -> List[str]:
        """
        检查是否有新文件生成

        Args:
            previous_files: 之前已知文件的集合

        Returns:
            新文件列表
        """
        import glob
        pattern = os.path.join(self.submission_dir, "submission*.csv")
        matching_files = glob.glob(pattern)
        new_files = [
            f for f in matching_files if f not in previous_files
        ]
        return new_files

    def validate_file(self, file_path: str) -> dict:
        """
        验证单个提交文件

        Args:
            file_path: 文件路径

        Returns:
            验证结果字典
        """
        print(f"[SubmissionMonitor] 开始验证: {os.path.basename(file_path)}")

        grader_res = run_grader(file_path, self.competition_name)
        self.tested_files[file_path] = grader_res
        validity = grader_res.get("validity", 0.0)
        score = grader_res.get("score", 0.0)
        medal = grader_res.get("medal", "none")

        print(
            f"[SubmissionMonitor] 验证完成: {os.path.basename(file_path)} "
            f"(validity={validity:.1f}, score={score:.4f}, medal={medal})"
        )

        return grader_res

    def should_terminate_early(self, grader_res: dict) -> bool:
        """
        判断是否应该早期终止（发现无效提交）

        Args:
            grader_res: Grader 结果

        Returns:
            是否应该早期终止
        """
        validity = grader_res.get("validity", 0.0)
        return validity < self.invalidity_threshold

    def get_tested_files(self) -> Dict[str, dict]:
        """获取所有已测试的文件结果"""
        return self.tested_files.copy()


# ============================================================================
# 全局常量（保持向后兼容）
# ============================================================================

# TODO: 逐步移除这些全局常量，改用 EvaluatorConfig
FORCED_SUBMISSION_DIR = PathConfig().forced_submission_dir


POSITIVE_COMPETETIONS = [
    '3d-object-detection-for-autonomous-vehicles', 'AI4Code', 'alaska2-image-steganalysis', 
    'aptos2019-blindness-detection', 
    'cassava-leaf-disease-classification', 'cdiscount-image-classification-challenge', 
    'chaii-hindi-and-tamil-question-answering', 
    'detecting-insults-in-social-commentary', 'facebook-recruiting-iii-keyword-extraction', 
    'freesound-audio-tagging-2019',
    'google-quest-challenge',
    'google-research-identify-contrails-reduce-global-warming',
    'h-and-m-personalized-fashion-recommendations',
    'herbarium-2020-fgvc7', 
    'herbarium-2021-fgvc8', 'herbarium-2022-fgvc9', 'histopathologic-cancer-detection', 
    'hotel-id-2021-fgvc8', 
    'hubmap-kidney-segmentation', 'imet-2020-fgvc7', 
    'invasive-species-monitoring', 'iwildcam-2019-fgvc6', 'iwildcam-2020-fgvc7', 
    'jigsaw-toxic-comment-classification-challenge', 
    'jigsaw-unintended-bias-in-toxicity-classification', 'kuzushiji-recognition', 
    'learning-agency-lab-automated-essay-scoring-2', 'ml2021spring-hw2', 
    'mlsp-2013-birds', 'movie-review-sentiment-analysis-kernels-only', 
    'nfl-player-contact-detection', 'osic-pulmonary-fibrosis-progression', 
    'paddy-disease-classification', 'plant-pathology-2020-fgvc7', 'plant-pathology-2021-fgvc8', 
    'plant-seedlings-classification', 'playground-series-s3e18', 
    'random-acts-of-pizza', 'ranzcr-clip-catheter-line-classification', 'rsna-breast-cancer-detection', 
    'rsna-miccai-brain-tumor-radiogenomic-classification', 
    'seti-breakthrough-listen', 'siim-covid19-detection', 'siim-isic-melanoma-classification', 
    'spaceship-titanic', 'tabular-playground-series-dec-2021', 
    'tabular-playground-series-may-2022', 'tensorflow-speech-recognition-challenge', 
    'tensorflow2-question-answering', 'text-normalization-challenge-english-language', 
    'text-normalization-challenge-russian-language', 'tgs-salt-identification-challenge', 
    'the-icml-2013-whale-challenge-right-whale-redux', 'tweet-sentiment-extraction', 
    'us-patent-phrase-to-phrase-matching', 'uw-madison-gi-tract-image-segmentation', 
    'vesuvius-challenge-ink-detection', 'vinbigdata-chest-xray-abnormalities-detection', 
    'whale-categorization-playground', 'aerial-cactus-identification', 
]


# 存储得到奖牌的 submission_csv
def move_and_rename_file(source_path, target_path, backup=True):
    """
    安全的文件移动和重命名，包含备份功能
    
    Args:
        source_path: 源文件路径
        target_path: 目标文件路径
        backup: 是否创建备份
    
    Returns:
        bool: 操作是否成功
    """
    try:
        source = Path(source_path)
        target = Path(target_path)
        
        # 验证源文件
        if not source.exists():
            print(f"错误: 源文件不存在 {source_path}")
            return False
        
        if not source.is_file():
            print(f"错误: 源路径不是文件 {source_path}")
            return False
        
        # 如果目标文件已存在，处理冲突
        if target.exists():
            if backup:
                # 创建备份
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = target.parent / f"{target.stem}_backup_{timestamp}{target.suffix}"
                shutil.move(str(target), str(backup_path))
                print(f"📦 已创建备份: {backup_path}")
            else:
                # 直接覆盖
                target.unlink()
        
        # 确保目标目录存在
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # 执行移动操作
        shutil.move(str(source), str(target))
        
        # 验证操作成功
        if target.exists() and not source.exists():
            print(f"✅ 成功: {source_path} -> {target_path}")
            return True
        else:
            print("❌ 操作可能未完全成功")
            return False
            
    except Exception as e:
        print(f"❌ 操作失败: {e}")
        return False
def get_epoch_folders(base_path=".", epoch_number=1):
    """
    获取匹配 *_{epoch_number} 模式的文件夹
    
    Args:
        base_path: 搜索的根路径，默认为当前目录
        epoch_number:  epoch编号，默认为1
    
    Returns:
        list: 匹配的文件夹路径列表
    """
    pattern = f"*_{epoch_number}"
    matching_folders = []
    
    # 方法1: 使用glob (最简单)
    glob_pattern = os.path.join(base_path, pattern)
    matching_folders = [
        f for f in glob.glob(glob_pattern) 
        if os.path.isdir(f)
    ]
    
    return matching_folders

# ret = get_epoch_folders(
#     "/mnt/cfs_bj_mt/workspace/.../petfinder-pawpularity-score/model_save",
#     epoch_number=1
# )
# print(ret)


# 存储模型
def save_model(
    original_model_dir: str,
    competition_name: str,
    save_epoch_idx: int,
    score: float = 0.0,
):
    """
    将模型整个文件夹移动到 model_saved 目录下。
    若 model_saved 中已存在同名文件夹，则添加时间戳避免覆盖。
    """

    # 目标保存路径：forced_submission/{competition}/model_saved
    model_save_dir = os.path.join(FORCED_SUBMISSION_DIR, competition_name, "model_saved")
    print(f"model_save_dir: {model_save_dir}")
    os.makedirs(model_save_dir, exist_ok=True)

    # 若指定 epoch，则切换到对应 epoch 的文件夹
    if save_epoch_idx != -1:
        target_dirs = get_epoch_folders(original_model_dir, save_epoch_idx)
        if target_dirs:
            original_model_dir = target_dirs[0]

    # 获取模型文件夹名字，例如 "/path/to/exp123" → "exp123"
    model_folder_name = os.path.basename(os.path.normpath(original_model_dir))
    final_save_path = os.path.join(model_save_dir, model_folder_name)

    # 若目标路径已存在，则加时间戳避免重名
    if os.path.exists(final_save_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_save_path = f"{final_save_path}_{timestamp}"
        print(f"目标模型名重复，使用带时间戳的新名称：{final_save_path}")

    # 执行目录整体移动
    try:
        shutil.move(original_model_dir, final_save_path)
        print(f"模型已成功移动到：{final_save_path}")
    except Exception as e:
        logger.error(f"移动模型失败: {e}", exc_info=True)
        raise

    return final_save_path
 

# 从文件名中提取submission的idx编号
def extract_submission_idx(filename) -> int:
    """
    从文件名中提取submission的idx编号
    
    Args:
        filename (str): 文件名，可以是完整路径或单纯文件名
        
    Returns:
        int: 提取到的idx编号，如果没有找到则返回-1
    """
    # 获取纯文件名（去掉路径）
    basename = os.path.basename(filename)
    
    # 定义正则表达式模式，匹配 submission_{数字}.csv 或 submission.csv
    pattern = r'^submission(?:_(\d+))?\.csv$'
    
    # 进行匹配
    match = re.match(pattern, basename)
    
    if match:
        idx_str = match.group(1)
        if idx_str is not None:
            try:
                return int(idx_str)
            except ValueError:
                return -1
        else:
            return -1  # submission.csv 的情况
    else:
        return -1  # 不匹配任何模式


def robust_loads(json_str):
    """
    尝试用 json.loads 解析，如果失败则用 ast.literal_eval 兜底
    """
    try:
        return json.loads(json_str)
    except Exception:
        try:
            return ast.literal_eval(json_str)
        except Exception:
            raise  # 最后还是抛出原始异常


def get_gpu_memory_info(gpu_ids: List[int] = None):
    """
    获取指定GPU列表的总显存和可用显存之和（单位MB）
    """
    if gpu_ids is None:
        gpu_ids = [0]
    try:
        pynvml.nvmlInit()
        total_memory_mb = 0
        free_memory_mb = 0
        for gpu_id in gpu_ids:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory_mb += mem_info.total / (1024 ** 2)
            free_memory_mb += mem_info.free / (1024 ** 2)
        return total_memory_mb, free_memory_mb
    except pynvml.NVMLError as error:
        print(f"无法获取GPU {gpu_ids} 的信息: {error}")
        return None, None
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def monitor_gpu_memory(gpu_ids: List[int], stop_event: threading.Event, q, interval: float = 0.5):
    """
    实时监控指定GPU列表的峰值显存占用（单位: MB），并不断写入队列
    新增功能：从线程启动开始计时，每20min打印一次当前执行耗时
    """
    if not gpu_ids:
        gpu_ids = [0]
    peak_used_memory = 0
    # ------------------- 新增：初始化计时相关变量 -------------------
    start_time = time.time()  # 记录线程启动的初始时间（秒级时间戳）
    interval_minutes = 20  # 定时打印间隔（20分钟）
    interval_seconds = interval_minutes * 60  # 转换为秒（1200秒）
    last_print_time = start_time  # 记录上一次打印的时间戳（初始为启动时间）
    # -------------------------------------------------------------

    try:
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(gpu_id) for gpu_id in gpu_ids]

        while not stop_event.is_set():
            try:
                # ------------------- 新增：定期打印耗时 -------------------
                current_time = time.time()
                # 计算当前耗时（秒转分钟，保留1位小数）
                elapsed_minutes = round((current_time - start_time) / 60, 1)
                # 判断是否达到20分钟间隔（当前时间 - 上一次打印时间 ≥ 1200秒）
                if current_time - last_print_time >= interval_seconds:
                    print(
                        f"GPU {gpu_ids} 监控线程已运行 {elapsed_minutes} 分钟，"
                        f"当前峰值显存：{peak_used_memory:.2f} MB"
                    )
                    last_print_time = current_time  # 更新上一次打印时间，避免重复打印
                # ---------------------------------------------------------

                # 累加所有GPU的显存占用
                current_used_mb = 0
                for handle in handles:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    current_used_mb += mem_info.used / (1024 ** 2)
                if current_used_mb > peak_used_memory:
                    peak_used_memory = current_used_mb
                    try:
                        q.put_nowait(peak_used_memory)
                    except:
                        pass
            except pynvml.NVMLError:
                break
            stop_event.wait(interval)  # 按原间隔休眠，不影响定时检查

    except Exception as e:
        print(f"GPU {gpu_ids} 监控线程出错: {e}")
    finally:
        # ------------------- 新增：线程退出时打印总耗时 -------------------
        total_elapsed_minutes = round((time.time() - start_time) / 60, 1)
        print(
            f"GPU {gpu_ids} 监控线程退出，总运行时间：{total_elapsed_minutes} 分钟，"
            f"最终峰值显存：{peak_used_memory:.2f} MB"
        )
        # -------------------------------------------------------------
        try:
            pynvml.nvmlShutdown()
        except:
            pass
        # 最后再写一次峰值
        try:
            q.put_nowait(peak_used_memory)
        except:
            pass


def submission_csv_analyze(submission_csv_path: str) -> str:
    """
    分析 submission.csv 文件，返回结果字典
    """
    pass
    

def run_python_script(
    py_path: str, 
    competition_name: str,
    gpu_ids: List[int] = None, 
    conda_env_name: str = None,
    timeout: float = 4 * 60 * 60, 
):
    """
    运行用户提交的python脚本，监控GPU显存，实时检查提交文件有效性。
    若检测到无效提交文件，立即安全终止进程并清理所有资源。
    """
    import os
    import glob
    import time
    import threading
    import subprocess
    import traceback
    from queue import Queue

    gpu_ids = gpu_ids or [0]
    cuda_visible_devices = ",".join(str(g) for g in gpu_ids)
    work_dir = os.path.dirname(py_path)
    py_basename = os.path.basename(py_path)
    print(f"工作目录: {work_dir}, 脚本文件: {py_basename}")

    # 查看显卡总显存以及当前可用显存
    total_mb, free_mb = get_gpu_memory_info(gpu_ids)
    if total_mb and free_mb:
        print(f"GPU {gpu_ids} 总显存: {total_mb:.2f} MB")
        print(f"GPU {gpu_ids} 可用显存: {free_mb:.2f} MB")

    # 初始化GPU监控
    if total_mb and free_mb:
        stop_monitor_event = threading.Event()
        result_queue = Queue()
        monitor_thread = threading.Thread(
            target=monitor_gpu_memory,
            args=(gpu_ids, stop_monitor_event, result_queue),
            daemon=True,
        )
        monitor_thread.start()

    t0 = time.time()
    proc = None
    stdout_thread = None
    stderr_thread = None
    
    # 用于收集输出的线程安全容器
    stdout_lines = []
    stderr_lines = []
    stdout_lines_lock = threading.Lock()
    stderr_lines_lock = threading.Lock()
    
    # 安全终止进程的辅助函数
    def safe_terminate_process(process, process_name="用户脚本"):
        """安全终止进程，先尝试温和终止，再强制杀死"""
        if process is None or process.poll() is not None:
            return True  # 进程已结束
        logger.warning(f"正在终止{process_name}进程...")
        try:
            # 1. 首先尝试温和终止
            process.terminate()
            time.sleep(2)  # 给进程清理时间
            
            # 2. 如果还在运行，强制杀死
            if process.poll() is None:
                logger.warning(f"{process_name}进程未响应terminate，发送kill信号")
                process.kill()
                time.sleep(1)
            
            # 3. 等待进程结束
            process.wait(timeout=5.0)
            print(f"{process_name}进程已成功终止")
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"{process_name}进程终止超时")
            return False
        except Exception as e:
            logger.error(f"终止{process_name}进程时出错: {e}")
            return False
    
    # 清理所有资源的辅助函数
    def cleanup_all_resources():
        """清理所有资源：停止监控线程、输出线程等"""
        # 1. 停止GPU监控线程
        if total_mb and free_mb:
            stop_monitor_event.set()
            monitor_thread.join(timeout=3.0)
            if monitor_thread.is_alive():
                logger.warning("GPU监控线程未正常退出")
        
        # 2. 清空结果队列
        if total_mb and free_mb:
            try:
                while not result_queue.empty():
                    result_queue.get_nowait()
            except:
                pass
        
        # 3. 等待输出线程结束
        thread_timeout = 2.0
        if stdout_thread and stdout_thread.is_alive():
            stdout_thread.join(timeout=thread_timeout)
            if stdout_thread.is_alive():
                logger.warning("stdout线程仍在运行")
        
        if stderr_thread and stderr_thread.is_alive():
            stderr_thread.join(timeout=thread_timeout)
            if stderr_thread.is_alive():
                logger.warning("stderr线程仍在运行")
        
        # 4. 确保进程已终止（最终保障）
        if proc and proc.poll() is None:
            safe_terminate_process(proc, "最终清理")
    
    # 已测试过的提交文件集合
    tested_submission_files = dict()
    try:
        # 初始化输出变量（防止超时时未定义）
        full_stdout = ""
        full_stderr = ""

        # 启动子进程
        if conda_env_name:
            cmd = f"conda run -n {conda_env_name} CUDA_VISIBLE_DEVICES={cuda_visible_devices} python3 {py_basename}"
        else:
            cmd = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} python3 {py_basename}"
        print(f"cmd:\n{cmd}")
        proc = subprocess.Popen(
            cmd,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            shell=True,
            text=True,
            bufsize=1,
        )
        
        # 流式读取输出的函数
        def stream_reader(stream, stream_type):
            """读取流并实时打印/保存"""
            lines = stdout_lines if stream_type == "stdout" else stderr_lines
            lock = stdout_lines_lock if stream_type == "stdout" else stderr_lines_lock
            
            try:
                for line in iter(stream.readline, ''):
                    if line:
                        line = line.rstrip('\n')
                        # 实时打印到日志
                        print(f"[用户脚本{stream_type.upper()}] {line}")
                        # 保存到容器
                        with lock:
                            lines.append(line)
            except Exception as e:
                logger.error(f"读取{stream_type}时出错: {e}")
            finally:
                try:
                    stream.close()
                except:
                    pass
        
        # 启动输出读取线程
        stdout_thread = threading.Thread(
            target=stream_reader,
            args=(proc.stdout, "stdout"),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=stream_reader, 
            args=(proc.stderr, "stderr"),
            daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()

        # 等待进程结束（带超时和实时文件检查）
        return_code = None
        start_time = time.time()

        # 检查是否生成 submission.csv
        submission_csv_path = os.path.join(FORCED_SUBMISSION_DIR, competition_name)
        os.makedirs(submission_csv_path, exist_ok=True)

        # 文件检查间隔（使用配置类中的值）
        check_interval = EvaluationThresholds.FILE_CHECK_INTERVAL
        last_check_time = time.time()

        # 早期终止配置：连续无效文件计数器
        consecutive_invalid_files = 0
        max_invalid_files = EvaluationThresholds.MAX_INVALID_FILES

        while True:
            return_code = proc.poll()
            if return_code is not None:
                break  # 进程已正常结束

            # 定期检查新生成的提交文件
            current_time = time.time()
            if current_time - last_check_time >= check_interval:
                last_check_time = current_time

                matching_files = glob.glob(os.path.join(submission_csv_path, "submission*.csv"))
                new_files = [
                    f for f in matching_files if f not in tested_submission_files
                ]

                for file_path in new_files:
                    print(f"检测到新提交文件: {file_path}")

                    # 验证文件有效性
                    grader_res = run_grader(file_path, competition_name)
                    tested_submission_files[file_path] = grader_res
                    validity = grader_res.get("validity", 0.0)

                    # 使用配置类中的阈值
                    if validity < EvaluationThresholds.INVALID_SUBMISSION_VALIDITY:
                        consecutive_invalid_files += 1
                        logger.error(
                            f"提交文件无效 (validity={validity:.2f}): {os.path.basename(file_path)} "
                            f"[连续无效: {consecutive_invalid_files}/{max_invalid_files}]"
                        )

                        # 检查是否达到早停阈值
                        if consecutive_invalid_files >= max_invalid_files:
                            logger.error(
                                f"连续 {consecutive_invalid_files} 个文件无效，达到早停阈值，"
                                f"终止进程以避免浪费时间"
                            )

                            # 1. 终止进程
                            safe_terminate_process(proc, "达到连续无效文件阈值")

                            # 2. 清理所有资源
                            cleanup_all_resources()

                            csv_read_error_info = ""
                            try:
                                submission_csv = pd.read_csv(file_path)
                            except Exception as e:
                                csv_read_error_info = str(f"read_csv error:{str(e)}")

                            # 3. 返回无效结果
                            result = {
                                **grader_res,
                                "success": False,
                                "wall_time": time.time() - t0,
                                "submission_csv": [file_path],
                                "return_code": ReturnCode.INVALID_SUBMISSION,
                            }
                            if not csv_read_error_info:
                                result["submission_csv.head()"] = str(submission_csv.head())
                            return result
                        # 否则，继续观察下一个文件
                    else:
                        # 文件有效，重置计数器
                        consecutive_invalid_files = 0
                        print(
                            f"提交文件有效 (validity={validity:.2f}): {os.path.basename(file_path)} "
                            f"[重置无效计数]"
                        )

            # 超时检查
            if time.time() - start_time > timeout:
                logger.error(f"脚本执行超时 ({timeout}秒)")
                raise subprocess.TimeoutExpired(cmd, timeout)

            # 短暂休眠避免忙等待
            time.sleep(0.1)
        
        # 等待输出线程完成
        t1 = time.time()
        if stdout_thread:
            stdout_thread.join(timeout=5.0)
        if stderr_thread:
            stderr_thread.join(timeout=5.0)
        
        print(f"脚本执行完成，返回值: {return_code}, 耗时: {t1 - t0:.2f}秒")
        
        # 获取完整的输出内容
        with stdout_lines_lock:
            full_stdout = "\n".join(stdout_lines)
        with stderr_lines_lock:
            full_stderr = "\n".join(stderr_lines)
        
        # 清理可能的特殊字符
        if full_stdout and "^^" in full_stdout:
            full_stdout = full_stdout.replace("^^", "")
        if full_stderr and "^^" in full_stderr:
            full_stderr = full_stderr.replace("^^", "")
        
        # 获取峰值显存
        peak_memory_mb = 0
        if total_mb and free_mb:
            while not result_queue.empty():
                peak_memory_mb = result_queue.get_nowait()
        
        # 最终检查所有提交文件
        matching_files = glob.glob(os.path.join(submission_csv_path, "submission*.csv"))
        print(f"submission_csv_path:{submission_csv_path}, 最终找到 {len(matching_files)} 个提交文件")
        
        # 构建返回结果
        result_base = {
            "success": return_code == 0 and len(matching_files) > 0,
            "error_info": {},
            "std_out": full_stdout[-1000:] if full_stdout else "",
            "wall_time": t1 - t0,
            "submission_csv": matching_files if len(matching_files) > 0 else None,
            "return_code": return_code,
            "tested_submission_files": tested_submission_files, 
        }
        
        # 如果执行失败，添加错误信息
        if not result_base["success"]:
            error_info = {
                "return_code": return_code,
                "has_stderr": len(stderr_lines) > 0,
            }
            if full_stderr:
                error_info["run_stderr_preview"] = full_stderr[-3000:]
            result_base["error_info"] = error_info
        
        # 添加GPU信息
        if total_mb and free_mb:
            memory_used = peak_memory_mb - (total_mb - free_mb)
            result_base.update({
                "peak_gpu_memory_mb": memory_used,
                "gpu_utilization": round(memory_used / free_mb, 4) if free_mb > 0 else 0,
            })
        
        return result_base
        
    except subprocess.TimeoutExpired:
        logger.error("脚本执行超时，正在终止进程...")
        t1 = time.time()

        # 安全终止进程
        if proc:
            safe_terminate_process(proc, "超时终止")
        # 清理资源
        cleanup_all_resources()

        # 获取峰值显存
        peak_memory_mb = 0
        if total_mb and free_mb:
            while not result_queue.empty():
                peak_memory_mb = result_queue.get_nowait()

        # 构建超时返回结果
        # 如果不存在测试过的提交文件，说明没有生成任何提交文件
        if not tested_submission_files:
            error_info = {"timeout": f"python script run timeout after {timeout} seconds"}

            if total_mb and free_mb:
                used_memory = calculate_gpu_memory(peak_memory_mb, total_mb, free_mb)
                gpu_util = calculate_gpu_utilization(used_memory, free_mb)
                result = build_evaluation_result(
                    success=False,
                    error_info=error_info,
                    wall_time=t1 - t0,
                    submission_csv=None,
                    peak_gpu_memory_mb=used_memory,
                    gpu_utilization=gpu_util,
                    std_out="",
                    return_code=ReturnCode.TIMEOUT,
                )
            else:
                result = build_evaluation_result(
                    success=False,
                    error_info=error_info,
                    wall_time=t1 - t0,
                    submission_csv=None,
                    std_out="",
                    return_code=ReturnCode.TIMEOUT,
                )
        else:  # 存在测试过的提交文件，说明至少生成了一个提交文件，需要进入二阶段
            error_info = {
                "timeout": (
                    f"python script run timeout after {timeout} seconds "
                    "(The training time is relatively long, and the training "
                    "for later epochs cannot be completed within the maximum time limit.)"
                )
            }

            base_result = {
                "success": True,
                "error_info": error_info,
                "std_out": full_stdout[-1000:] if full_stdout else "",
                "wall_time": t1 - t0,
                "submission_csv": list(tested_submission_files.keys()),
                "tested_submission_files": tested_submission_files,
                "return_code": ReturnCode.TIMEOUT,
            }

            # 添加GPU信息
            if total_mb and free_mb:
                used_memory = calculate_gpu_memory(peak_memory_mb, total_mb, free_mb)
                gpu_util = calculate_gpu_utilization(used_memory, free_mb)
                base_result["peak_gpu_memory_mb"] = used_memory
                base_result["gpu_utilization"] = gpu_util

            result = base_result

        return result
        
    except Exception as ex:
        logger.error(f"执行脚本时发生异常: {ex}")
        t1 = time.time()

        # 安全终止进程
        if proc:
            safe_terminate_process(proc, "异常终止")

        # 清理资源
        cleanup_all_resources()

        # 获取峰值显存
        peak_memory_mb = 0
        if total_mb and free_mb:
            while not result_queue.empty():
                peak_memory_mb = result_queue.get_nowait()

        # 构建异常返回结果
        error_info = get_error_info_from_exception(ex, traceback.format_exc())

        if total_mb and free_mb:
            used_memory = calculate_gpu_memory(peak_memory_mb, total_mb, free_mb)
            gpu_util = calculate_gpu_utilization(used_memory, free_mb)
            result = build_evaluation_result(
                success=False,
                error_info=error_info,
                wall_time=t1 - t0,
                submission_csv=None,
                peak_gpu_memory_mb=used_memory,
                gpu_utilization=gpu_util,
                std_out="",
                return_code=ReturnCode.GENERAL_ERROR,
            )
        else:
            result = build_evaluation_result(
                success=False,
                error_info=error_info,
                wall_time=t1 - t0,
                submission_csv=None,
                std_out="",
                return_code=ReturnCode.GENERAL_ERROR,
            )

        return result
        
    finally:
        # 最终资源清理（确保不会遗漏）
        cleanup_all_resources()
        print(f"执行流程完成，总耗时: {time.time() - t0:.2f}秒")


# ============================================================================
# 并行评估函数
# ============================================================================

def parallel_grade_submissions(
    submission_files: List[str],
    competition_name: str,
    cached_results: Dict[str, dict],
    config: ParallelConfig,
    data_dir: str = "/mnt/cfs_bj_mt/workspace/caolizhe/public/mlebench-competitions",
) -> Dict[str, dict]:
    """
    并行评估多个 submission 文件

    Args:
        submission_files: 待评估的文件路径列表
        competition_name: 比赛名称
        cached_results: 已缓存的评估结果 {file_path: result_dict}
        config: 并行配置
        data_dir: 数据目录

    Returns:
        所有文件的评估结果字典 {file_path: result_dict}
    性能优化：
        - 只对未缓存的文件并行调用 run_grader()
        - 使用 ThreadPoolExecutor 并行执行
        - 单个失败不影响其他
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 1. 分离缓存和未缓存的文件
    uncached_files = [f for f in submission_files if f not in cached_results]
    cached_count = len(submission_files) - len(uncached_files)

    print(
        f"[并行评估] 总文件数={len(submission_files)}, "
        f"已缓存={cached_count}, 需评估={len(uncached_files)}"
    )

    # 2. 如果全部已缓存，直接返回
    if not uncached_files:
        print("[并行评估] 所有文件均已缓存，跳过 grader 调用")
        return cached_results

    # 3. 如果未启用并行，回退到串行
    if not config.enable_parallel:
        print("[并行评估] 并行已禁用，使用串行调用")
        for file_path in uncached_files:
            print(f"[串行评估] 正在评估: {os.path.basename(file_path)}")
            result = run_grader(
                sub_csv=file_path,
                competition=competition_name,
                data_dir=data_dir,
                timeout=config.grader_timeout,
            )
            cached_results[file_path] = result
        return cached_results

    # 4. 并行调用 run_grader()
    parallel_start_time = time.time()
    results = {}
    completed_count = 0
    failed_count = 0

    try:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(
                    run_grader,
                    sub_csv=file_path,
                    competition=competition_name,
                    data_dir=data_dir,
                    timeout=config.grader_timeout,
                ): file_path
                for file_path in uncached_files
            }

            print(
                f"[并行评估] 已提交 {len(future_to_file)} 个任务, "
                f"线程池大小={config.max_workers}"
            )

            # 收集结果
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results[file_path] = result
                    completed_count += 1

                    # 记录完成情况
                    validity = result.get("validity", 0.0)
                    score = result.get("score", 0.0)
                    medal = result.get("medal", "none")

                    print(
                        f"[并行评估] 完成 {completed_count}/{len(uncached_files)}: "
                        f"{os.path.basename(file_path)} "
                        f"(validity={validity:.1f}, score={score:.4f}, medal={medal})"
                    )

                except Exception as e:
                    failed_count += 1
                    logger.error(
                        f"[并行评估] 异常: {os.path.basename(file_path)} - {e}"
                    )
                    # 为失败的文件构造错误结果
                    results[file_path] = {
                        "validity": 0.0,
                        "score": 0.0,
                        "medal": "none",
                        "above_median": False,
                        "error_info": {
                            "exception": str(e),
                            "exception_type": type(e).__name__,
                        },
                        "eval_wall_time": 0.0,
                        "competition_report": {},
                    }

    except Exception as e:
        logger.error(f"[并行评估] 线程池执行异常: {e}")
        raise

    # 5. 计算性能指标
    parallel_elapsed = time.time() - parallel_start_time
    speedup = len(uncached_files) / max(parallel_elapsed, 0.1) if parallel_elapsed > 0 else 0

    print(
        f"[并行评估] 完成: 成功={completed_count}, 失败={failed_count}, "
        f"耗时={parallel_elapsed:.2f}秒, "
        f"平均每个={parallel_elapsed / len(uncached_files):.2f}秒"
    )

    # 6. 合并缓存结果和新结果
    all_results = {**cached_results, **results}

    return all_results


def run_grader(
    sub_csv, 
    competition, 
    data_dir: str = "/mnt/cfs_bj_mt/workspace/caolizhe/public/mlebench-competitions", 
    timeout: float = 3600 * 8
):
    """
    调用 mlebench grade-sample 命令，捕获并解析输出，返回评测 json dict。
    自动处理 grader 无输出、输出非 json、json 解析失败等情况。
    兼容非标准 JSON（如单引号、None 等）。
    """
    cmd = [
        # "conda", "run", "-n", "my_mle", "mlebench",
        "mlebench",
        "grade-sample",
        sub_csv,
        competition ,
        "--data-dir", data_dir
    ]
    try:
        t0 = time.time()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        out, err = proc.communicate(timeout=timeout)
        if "^^" in err:
            err = err.replace("^^", "")
        if "^^" in out:
            out = out.replace("^^", "")
        t1 = time.time()

        # 尝试从stdout和stderr中提取json
        def extract_json(txt):
            m = re.search(r'(\{[\s\S]*?\})', txt)
            return m.group(1) if m else None

        # print(f"out:{out[-1000:]}")
        print(f"err:{err[-1000:]}")

        # 避免数据地址问题
        if "No such file or directory:" in err:
            cmd = [
                "mlebench",
                "grade-sample",
                sub_csv,
                competition,  # 修复PY030: 逗号后加空格
                # "--data-dir", "/mnt/cfs_bj_mt/workspace/caolizhe/pubic/mlebench-competitions/"
                "--data-dir", 
                "/mnt/cfs_bj_mt/workspace/caolizhe/pubic/mlebench-competitions/"  # 修复PY023: 列宽超长换行
            ]
            t0 = time.time()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
            )
            out, err = proc.communicate(timeout=timeout)
            t1 = time.time()

        json_str = extract_json(out) or extract_json(err)
        json_from = 'stdout' if extract_json(out) else ('stderr' if extract_json(err) else None)
        print(f"json_str:{json_str}")

        # 一般不会，除非是文件路径存在问题
        if not json_str:
            return {
                "validity": 0.0,
                "score": "null",
                "medal": "none",
                "above_median": False,
                "error_info": {
                    # "grader_stdout": out[-1000:],
                    "grader_stderr": err[-1000:],
                    "reason": "No JSON output found in grader stdout or stderr"
                },
                "std_out": out[-1000:],
                "meta_data": {},
                "eval_wall_time": t1 - t0,
            }

        # 尝试解析json，优先标准json，失败后用ast.literal_eval兜底
        try:
            res = robust_loads(json_str)
        except Exception as ex:
            return {
                "validity": 0.0,
                "score": 0.0,
                "medal": "none",
                "above_median": False,
                "error_info": {
                    # "grader_stdout": out[-1000:],
                    "grader_stderr": err[-1000:],
                    "json_str": json_str,
                    "exception": str(ex),
                    "traceback": traceback.format_exc(),
                    "reason": f"JSON decode error from {json_from}"
                },
                "std_out": out[-1000:],
                "meta_data": {},
                "eval_wall_time": t1 - t0,
                "competition_report": {}, 
            }

        print(f"res:\n{res}")
        # 校验得分是否正常
        try:
            score = float(res.get("score", 0.0))
        except Exception as ex:
            return {
                "validity": 0.0,
                "score": res.get("score", 0.0),
                "medal": "none",
                "above_median": False,
                "error_info": {
                    # "grader_stdout": out[-1000:],
                    "grader_stderr": err[-1000:],
                    # "competition_report": json_str,
                    # "exception": str(ex),
                    # "traceback": traceback.format_exc(),
                    # "reason": f"JSON decode error from {json_from}"
                },
                "grader_stdout": out[-1000:],
                "meta_data": {},
                "eval_wall_time": t1 - t0,
                "competition_report": res, 
            }

        # 成功提取与解析
        return {
            "validity": float(res.get("valid_submission", False)),
            "score": float(res.get("score", 0.0)),
            "medal": (
                "gold" if res.get("gold_medal") else
                "silver" if res.get("silver_medal") else
                "bronze" if res.get("bronze_medal") else
                "none"
            ),
            "above_median": bool(res.get("above_median", False)),
            "error_info": {},
            "meta_data": res,
            "eval_wall_time": t1 - t0,
            "competition_report": res, 
        }

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return {
            "validity": 0.0,
            "score": 0.0,
            "medal": "none",
            "above_median": False,
            "error_info": {"timeout": "grading process timeout"},
            "meta_data": {},
            "eval_wall_time": timeout,
            "competition_report": {}, 
        }
    except Exception as ex:
        return {
            "validity": 0.0,
            "score": 0.0,
            "medal": "none",
            "above_median": False,
            "error_info": {
                "exception": str(ex),
                "traceback": traceback.format_exc()
            },
            "meta_data": {},
            "eval_wall_time": 0.0,
            "competition_report": {}, 
        }


def evaluate(
    path_user_py,
    competition_name: str = "",
    gpu_ids: List[int] = None,
    conda_env_name: str = None,
    data_dir: str = "/mnt/cfs_bj_mt/workspace/caolizhe/public/mlebench-competitions",
    timeout: float = 6 * 60 * 60,
):
    """
    评估用户提交的代码

    Args:
        path_user_py: 用户代码路径
        competition_name: 比赛名称
        gpu_ids: Optional list of GPU IDs allocated for this evaluation (e.g., [0] or [0, 1])
        conda_env_name: Conda环境名称
        data_dir: 数据目录
        timeout: 超时时间（秒）

    Returns:
        评估结果字典
    """
    # 创建配置对象（替代全局变量）
    config = EvaluatorConfig()
    config.paths.data_dir = data_dir
    gpu_ids = gpu_ids or None
    config.gpu_ids = gpu_ids

    # 根据比赛类型设置指标方向（替代全局 LOWER_IS_BETTER）
    config.lower_is_better = competition_name not in POSITIVE_COMPETETIONS
    print(f"competition_name:{competition_name}, lower_is_better:{config.lower_is_better}, gpu_ids:{gpu_ids}")


    # 检查当前文件夹下是否已经有submission_csv，存在就删除，避免使用之前的结果
    submission_csv_path = os.path.join(config.paths.forced_submission_dir, competition_name)
    os.makedirs(submission_csv_path, exist_ok=True)
    # 匹配所有以 submission 开头，以 .csv 结尾的文件
    sub_csv_paths: List[str] = glob.glob(os.path.join(submission_csv_path, "submission*.csv"))
    print(f"current sub_csv_paths:\n{sub_csv_paths}")
    for sub_csv_path in sub_csv_paths:
        # 如果是文件夹
        if os.path.isdir(sub_csv_path):
            shutil.rmtree(sub_csv_path)
        elif os.path.isfile(sub_csv_path):
            os.remove(sub_csv_path)

    # 确保模型保存地址存在
    # 检查model_save目录下是否存在模型权重，存在就删除
    model_save_path = os.path.join(submission_csv_path, "model_save")
    os.makedirs(model_save_path, exist_ok=True)
    print(f"model_save_path:\n{model_save_path}")
    for model_dir in os.listdir(model_save_path):
        # 如果是文件夹
        if os.path.isdir(os.path.join(model_save_path, model_dir)):
            shutil.rmtree(os.path.join(model_save_path, model_dir))
        elif os.path.isfile(os.path.join(model_save_path, model_dir)):
            os.remove(os.path.join(model_save_path, model_dir))

    step1 = run_python_script(
        path_user_py, competition_name, gpu_ids, conda_env_name=conda_env_name, timeout=timeout)
    print(f"step1:\n{step1}")
    # 保存 stderr 信息，即使 success=True 也可能需要传递给下游
    run_stderr = step1.get("error_info", {}).get("run_stderr_preview", "")

    if not step1["success"]:
        # 脚本报错但已产生有效提交文件时，保留分数并传递 stderr 给下游修复
        tested = step1.get("tested_submission_files", {})
        valid_results = {k: v for k, v in tested.items() if v.get("validity", 0) >= 0.9}
        if valid_results:
            best_key = max(valid_results, key=lambda k: valid_results[k].get("score", 0))
            best = valid_results[best_key]
            print(
                f"[evaluator] 脚本报错(return_code={step1['return_code']})但存在有效提交，"
                f"使用已有评分: score={best.get('score')}"
            )
            # 放行走正常评分流程，stderr 会在最终结果中附加
            step1["success"] = True
            step1["submission_csv"] = list(valid_results.keys())
        else:
            # 超时导致的（多半是模式模型下载问题）
            if "timeout" in step1["error_info"] and step1["error_info"]["timeout"] == "python script run timeout":
                timeout_program_save_folder = os.path.join(submission_csv_path, "timeout_program_save")
                if not os.path.exists(timeout_program_save_folder):
                    os.makedirs(timeout_program_save_folder)
                move_and_rename_file(
                    path_user_py, os.path.join(timeout_program_save_folder, f"timeout_code.py"))

            result = {
                "validity": 0.0,
                "score": 0,
                "combined_score": 0.0,
                "medal": "none",
                "above_median": False,
                "error_info": {
                    "run_error": step1["error_info"]
                },
                "std_out": step1.get("std_out", ""),
                "meta_data": {},
                "eval_wall_time": step1["wall_time"],
            }
            if step1.get("submission_csv.head()", ""):
                result["error_info"]["submission_csv.head()"] = step1["submission_csv.head()"]
            return result

    print("------------------ step1 run successfully ---------------------")

    try:
        sub_csv_paths = step1["submission_csv"]
        tested_submission_files = step1.get("tested_submission_files", {})
        grader_res = None
        # 记录最高得分出现的轮次, 得分与比赛报告
        best_score_record = {
            "validity": 0.0, "epoch_idx": -1,
            "score": -1.0, "combined_score": -1.0, 
            "medal": "none", "above_median": False, 
            "competition_report": {}, "submission_csv": "", 
            "model_save_path": None, 
        }
        # 记录每个epoch的结果 epoch:score
        score_each_epoch = {}

        # ========== 并行评估所有 submission 文件 ==========
        grader_start_time = time.time()
        all_grader_results = parallel_grade_submissions(
            submission_files=sub_csv_paths,
            competition_name=competition_name,
            cached_results=tested_submission_files,
            config=config.parallel,
            data_dir=config.paths.data_dir,
        )
        grader_elapsed = time.time() - grader_start_time
        print(f"[evaluate] Grader 阶段总耗时: {grader_elapsed:.2f}秒")
        # ==================================================

        # 按原始顺序处理所有结果
        for sub_csv_path in sub_csv_paths:
            grader_res = all_grader_results[sub_csv_path]
            grader_res["eval_wall_time"] += step1["wall_time"]
            grader_res["peak_gpu_utilization"] = step1["gpu_utilization"]
            
            # combined_score 归一化
            def scale_score(raw_score):
                # 调整方向：越好越大（使用配置对象）
                score = -raw_score if config.lower_is_better else raw_score
                # 用 sigmoid 压缩到 (0,1)
                combined_score = 1 / (1 + np.exp(-score))
                return combined_score

            def scale_score_with_medal(raw_score, medal="none"):
                def softsign(x):
                    return x / (1 + abs(x))

                score = -raw_score if config.lower_is_better else raw_score
                combined_score = softsign(score)  # 输出区间 (-1,1)

                # 可映射到 (0,1)
                combined_score = (combined_score + 1) / 2

                offsets = {"none": 0, "bronze": 1, "silver": 2, "gold": 3}
                return combined_score + offsets.get(medal, 0)

            # 先判断有效性，无效直接得分为0
            if grader_res["validity"] <= 0:
                grader_res["combined_score"] = 0.0
                if best_score_record["validity"] <= 0.1:
                    best_score_record = grader_res
                    best_score_record["submission_csv"] = sub_csv_path
                    break
            else:
                grader_res["combined_score"] = scale_score_with_medal(
                    grader_res["score"], grader_res.get("medal", "none")
                )

                # 提取 epoch 编号
                epoch_idx = extract_submission_idx(sub_csv_path)
                if epoch_idx != -1:  # 如果有 epoch 编号，即训练中
                    score_each_epoch[str(epoch_idx)] = grader_res["score"]

                # 记录最好的
                if best_score_record["combined_score"] < grader_res["combined_score"]:  
                    best_score_record = grader_res
                    best_score_record["submission_csv"] = sub_csv_path
                    best_score_record["epoch_idx"] = epoch_idx
                # print(f"best_score_record:\n{best_score_record}")
        
        score = best_score_record["score"]
        combined_score = best_score_record.get("combined_score", 0.0)
        validity = best_score_record.get("validity", 0.0)
        print(f"best_score_record:\n{best_score_record}")
        
        # 首先结果有效才判断
        if best_score_record["validity"] >= 0.9:
            submission_csv = pd.read_csv(best_score_record["submission_csv"])
            out = (
                "The first few lines of the submission file generated by the program "
                f"are as follows:\n{str(submission_csv.head())}\n"
            )
            out = "The indicators of the models trained with different epoches on the test set are as follows:\n"
            is_unfitting = True  # 判断是否欠拟合
            pre_score = 0
            for epoch_idx in sorted([int(item) for item in list(score_each_epoch.keys())]):
                out += f"epoch:{epoch_idx}, test datasets score: {score_each_epoch[str(epoch_idx)]}\n"
                if pre_score == 0:
                    pre_score = score_each_epoch[str(epoch_idx)]
                elif score_each_epoch[str(epoch_idx)] < pre_score:
                    is_unfitting = False 
                pre_score = score_each_epoch[str(epoch_idx)]

            # 放在 std_out 中
            out += (
                "Suggestions: If it is underfitting, you can increase the number of "
                "epochs for training, otherwise you can reduce it appropriately.\n"
            )
            if is_unfitting:
                out += (
                    "Judging from the test set indicators of each epoch, "
                    "you should increase the number of training epochs appropriately."
                )
            best_score_record["std_out"] = out

            # 转移保存的训练模型
            try:
                save_epoch_idx = best_score_record["epoch_idx"]
                print(f"model_save_path:{model_save_path}, save_epoch_idx:{save_epoch_idx}")
                final_save_path = save_model(
                    original_model_dir=model_save_path, 
                    competition_name=competition_name, 
                    save_epoch_idx=save_epoch_idx, 
                    score=best_score_record["combined_score"]
                )
                print(f"final_save_path:\n{final_save_path}")
                best_score_record["model_save_path"] = final_save_path
            except Exception as e:
                logger.warning(f"Failed to transfer model: {str(e)}")
                best_score_record["model_save_path"] = None

    finally:
        # 安全删除所有的 submission.csv
        try:
            sub_csv_folder = os.path.join("/".join(sub_csv_path.split("/")[:-1]), "submission_save")
            if not os.path.exists(sub_csv_folder):
                os.makedirs(sub_csv_folder)
            print(f"move all submission files into folder:{sub_csv_folder}")
            
            program_save_folder = os.path.join("/".join(sub_csv_path.split("/")[:-1]), "program_save")
            if not os.path.exists(program_save_folder):
                os.makedirs(program_save_folder)
            print(f"move all program files into folder:{program_save_folder}")

            # 保存迁移最优的预测文件 && code（使用配置类中的阈值）
            thresholds = config.thresholds
            if combined_score > thresholds.GOLD_MEDAL_SCORE:
                print("=================== Get gold medal ===================")
                move_and_rename_file(
                    best_score_record["submission_csv"],
                    os.path.join(sub_csv_folder, f"submission_gold_{score}.csv"))
                move_and_rename_file(path_user_py, os.path.join(program_save_folder, f"code_gold_{score}.py"))
            elif combined_score > thresholds.SILVER_MEDAL_SCORE:
                print("=================== Get silver medal ===================")
                move_and_rename_file(
                    best_score_record["submission_csv"],
                    os.path.join(sub_csv_folder, f"submission_silver_{score}.csv"))
                move_and_rename_file(path_user_py, os.path.join(program_save_folder, f"code_silver_{score}.py"))
            elif combined_score > thresholds.BRONZE_MEDAL_SCORE:
                print("=================== Get bronze medal ===================")
                move_and_rename_file(
                    best_score_record["submission_csv"],
                    os.path.join(sub_csv_folder, f"submission_bronze_{score}.csv"))
                move_and_rename_file(path_user_py, os.path.join(program_save_folder, f"code_bronze_{score}.py"))
            elif validity > 0.9:  # 保存分数最高的预测文件
                move_and_rename_file(
                    best_score_record["submission_csv"],
                    os.path.join(sub_csv_folder, f"submission_{score}.csv"))
                # move_and_rename_file(
                #     path_user_py, os.path.join(program_save_folder, f"code_{score}.py"))
            
            # 其余文件直接删除
            for sub_csv_path in sub_csv_paths:
                if sub_csv_path and os.path.exists(sub_csv_path):
                    os.remove(sub_csv_path)

        except Exception as e:
            # 删除失败也不影响评测流程，只打印警告
            print(f"Warning: Failed to remove {sub_csv_path}: {str(e)}")
        
    # 如果脚本有 stderr 但被放行了，把 stderr 附加到最终结果的 error_info 中
    # 这样下游 error_aware / debug_judge 仍能看到错误并尝试修复
    if run_stderr and best_score_record.get("validity", 0) >= 0.9:
        if not best_score_record.get("error_info"):
            best_score_record["error_info"] = {}
        if isinstance(best_score_record["error_info"], dict):
            best_score_record["error_info"]["run_stderr_preview"] = run_stderr
        else:
            best_score_record["error_info"] = {"run_stderr_preview": run_stderr}

    return best_score_record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path_user_py", type=str, help="Path to user python script")
    parser.add_argument(
        "--competition_name", type=str,
        default="nomad2018-predict-transparent-conductors",
        help="Competition name"
    )
    parser.add_argument(
        "--gpu_ids", type=str, default="0",
        help="Comma-separated GPU IDs, e.g. '0' or '0,1'"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/mnt/cfs_bj_mt/workspace/caolizhe/public/mlebench-competitions",
        help="Data dir"
    )
    parser.add_argument("--timeout", type=int, default=3600 * 4, help="Timeout seconds")
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    result = evaluate(
        args.path_user_py,
        args.competition_name,
        gpu_ids,
        None,
        args.data_dir,
        timeout=args.timeout)
    print("--- RESULT ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))