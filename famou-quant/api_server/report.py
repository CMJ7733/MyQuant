"""
Report generation module for evolutionary experiment results.

This module provides functionality to analyze experiment checkpoints, generate insights
using LLM, and produce HTML reports. It supports both legacy and new experiment formats,
handling data extraction, code analysis, and visualization data preparation.
"""

import os
import json
import re
import statistics
import math
import random
import asyncio
from openai import (
    AsyncOpenAI,
    RateLimitError,
    APIError,
    APITimeoutError,
    APIConnectionError,
)

CLIENT = None


async def generate(model: str, prompt: str, max_retries: int = 5) -> str:
    """
    异步生成文本，并手动实现重试逻辑。
    """
    exceptions_to_retry = (
        RateLimitError,
        APIError,
        APITimeoutError,
        APIConnectionError,
    )
    base_delay = 1  # 基础延迟时间（秒）

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}...")
            completion = await CLIENT.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return completion.choices[0].message.content
        except exceptions_to_retry as e:
            if attempt == max_retries - 1:
                print(f"Final attempt failed. Raising exception: {e}")
                raise

            # 指数退避 + 随机抖动
            delay = (base_delay * 2**attempt) + random.uniform(0, 1)
            print(
                f"Attempt {attempt + 1} failed with {type(e).__name__}. Retrying in {delay:.2f} seconds..."
            )
            await asyncio.sleep(delay)
        except Exception as e:
            # 捕获其他非预期的异常，直接抛出，不重试
            print(f"An unexpected error occurred: {e}")
            raise

    raise Exception("Failed to generate completion after all retries.")


def extract_json(response_text: str):
    """
    Extracts a JSON object from a string. It handles cases where the JSON is
    embedded in a larger text or a markdown code block.
    """
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    try:
        start_index = response_text.find("{")
        end_index = response_text.rfind("}")
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = response_text[start_index : end_index + 1]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        raise e

    return {}


async def scan_ckpt(path: str):
    """扫描新版扁平化 checkpoint 和 programs 目录

    新版目录结构:
        <experiment_root>/
        ├── programs/<id>.json        — 所有程序（含 seed）
        └── experiment_checkpoint_N.json — 迭代快照

    程序 JSON 关键字段:
        id: str ("seed_0" 或 "{iter}_{gen}_{hash}")
        code: str
        combined_score: float (顶层)
        metrics: {...} (不含 combined_score)
        iteration: int
        generation: int
        parent_id: str | null
        language: str
        validity: float (顶层)

    Checkpoint JSON 关键字段:
        current_iteration: int
        best_program_id: str
        best_program_score: float
    """
    # 1. 从 programs/ 目录收集所有程序数据
    programs_dir = os.path.join(path, "programs")
    data = {}
    for filename in os.listdir(programs_dir):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(programs_dir, filename), "r", encoding="utf-8") as f:
            content = json.load(f)
            data[content["id"]] = content

    # 2. 遍历 checkpoint 文件，构建 best_scores 轨迹
    ckpt_pattern = re.compile(r"^experiment_checkpoint_(\d+)\.json$")
    ckpt_files = sorted(
        [f for f in os.listdir(path) if ckpt_pattern.match(f)],
        key=lambda x: int(ckpt_pattern.match(x).group(1)),
    )

    best_scores = []
    best_program_id = ""
    for ckpt_file in ckpt_files:
        with open(os.path.join(path, ckpt_file), "r", encoding="utf-8") as f:
            ckpt = json.load(f)
            score = ckpt.get("best_program_score", 0)
            if not best_scores or best_scores[-1][1] < score:
                best_scores.append([ckpt["current_iteration"], score])
                best_program_id = ckpt["best_program_id"]

    return data, best_scores, best_program_id


async def get_scatters(data):
    """获取散点图数据

    新版字段: iteration (替代 iteration_found), combined_score (顶层, 替代 metrics.combined_score)
    """
    scatters = {}
    for k, v in data.items():
        if v["iteration"] not in scatters:
            score = v.get("combined_score")
            if score is None or math.isinf(score) or math.isnan(score):
                continue
            scatters[v["iteration"]] = v.get("combined_score", 0)
    return [[k, v] for k, v in scatters.items()]


async def get_best_program_path(data, best_program_id, model: str):
    """获取最佳程序路径

    新版变更:
        - iteration_found -> iteration
        - metrics.combined_score -> combined_score (顶层)
        - 移除 previous_ids 回溯逻辑: 新版 parent_id 为 None 即为 seed, 无需额外判断
    """
    prompt = """以下是两段解决同一问题的代码，其中一段是提升版本，另一段是原始版本。请用十个字以内总结提升版本相较于原始版本的一个最核心的具体改进之处。
原始版本代码:
{}

提升版本代码:

{}

请直接返回你的总结不要返回其他内容。
    """
    path = []
    current_id = best_program_id
    while current_id is not None and current_id in data:
        path.append(
            [
                data[current_id]["iteration"],
                data[current_id].get("combined_score", 0),
            ]
        )
        parent_id = data[current_id].get("parent_id")
        if parent_id is not None and parent_id != "" and parent_id in data:
            prompt_text = prompt.format(
                data[parent_id]["code"], data[current_id]["code"]
            )
            response = await generate(model, prompt_text)
            print(response)
            path[-1].append(response)
        current_id = parent_id if parent_id not in (None, "") else None
    return path[::-1]


async def get_evolution_map(data, best_scores, best_program_id, model: str):
    """获取进化地图数据"""
    ret = {}
    scatters = await get_scatters(data)
    best_path = await get_best_program_path(data, best_program_id, model)
    ret["scatter_points"] = scatters
    best_scores.append(
        [max([point[0] for point in scatters]), max([point[1] for point in best_scores])])
    ret["best_scores"] = best_scores
    ret["y_axis_min"] = statistics.median([i[1] for i in scatters])
    ret["top1_path"] = best_path
    return ret


async def get_best_solution(path, best_data, model: str):
    """获取最佳解信息

    新版变更:
        - 不再从 init/init/checkpoints/ 读取初始程序
        - 改为从 programs/ 目录查找 seed 程序 (generation=0 或 id 以 "seed_" 开头)
        - init_metrics: 新版 combined_score 在顶层, metrics 子对象不含 combined_score
    """
    prompt = """以下是两段解决同一问题的代码，其中一段是提升版本，另一段是原始版本。请在以下几个维度评估提升版本和原始版本，每个维度都请用自然语言回答：
    1. 提升版本的核心策略
    2. 提升版本和原始版本思路和做法上的不同
    3. 提升版本的指标和原始版本的指标的不同

    原始版本代码：
    {}
    原始版本指标:
    {}

    提升版本代码：
    {}
    提升版本指标:
    {}
    请以json格式返回一个字典，包含以下键值对:
    {{
        "strategy": "{{提升版本的核心策略}}",
        "code_diff": "{{提升版本和原始版本思路和做法上的不同}}",
        "score_diff": "{{提升版本的指标和原始版本的指标的不同}}"
    }}
    """

    # 从 programs/ 目录查找 seed 程序
    programs_dir = os.path.join(path, "programs")
    init_data = None
    for filename in os.listdir(programs_dir):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(programs_dir, filename), "r", encoding="utf-8") as f:
            prog = json.load(f)
            if prog.get("generation") == 0 or prog.get("id", "").startswith("seed_"):
                init_data = prog
                break

    if init_data is None:
        raise FileNotFoundError("No seed program found in programs/")

    init_code = init_data["code"]
    # 新版: combined_score 在顶层, 将其合并进 metrics 用于展示
    init_metrics = dict(init_data.get("metrics", {}))
    init_metrics["combined_score"] = init_data.get("combined_score", 0)
    init_metrics["validity"] = init_data.get("validity", 0)

    # 新版: best_data 同样需要将顶层字段合并进 metrics 用于展示
    best_metrics = dict(best_data.get("metrics", {}))
    best_metrics["combined_score"] = best_data.get("combined_score", 0)
    best_metrics["validity"] = best_data.get("validity", 0)

    prompt_text = prompt.format(
        init_code, init_metrics, best_data["code"], best_metrics
    )
    response = await generate(model, prompt_text)
    result = None
    while not result:
        try:
            print(response)
            result = extract_json(response)
        except Exception as e:
            print(e)
            response = await generate(model, prompt_text)

    ret = {
        "program_id": best_data["iteration"],
        "score": best_data.get("combined_score", 0),
        "strategy": result["strategy"],
        "code_diff": result["code_diff"],
        "score_diff": {
            "combined_score_old": init_metrics.get("combined_score", 0),
            "combined_score_new": best_data.get("combined_score", 0),
            "ratio": (
                best_data.get("combined_score", 0)
                / (init_metrics.get("combined_score", 0)
                if init_metrics.get("combined_score", 0) != 0
                else 0.0001)
            ),
            "analysis": result["score_diff"],
        },
    }
    return ret


async def get_code(best_data, model: str):
    """分析代码

    字段 language 和 code 与旧版一致，无需修改。
    """
    prompt = """我们给你提供了一段代码，请分块分析每部分代码的作用和逻辑，并json的形式返回，其中key为行号范围使用-分隔(也可以是单独一个数字代表单行)，value为对应的注释，例如：
{{
    "1-4": "这是第一部分代码",
    "6-9": "这是第二部分代码"
}}

代码：
{}

请直接返回你的json
"""
    best_code = best_data["code"]

    lines = best_code.split("\n")

    code_by_line = ""
    for idx, line in enumerate(lines):
        code_by_line += f"line {idx+1}: {line}\n"

    result = None
    response = await generate(model, prompt.format(code_by_line))
    while not result:
        try:
            print(response)
            result = extract_json(response)
            result = list(
                sorted(result.items(), key=lambda item: int(item[0].split("-")[0]))
            )
        except Exception as e:
            print(e)
            response = await generate(model, prompt.format(code_by_line))

    return {"language": best_data["language"], "lines": lines, "explanations": result}


def _load_evolve_data(path):
    """从 config.yaml + 最后一个 checkpoint 构建实验元数据

    新版产出不含 evolve.json，需从以下文件构建等价数据:
        config.yaml → user_requirements, evaluation_file, parameters
        experiment_checkpoint_N.json → execution_results

    返回数据结构:
        {
            "execution_results": { total_iteration, best_combined_score, ... },
            "parameters": { population_size, num_islands, max_iterations, coldstart },
            "user_requirements": "...",
            "evaluation_file": "..."
        }
    """
    import yaml

    with open(os.path.join(path, "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 从最后一个 checkpoint 获取执行结果
    ckpt_pattern = re.compile(r"^experiment_checkpoint_(\d+)\.json$")
    ckpt_files = sorted(
        [f for f in os.listdir(path) if ckpt_pattern.match(f)],
        key=lambda x: int(ckpt_pattern.match(x).group(1)),
    )
    last_ckpt = {}
    if ckpt_files:
        with open(os.path.join(path, ckpt_files[-1]), "r", encoding="utf-8") as f:
            last_ckpt = json.load(f)

    # 从 best_program_id 对应的程序 JSON 中读取真实的 iteration 和 generation
    best_iteration = last_ckpt.get("current_iteration", 0)
    best_generation = 0
    best_program_id = last_ckpt.get("best_program_id", "")
    if best_program_id:
        best_prog_file = os.path.join(path, "programs", f"{best_program_id}.json")
        if os.path.isfile(best_prog_file):
            with open(best_prog_file, "r", encoding="utf-8") as f:
                best_prog = json.load(f)
            best_iteration = best_prog.get("iteration", best_iteration)
            best_generation = best_prog.get("generation", 0)

    exp_config = config.get("experiment", {})
    island_config = exp_config.get("island", {})

    # 统计 programs/ 目录下所有程序数（包括 init/seed）
    programs_dir = os.path.join(path, "programs")
    total_programs = (
        len([f for f in os.listdir(programs_dir) if f.endswith(".json")])
        if os.path.isdir(programs_dir)
        else 0
    )
    data = {
        "execution_results": {
            "total_iteration": total_programs, #生成节点数, 需要包括初始化节点个数
            "best_combined_score": last_ckpt.get("best_program_score", 0),
            "best_iteration": best_iteration,
            "best_generation": best_generation,
        },
        "parameters": {
            "population_size": island_config.get("population_size", 0),
            "num_islands": island_config.get("num_islands", 0),
            "max_iterations": exp_config.get("max_iterations", 0),
            "coldstart": False,
        },
        "user_requirements": exp_config.get("task_description", ""),
        "evaluation_file": exp_config.get("evaluator", ""),
    }
    return data


async def get_backround(path, model: str):
    """分析问题背景

    新版产出不含 evolve.json，元数据从 config.yaml + checkpoint 构建:
        - user_requirements 来自 config.yaml experiment.task_description
        - evaluation_file 来自 config.yaml experiment.evaluator
        - execution_results 来自最后一个 experiment_checkpoint_N.json
        - evaluation_file 可能是相对路径，需做路径解析
        - coldstart 信息从元数据中获取
    """
    prompt = """根据下面提供的问题定义和评估代码使用中文做一段摘要，不要输出任何JSON或列表，不要罗列代码。可以覆盖以下点：

这是单一目标还是多目标；

采用了哪些指标及其方向/单位与权重（若有）；

是否做了归一化/变换（如 min-max、log、clip）；

惩罚项与有效性处理（编译/运行失败、缺失值如何计分）；

综合分如何计算（用一句话+简洁伪公式描述即可）；

稳定性/随机性（是否设seed、多次评估取均/中位等）；

若能判断，与"用户目标/KPI"的一致性与1–3条改进建议。

问题定义：
{}

评估代码：
{}

请直接返回你的摘要。
    """
    data = _load_evolve_data(path)

    # 处理 evaluation_file: 可能是绝对路径或相对路径
    eval_file = data.get("evaluation_file", "")
    if eval_file:
        if not os.path.isabs(eval_file):
            eval_file = os.path.join(path, eval_file)
        with open(eval_file, "r", encoding="utf-8") as f:
            code = f.read()
    else:
        code = ""

    prompt_text = prompt.format(data.get("user_requirements", ""), code)
    response = await generate(model, prompt_text)
    print(response)
    data.pop("evaluation_file", None)
    data["user_requirements"] = response

    return data


def inject_data(data_path, output_path):
    """
    Reads data from a JSON file and injects it into an HTML file,
    modifying the HTML to read the embedded data.
    """
    print(f"Starting data injection...")
    HTML_INPUT_FILE = os.path.join(os.path.dirname(__file__), "dashboard.html")

    # --- 1. Read HTML Template ---
    try:
        with open(HTML_INPUT_FILE, "r", encoding="utf-8") as f:
            html_content = f.read()
        print(f"Successfully read '{HTML_INPUT_FILE}'")
    except FileNotFoundError:
        print(f"Error: HTML input file '{HTML_INPUT_FILE}' not found.")
        return
    except Exception as e:
        print(f"Error reading HTML file: {e}")
        return

    # --- 2. Read JSON Data ---
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            json_data_string = f.read()

        json.loads(json_data_string)
        print(f"Successfully read and validated '{data_path}'")
    except FileNotFoundError:
        print(f"Error: JSON input file '{data_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: '{data_path}' does not contain valid JSON.")
        return
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # --- 3. Create the <script> tag with JSON data ---
    script_tag_to_inject = f'<script id="injected-data" type="application/json">{json_data_string}</script>'

    # --- 4. Inject the data into the HTML <head> ---
    if "</head>" in html_content:
        html_content = html_content.replace(
            "</head>", f"{script_tag_to_inject}\n</head>", 1
        )
        print("Injected JSON data script tag into HTML <head>.")
    else:
        print("Warning: No </head> tag found. Data not injected.")
        return

    # --- 5. Modify the JavaScript fetch call ---
    original_fetch = "const response = await fetch('./data-1.json')"
    replacement_fetch = (
        "const jsonData = document.getElementById('injected-data').textContent;"
    )

    if original_fetch in html_content:
        html_content = html_content.replace(original_fetch, replacement_fetch, 1)
        print("Modified JavaScript 'fetch' line.")
    else:
        print(f'Warning: Could not find the line: "{original_fetch}"')
        print("The script might not work as expected.")

    # --- 6. Modify the JavaScript .json() call ---
    original_parse = "globalData = await response.json()"
    replacement_parse = "globalData = JSON.parse(jsonData);"

    if original_parse in html_content:
        html_content = html_content.replace(original_parse, replacement_parse, 1)
        print("Modified JavaScript '.json()' line.")
    else:
        print(f'Warning: Could not find the line: "{original_parse}"')
        print("The script might not work as expected.")

    # --- 7. Write the new HTML file ---
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"\nSuccess! New file created: '{output_path}'")
    except Exception as e:
        print(f"Error writing output file: {e}")


async def build_report(path, llm_config, output_dir):
    """build report

    path: 实验产出根目录，如 circle_packing_evolution_20260408_160707_328b/

    兼容两种产出格式:
        - 含 evolve.json: 直接读取元数据
        - 不含 evolve.json: 从 config.yaml + checkpoint 构建

    输出:
        - output_dir/data.json     — 聚合数据
        - output_dir/report.html   — 可视化报告
    """
    global CLIENT
    CLIENT = AsyncOpenAI(
        api_key=llm_config.api_key,
        base_url=llm_config.api_base,
        max_retries=3,
    )
    model = llm_config.model
    data_path = os.path.join(output_dir, "data.json")
    output_path = os.path.join(output_dir, "report.html")
    bak_output_path = os.path.join(path, "report.html")
    

    data, best_scores, best_program_id = await scan_ckpt(path)
    evolution_map = await get_evolution_map(data, best_scores, best_program_id, model)
    best_solution = await get_best_solution(path, data[best_program_id], model)
    code = await get_code(data[best_program_id], model)
    background = await get_backround(path, model)
    out = {
        "evolution_map": evolution_map,
        "best_solution": best_solution,
        "code": code,
        "background": background,
    }

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    inject_data(data_path, output_path)
    inject_data(data_path, bak_output_path)
    


if __name__ == "__main__":
    inject_data(
        "/data/baidu/acg/fm-agent/famou/examples/u_id/mock_task/v_1/data.json",
        "./report.html",
    )
