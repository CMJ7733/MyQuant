"""
Algorithm modules for Famou 2.0.

Base classes (import from here):
- SelectModule: Choose parent programs for generation
- GenerateModule: Create new programs from parents
- EvaluateModule: Score programs (required, uses user evaluator)
- JudgeModule: Enrich programs post-generation (LLM feedback, features, metadata, state)
- PopulationModule: Update population after rollout

Implementations (import from submodule files):

Selection:
- famou.modules.select.elite.EliteSelect (recommended)
- famou.modules.select.random.RandomSelect
- famou.modules.select.tournament.TournamentSelect
- famou.modules.select.diversity.DiversitySelect
- famou.modules.select.cluster_adaptive.ClusterAdaptiveSelect

Generation:
- famou.modules.generate.mutation.MutationGenerate
- famou.modules.generate.crossover.CrossoverGenerate
- famou.modules.generate.error_driven.ErrorDrivenGenerate

Enrichment (all inherit JudgeModule):
- famou.modules.judge.llm_judge.LLMJudge (LLM feedback)
- famou.modules.judge.embedding_feature.EmbeddingFeature (semantic embeddings)
- famou.modules.judge.experience_collector.ExperienceCollector (strategy state)

Population:
- famou.modules.population.topk.TopKPopulation
- famou.modules.population.cluster.ClusterPopulation

Each module follows the protocol:
    execute(context: Context, result: RolloutResult) -> RolloutResult
"""

from famou.modules.select import SelectModule
from famou.modules.generate import GenerateModule
from famou.modules.evaluate import EvaluateModule
from famou.modules.judge import JudgeModule
from famou.modules.population import PopulationModule

__all__ = [
    "SelectModule",
    "GenerateModule",
    "EvaluateModule",
    "JudgeModule",
    "PopulationModule",
]
