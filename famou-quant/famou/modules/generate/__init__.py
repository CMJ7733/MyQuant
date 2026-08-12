"""
Generation modules for Famou 2.0.

Base class:
- GenerateModule: Inherit to create custom generation strategies

Built-in implementations (import from files directly):

Evolutionary Generation:
- mutation.MutationGenerate: LLM-based code mutation
- crossover.CrossoverGenerate: Combine two parent programs
- error_driven.ErrorDrivenGenerate: Focus on fixing program errors

Specialized Generation:
- model_fusion.ModelFusionGenerate: Create ensemble from multiple models
- gen_init_feasible.GenInitFeasibleGenerate: Generate simple baseline solutions
- draft.DraftGenerate: Propose novel and diverse solutions
- feature_engineering.FeatureEngineeringGenerate: Advanced feature engineering
- test_time_adaptation.TestTimeAdaptationGenerate: Test-time augmentation and TTA
- data_augmentation.DataAugmentationGenerate: Training-time data augmentation
- data_cleaning.DataCleaningGenerate: Data cleaning and preprocessing

Experience-Guided Generation:
- experience_guided.ExperienceGuidedGenerate: Use past successful transformations
"""

from famou.modules.generate.base import GenerateModule

__all__ = ["GenerateModule"]
