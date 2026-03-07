"""Shared utility module for the research analysis pipeline.

This package provides data loading, metric computation, voting/sampling,
statistical testing, and visualisation utilities for analysing whether
SFT-trained LLMs can evaluate research idea quality.
120 articles are rated by AI models and human raters (experts and students).

Submodules
----------
constants
    Label orderings, colour palettes, model display names, and file paths.
data_loader
    Functions to load JSONL data files and normalise labels.
metrics
    Classification metrics, calibration metrics, and agreement measures.
voting
    Majority voting, probability distributions, and Monte Carlo sampling.
statistical_tests
    Parametric and non-parametric tests, bootstrap CI, multiple-testing
    corrections, and inter-rater reliability measures.
visualization
    Nature/Science-style plotting utilities.
"""

# -- constants ---------------------------------------------------------------
from .constants import (
    AI_TO_HUMAN,
    AI_TO_LEVEL,
    ALL_COMBINED_PATH,
    ARTICLES_PATH,
    CHAT_MODELS,
    CHAT_MODELS_PRIMARY,
    CHAT_MODELS_SUPPLEMENTAL,
    CHAT_PATH,
    COLORS,
    DATA_ROOT,
    ENRICHED_EXPERT_FILTERED_PATH,
    ENRICHED_EXPERT_PATH,
    ENRICHED_STUDENT_FILTERED_PATH,
    ENRICHED_STUDENT_PATH,
    EXPERT_FILTERED_PATH,
    EXPERT_PATH,
    EXPERT_PROFILES_DIR,
    FIGURES_DIR,
    FRONTIER_DISPLAY_NAMES,
    FRONTIER_MODELS,
    FRONTIER_PATH,
    HUMAN_TO_AI,
    LABEL_ORDER,
    LEVEL_TO_AI,
    MODEL_DISPLAY_NAMES,
    N_ARTICLES,
    N_PER_TIER,
    OLD_STUDENT_BG_PATH,
    RESULTS_ROOT,
    SFT_BASE_MODELS,
    SFT_MODELS,
    STATS_DIR,
    STUDENT_MERGED_PATH,
    STUDENT_NEW_PATH,
    STUDENT_OLD_PATH,
    TABLES_DIR,
    THINKING_AVG8_PATH,
    THINKING_LEGACY_PATH,
    THINKING_PATH,
    TRAINED_PATH,
    UNFILTERED_EXPERT_PATH,
    UNFILTERED_STUDENT_PATH,
    VAL_PATH,
)

# -- data_loader -------------------------------------------------------------
from .data_loader import (
    build_article_index,
    extract_individual_ratings,
    get_ground_truth,
    get_project_root,
    load_all_combined,
    load_articles,
    load_chat_data,
    load_frontier_data,
    load_enriched_expert,
    load_enriched_student,
    load_expert_filtered_ratings,
    load_expert_ratings,
    load_jsonl,
    load_student_merged_ratings,
    load_student_new_ratings,
    load_student_old_ratings,
    load_thinking_data,
    load_trained_data,
    load_unfiltered_expert_ratings,
    load_unfiltered_student_ratings,
    load_val_data,
    normalize_level,
)

# -- metrics -----------------------------------------------------------------
from .metrics import (
    compute_accuracy,
    compute_balanced_accuracy,
    compute_brier_score,
    compute_cohens_kappa,
    compute_confusion_matrix,
    compute_ece,
    compute_macro_f1,
    compute_per_class_metrics,
    compute_prediction_distribution,
    compute_weighted_f1,
    get_top_predictions,
    logp_to_prob,
)

# -- voting ------------------------------------------------------------------
from .voting import (
    majority_vote,
    monte_carlo_sample_vote,
    norm_level,
    rating_to_level,
    ratings_to_prob_distribution,
)

# -- statistical_tests -------------------------------------------------------
from .statistical_tests import (
    bonferroni_correction,
    bootstrap_ci,
    chi_square_test,
    cochran_q_test,
    fdr_correction,
    fleiss_kappa,
    krippendorff_alpha,
    kruskal_wallis,
    mann_whitney_u,
    mcnemar_test,
    one_sample_ttest,
    permutation_test,
    spearman_correlation,
)

# -- visualization -----------------------------------------------------------
from .visualization import (
    plot_accuracy_curve,
    plot_calibration_diagram,
    plot_confusion_matrix,
    plot_grouped_bar,
    plot_heatmap,
    plot_violin,
    save_figure,
    set_nature_style,
)

__all__ = [
    # constants
    'AI_TO_HUMAN',
    'AI_TO_LEVEL',
    'ALL_COMBINED_PATH',
    'ARTICLES_PATH',
    'CHAT_MODELS',
    'CHAT_MODELS_PRIMARY',
    'CHAT_MODELS_SUPPLEMENTAL',
    'CHAT_PATH',
    'COLORS',
    'DATA_ROOT',
    'ENRICHED_EXPERT_FILTERED_PATH',
    'ENRICHED_EXPERT_PATH',
    'ENRICHED_STUDENT_FILTERED_PATH',
    'ENRICHED_STUDENT_PATH',
    'EXPERT_FILTERED_PATH',
    'EXPERT_PATH',
    'EXPERT_PROFILES_DIR',
    'FIGURES_DIR',
    'FRONTIER_DISPLAY_NAMES',
    'FRONTIER_MODELS',
    'FRONTIER_PATH',
    'HUMAN_TO_AI',
    'LABEL_ORDER',
    'LEVEL_TO_AI',
    'MODEL_DISPLAY_NAMES',
    'N_ARTICLES',
    'N_PER_TIER',
    'OLD_STUDENT_BG_PATH',
    'RESULTS_ROOT',
    'SFT_BASE_MODELS',
    'SFT_MODELS',
    'STATS_DIR',
    'STUDENT_MERGED_PATH',
    'STUDENT_NEW_PATH',
    'STUDENT_OLD_PATH',
    'TABLES_DIR',
    'THINKING_AVG8_PATH',
    'THINKING_LEGACY_PATH',
    'THINKING_PATH',
    'TRAINED_PATH',
    'UNFILTERED_EXPERT_PATH',
    'UNFILTERED_STUDENT_PATH',
    'VAL_PATH',
    # data_loader
    'build_article_index',
    'extract_individual_ratings',
    'get_ground_truth',
    'get_project_root',
    'load_all_combined',
    'load_articles',
    'load_chat_data',
    'load_frontier_data',
    'load_enriched_expert',
    'load_enriched_student',
    'load_expert_filtered_ratings',
    'load_expert_ratings',
    'load_jsonl',
    'load_student_merged_ratings',
    'load_student_new_ratings',
    'load_student_old_ratings',
    'load_thinking_data',
    'load_trained_data',
    'load_unfiltered_expert_ratings',
    'load_unfiltered_student_ratings',
    'load_val_data',
    'normalize_level',
    # metrics
    'compute_accuracy',
    'compute_balanced_accuracy',
    'compute_brier_score',
    'compute_cohens_kappa',
    'compute_confusion_matrix',
    'compute_ece',
    'compute_macro_f1',
    'compute_per_class_metrics',
    'compute_prediction_distribution',
    'compute_weighted_f1',
    'get_top_predictions',
    'logp_to_prob',
    # voting
    'majority_vote',
    'monte_carlo_sample_vote',
    'norm_level',
    'rating_to_level',
    'ratings_to_prob_distribution',
    # statistical_tests
    'bonferroni_correction',
    'bootstrap_ci',
    'chi_square_test',
    'cochran_q_test',
    'fdr_correction',
    'fleiss_kappa',
    'krippendorff_alpha',
    'kruskal_wallis',
    'mann_whitney_u',
    'mcnemar_test',
    'one_sample_ttest',
    'permutation_test',
    'spearman_correlation',
    # visualization
    'plot_accuracy_curve',
    'plot_calibration_diagram',
    'plot_confusion_matrix',
    'plot_grouped_bar',
    'plot_heatmap',
    'plot_violin',
    'save_figure',
    'set_nature_style',
]
