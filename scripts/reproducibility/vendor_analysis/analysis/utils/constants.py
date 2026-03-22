"""Constants and mappings for the analysis pipeline.

This module defines all shared constants, label mappings, color palettes,
model display names, and file paths used across the research analysis project.
The project analyzes whether SFT-trained LLMs can evaluate research idea quality,
using 120 articles rated by AI models and human raters (experts and students).
"""

import os
from typing import Dict, List

# ---------------------------------------------------------------------------
# Label order (AI model output labels, from best to worst)
# ---------------------------------------------------------------------------
LABEL_ORDER: List[str] = ['exceptional', 'strong', 'fair', 'limited']

# ---------------------------------------------------------------------------
# Human survey rating to AI label mapping
# ---------------------------------------------------------------------------
HUMAN_TO_AI: Dict[str, str] = {
    'Top': 'exceptional',
    'Top-': 'strong',
    'Good': 'fair',
    'Fair': 'limited',
}
AI_TO_HUMAN: Dict[str, str] = {v: k for k, v in HUMAN_TO_AI.items()}

# ---------------------------------------------------------------------------
# Level field to AI label mapping (used in articles JSONL)
# ---------------------------------------------------------------------------
LEVEL_TO_AI: Dict[str, str] = {
    'top': 'exceptional',
    'top-': 'strong',
    'good': 'fair',
    'fair': 'limited',
}
AI_TO_LEVEL: Dict[str, str] = {v: k for k, v in LEVEL_TO_AI.items()}

# ---------------------------------------------------------------------------
# Dataset dimensions
# ---------------------------------------------------------------------------
N_PER_TIER: int = 30
N_ARTICLES: int = 120

# ---------------------------------------------------------------------------
# Nature/Science color palette
# ---------------------------------------------------------------------------
COLORS: Dict[str, str] = {
    'exceptional': '#E64B35',  # Red
    'strong': '#4DBBD5',       # Cyan
    'fair': '#00A087',         # Teal
    'limited': '#3C5488',      # Blue
    'ai': '#E64B35',           # Red for AI
    'expert': '#4DBBD5',       # Cyan for experts
    'student': '#00A087',      # Teal for students
    'sft': '#F39B7F',          # Salmon for SFT
    'baseline': '#8491B4',     # Gray-blue for baseline
    'old_student': '#7E6148',   # Brown for old students
    'new_student': '#B09C85',   # Tan for new students
    'filtered': '#00A087',      # Teal for filtered
    'unfiltered': '#8491B4',    # Gray-blue for unfiltered
    'match_high': '#E64B35',    # Red for high match
    'match_low': '#3C5488',     # Blue for low match
}

# ---------------------------------------------------------------------------
# Model display names for plots
# ---------------------------------------------------------------------------
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    # Flagship (thinking) models
    'gpt-5.2': 'GPT-5.2',
    'claude-sonnet-4-20250514': 'Claude Sonnet 4',
    'claude-3-7-sonnet-20250219': 'Claude 3.7 Sonnet',
    'kimi-k2': 'Kimi K2',
    'seed1.5-thinking': 'Seed 1.5',
    'gemini-2.5-pro-preview-05-06': 'Gemini 2.5 Pro',
    'gemini-2.5-flash-preview-04-17': 'Gemini 2.5 Flash',
    'grok-3-mini': 'Grok 3 Mini',
    'glm-5-plus-0111': 'GLM-5 Plus',
    'MiniMax-M1-80k': 'MiniMax M1',
    'deepseek-r1': 'DeepSeek R1',
    # Canonical frontier keys in 120_frontier.jsonl
    'openai/gpt-5.2-high': 'GPT-5.2 High',
    'qwen/qwen3.5-plus-02-15': 'Qwen 3.5 Plus',
    'doubao-seed-2-0-pro-260215': 'Seed 2.0',
    'google/gemini-2.5-pro': 'Gemini 2.5 Pro',
    'google/gemini-3.1-pro-preview': 'Gemini 3.1 Pro',
    # Chat models (with logp)
    'kimi-k2-0905-preview': 'Kimi K2 Chat',
    'deepseek-chat': 'DeepSeek Chat',
    # SFT models
    'gpt-4.1-ob': 'SFT GPT-4.1-ob',
    'gpt-4.1-nano-ob': 'SFT GPT-4.1-nano-ob',
    'qwen3-30b-ob': 'SFT Qwen3-30B-ob',
    'qwen3-4b-ob': 'SFT Qwen3-4B-ob',
    # Ensembles and special
    'best_2_model_combo': 'SFT 2-Model Ensemble',
    'human_voting': 'Human Voting',
}

# Canonical 11-model conservative frontier cohort used across analysis scripts.
# Gemini 3.1 Pro is retained with contamination-risk caveats in manuscript text.
FRONTIER_MODELS: List[str] = [
    'z-ai/glm-5',
    'moonshotai/kimi-k2.5',
    'google/gemini-2.5-pro',
    'google/gemini-3.1-pro-preview',
    'anthropic/claude-opus-4.6',
    'openai/gpt-5.2-high',
    'x-ai/grok-4.1-fast',
    'minimax/minimax-m2.5',
    'deepseek/deepseek-v3.2-speciale',
    'qwen/qwen3.5-plus-02-15',
    'doubao-seed-2-0-pro-260215',
]

FRONTIER_DISPLAY_NAMES: Dict[str, str] = {
    'z-ai/glm-5': 'GLM-5',
    'moonshotai/kimi-k2.5': 'Kimi K2.5',
    'google/gemini-2.5-pro': 'Gemini 2.5 Pro',
    'google/gemini-3.1-pro-preview': 'Gemini 3.1 Pro',
    'anthropic/claude-opus-4.6': 'Claude Opus 4.6',
    'openai/gpt-5.2-high': 'GPT-5.2 High',
    'x-ai/grok-4.1-fast': 'Grok 4.1 Fast',
    'minimax/minimax-m2.5': 'MiniMax M2.5',
    'deepseek/deepseek-v3.2-speciale': 'DeepSeek V3.2',
    'qwen/qwen3.5-plus-02-15': 'Qwen 3.5 Plus',
    'doubao-seed-2-0-pro-260215': 'Seed 2.0',
}

# Chat models tracked in analyses.
# PRIMARY models are used in main manuscript tables/figures.
CHAT_MODELS_PRIMARY: List[str] = [
    'gpt-5.2',
    'kimi-k2-0905-preview',
    'deepseek-chat',
]

# Supplemental base models currently present in 120_chat.jsonl.
# These are not included in main-chat comparison tables by default.
CHAT_MODELS_SUPPLEMENTAL: List[str] = [
    'gpt-4.1',
    'gpt-4.1-nano',
    'qwen3-4b',
]

# Backward-compatible alias used by existing scripts.
CHAT_MODELS: List[str] = CHAT_MODELS_PRIMARY

# Public SFT model keys (from sft_predictions.jsonl, excluding ensembles and human_voting)
SFT_MODELS: List[str] = ['gpt-4.1-ob', 'gpt-4.1-nano-ob', 'qwen3-30b-ob', 'qwen3-4b-ob']

# SFT base model mapping
SFT_BASE_MODELS: Dict[str, str] = {
    'gpt-4.1-ob': 'GPT-4.1-ob',
    'gpt-4.1-nano-ob': 'GPT-4.1-nano-ob',
    'qwen3-30b-ob': 'Qwen3-30B-ob',
    'qwen3-4b-ob': 'Qwen3-4B-ob',
}

# ---------------------------------------------------------------------------
# Data paths (relative to project root)
# ---------------------------------------------------------------------------
DATA_ROOT: str = 'data'
ARTICLES_PATH: str = f'{DATA_ROOT}/benchmark/benchmark_articles_120.jsonl'
TRAINED_PATH: str = f'{DATA_ROOT}/predictions/sft_predictions.jsonl'
CHAT_PATH: str = f'{DATA_ROOT}/predictions/chat_predictions.jsonl'
FRONTIER_PATH: str = f'{DATA_ROOT}/predictions/frontier_10models_8runs.jsonl'
THINKING_AVG8_PATH: str = FRONTIER_PATH
THINKING_LEGACY_PATH: str = FRONTIER_PATH
GEMINI31_STANDALONE_PATH: str = f'{DATA_ROOT}/predictions/gemini_3_1_pro_standalone.jsonl'
PROMPT_EXPERT_PATH: str = f'{DATA_ROOT}/predictions/prompt_variants/expert_prompt_predictions.jsonl'
# Backward-compatible alias for "thinking" analyses: now points to canonical frontier file.
THINKING_PATH: str = FRONTIER_PATH
VAL_PATH: str = TRAINED_PATH  # backward compat alias (120_val.jsonl no longer exists)

# Expert paths: unfiltered is PRIMARY, filtered is for sensitivity analysis
EXPERT_PATH: str = f'{DATA_ROOT}/human_ratings/reproducibility/expert_reproducibility.jsonl'
EXPERT_FILTERED_PATH: str = f'{DATA_ROOT}/human_ratings/reproducibility/expert_reproducibility_filtered.jsonl'

# Junior paths: final manuscript analyses use the filtered combined junior
# panel (old + new pooled). Legacy aliases are retained only so older
# scripts can still import them without ambiguity in the underlying file.
STUDENT_FILTERED_PATH: str = f'{DATA_ROOT}/human_ratings/reproducibility/student_reproducibility_filtered.jsonl'
STUDENT_UNFILTERED_PATH: str = f'{DATA_ROOT}/human_ratings/reproducibility/student_reproducibility.jsonl'
STUDENT_NEW_PATH: str = STUDENT_FILTERED_PATH   # deprecated alias; not new-only
STUDENT_MERGED_PATH: str = STUDENT_FILTERED_PATH  # deprecated alias
STUDENT_OLD_PATH: str = STUDENT_UNFILTERED_PATH   # deprecated alias; not old-only
ALL_COMBINED_PATH: str = STUDENT_UNFILTERED_PATH

# Unfiltered archive paths (EXPERT_PATH already points to unfiltered)
UNFILTERED_EXPERT_PATH: str = EXPERT_PATH
UNFILTERED_STUDENT_PATH: str = STUDENT_UNFILTERED_PATH

# Enriched dataset paths
ENRICHED_EXPERT_PATH: str = EXPERT_PATH
ENRICHED_EXPERT_FILTERED_PATH: str = EXPERT_FILTERED_PATH
ENRICHED_STUDENT_PATH: str = STUDENT_UNFILTERED_PATH
ENRICHED_STUDENT_FILTERED_PATH: str = STUDENT_FILTERED_PATH

# Background data paths
OLD_STUDENT_BG_PATH: str = STUDENT_UNFILTERED_PATH
EXPERT_PROFILES_DIR: str = f'{DATA_ROOT}/human_ratings/reproducibility'
# ---------------------------------------------------------------------------
# Results paths
# ---------------------------------------------------------------------------
# Optional override:
#   RESULTS_ROOT="results_reanalysis_2026-02-27" python3.14 analysis/...
_RESULTS_ROOT_ENV = os.environ.get('RESULTS_ROOT', 'results').strip()
RESULTS_ROOT: str = _RESULTS_ROOT_ENV or 'results'
FIGURES_DIR: str = f'{RESULTS_ROOT}/figures'
TABLES_DIR: str = f'{RESULTS_ROOT}/tables'
STATS_DIR: str = f'{RESULTS_ROOT}/statistics'
