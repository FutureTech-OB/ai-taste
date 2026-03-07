"""Background data loading and encoding utilities.

Provides loaders for old/new student background data and expert profiles,
plus field parsing and encoding helpers for predictor analysis.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .constants import DATA_ROOT
from .data_loader import get_project_root, load_jsonl


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OLD_STUDENT_BG_PATH = f'{DATA_ROOT}/human_ratings/old_students_background.jsonl'
EXPERT_PROFILES_PATH = f'{DATA_ROOT}/expert_profiles/profiles.jsonl'
NEW_STUDENT_CSV = f'{DATA_ROOT}/raw_csv/AI for science--- student_new.csv'

UNFILTERED_EXPERT_PATH = f'{DATA_ROOT}/human_ratings/archive/survey_all_expert.jsonl'
UNFILTERED_STUDENT_PATH = f'{DATA_ROOT}/human_ratings/archive/survey_all_student.jsonl'

ENRICHED_EXPERT_PATH = f'{DATA_ROOT}/human_ratings/expert_enriched.jsonl'
ENRICHED_EXPERT_FILTERED_PATH = f'{DATA_ROOT}/human_ratings/expert_enriched_filtered.jsonl'
ENRICHED_STUDENT_PATH = f'{DATA_ROOT}/human_ratings/student_enriched.jsonl'
ENRICHED_STUDENT_FILTERED_PATH = f'{DATA_ROOT}/human_ratings/student_enriched_filtered.jsonl'


# ---------------------------------------------------------------------------
# Background data loaders
# ---------------------------------------------------------------------------

def load_old_student_backgrounds() -> Dict[str, Dict[str, Any]]:
    """Load old student background data, keyed by rater_name.

    Returns:
        Dict mapping rater_name to background dict.
    """
    records = load_jsonl(OLD_STUDENT_BG_PATH)
    return {rec['rater_name']: rec.get('background', {}) for rec in records}


def load_new_student_backgrounds_from_csv() -> Dict[str, Dict[str, Any]]:
    """Load new student background data from Qualtrics CSV.

    Parses the CSV with GBK encoding, extracts background fields
    (columns 18-28 in human-readable row), and returns a dict keyed
    by respondent name.

    Returns:
        Dict mapping respondent name to background dict.
    """
    root = get_project_root()
    csv_path = root / NEW_STUDENT_CSV

    backgrounds: Dict[str, Dict[str, Any]] = {}

    with open(csv_path, encoding='gbk') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 4:
        return backgrounds

    headers_readable = rows[1]
    data_rows = rows[3:]

    # Find column indices by human-readable header (row 1)
    col_map: Dict[str, int] = {}
    for i, label in enumerate(headers_readable):
        ll = label.lower().strip()
        if ll == 'name':
            col_map['name'] = i
        elif ll == 'gender':
            col_map['gender'] = i
        elif 'university' in ll or 'institution' in ll:
            col_map['school'] = i
        elif 'major' in ll or 'research direction' in ll:
            col_map['major_research'] = i
        elif 'current status' in ll:
            col_map['current_identity'] = i
        elif 'number of published' in ll and 'first' not in ll:
            col_map['published_papers'] = i
        elif 'first author' in ll:
            col_map['first_author_papers'] = i
        elif 'representative english' in ll:
            col_map['representative_english_pubs'] = i
        elif 'peer review' in ll:
            col_map['review_experience'] = i
        elif 'research fields' in ll or 'familiar research' in ll:
            col_map['research_areas'] = i
        elif 'generative ai' in ll or 'familiarity with' in ll:
            col_map['ai_familiarity'] = i
        elif ll == 'response id':
            col_map['response_id'] = i

    for row in data_rows:
        if len(row) < 20:
            continue

        name = row[col_map.get('name', 18)].strip() if col_map.get('name') else ''
        if not name:
            continue

        bg: Dict[str, Any] = {'name': name}
        for field, idx in col_map.items():
            if field in ('name', 'response_id'):
                continue
            val = row[idx].strip() if idx < len(row) else ''
            if field == 'research_areas' and val:
                bg[field] = [x.strip() for x in val.split(',') if x.strip()]
            else:
                bg[field] = val

        response_id = row[col_map.get('response_id', 8)].strip() if col_map.get('response_id') else ''
        backgrounds[name] = bg
        if response_id:
            backgrounds[f'__rid__{response_id}'] = bg

    return backgrounds


def load_expert_profiles() -> Dict[str, Dict[str, Any]]:
    """Load expert profiles from profiles.jsonl.

    Returns:
        Dict mapping expert name to profile dict.
    """
    root = get_project_root()
    path = root / EXPERT_PROFILES_PATH
    if not path.exists():
        return {}
    records = load_jsonl(EXPERT_PROFILES_PATH)
    return {rec.get('name', rec.get('survey_name', '')): rec for rec in records}


def load_enriched_data(path_constant: str) -> List[Dict[str, Any]]:
    """Load an enriched JSONL dataset.

    Parameters:
        path_constant: Relative path from project root.

    Returns:
        List of enriched article records.
    """
    return load_jsonl(path_constant)


# ---------------------------------------------------------------------------
# Field parsing helpers
# ---------------------------------------------------------------------------

def parse_confidence(conf_str: Any) -> Optional[int]:
    """Extract integer confidence from string like '4 - Confident'.

    Returns:
        Integer 1-5, or None if parsing fails.
    """
    if conf_str is None:
        return None
    try:
        return int(str(conf_str).strip()[0])
    except (ValueError, IndexError, TypeError):
        return None


def parse_familiarity(fam_str: Any) -> Optional[int]:
    """Extract integer familiarity from string like '3 - Neutral'.

    Returns:
        Integer 1-5, or None if parsing fails.
    """
    if fam_str is None:
        return None
    try:
        return int(str(fam_str).strip()[0])
    except (ValueError, IndexError, TypeError):
        return None


def parse_read_before(val: Any) -> Optional[bool]:
    """Parse read-before field to boolean.

    Returns:
        True if 'yes', False if 'no', None otherwise.
    """
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ('yes', 'true', '1'):
        return True
    if s in ('no', 'false', '0'):
        return False
    return None


def parse_duration(val: Any) -> Optional[float]:
    """Parse duration field to float seconds.

    Returns:
        Duration in seconds, or None.
    """
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_phd_year(identity_str: Any) -> Optional[int]:
    """Extract PhD year number from identity string.

    Handles Chinese like '博士一年级' -> 1, '博士三年级' -> 3,
    and English like 'PhD Year 2' -> 2, 'Postdoc' -> 7.

    Returns:
        Integer year, or None.
    """
    if identity_str is None:
        return None
    s = str(identity_str).strip()

    # Chinese PhD year
    cn_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6}
    for ch, num in cn_map.items():
        if f'博士{ch}年级' in s:
            return num

    # Postdoc
    if '博士后' in s.lower() or 'postdoc' in s.lower() or '博后' in s:
        return 7

    # English PhD year
    m = re.search(r'(?:PhD|博士)\s*(?:Year\s*)?(\d+)', s, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Masters
    if '硕士' in s or 'master' in s.lower():
        return 0

    return None


def parse_publications(pub_str: Any) -> Optional[int]:
    """Parse publication count from string like '3-5篇' or '0篇'.

    Returns the midpoint for ranges, exact value for single numbers.

    Returns:
        Estimated publication count, or None.
    """
    if pub_str is None:
        return None
    s = str(pub_str).strip()

    # Exact number
    m = re.match(r'^(\d+)\s*篇?$', s)
    if m:
        return int(m.group(1))

    # Range
    m = re.match(r'^(\d+)\s*[-–]\s*(\d+)', s)
    if m:
        return (int(m.group(1)) + int(m.group(2))) // 2

    # "6+" or "6篇以上"
    m = re.match(r'^(\d+)\s*[+篇]', s)
    if m:
        return int(m.group(1))

    return None


def parse_review_experience(rev_str: Any) -> Optional[str]:
    """Categorize review experience.

    Returns:
        'none', 'some' (1-5), 'experienced' (6+), or None.
    """
    if rev_str is None:
        return None
    s = str(rev_str).strip().lower()

    if s in ('无', 'none', 'no', '0'):
        return 'none'
    if '1-5' in s or '有' in s:
        return 'some'
    if '6' in s or '10' in s or 'many' in s:
        return 'experienced'

    return 'some' if s else None


def encode_ai_familiarity(fam_str: Any) -> Optional[str]:
    """Encode AI familiarity into categories.

    Returns:
        'occasional', 'frequent', or 'deep', or None.
    """
    if fam_str is None:
        return None
    s = str(fam_str).strip()

    if '深度' in s or 'deep' in s.lower() or '科研' in s:
        return 'deep'
    if '经常' in s or 'frequent' in s.lower() or '日常' in s:
        return 'frequent'
    if '偶尔' in s or 'occasional' in s.lower():
        return 'occasional'
    if '从未' in s or 'never' in s.lower():
        return 'never'

    return None


def phd_year_to_group(year: Optional[int]) -> Optional[str]:
    """Group PhD year into categories for analysis.

    Returns:
        'yr1-2', 'yr3-4', 'yr5+', or None.
    """
    if year is None:
        return None
    if year <= 0:
        return 'masters'
    if year <= 2:
        return 'yr1-2'
    if year <= 4:
        return 'yr3-4'
    return 'yr5+'


def publications_to_group(count: Optional[int]) -> Optional[str]:
    """Group publication count into categories.

    Returns:
        '0', '1-2', '3-5', '6+', or None.
    """
    if count is None:
        return None
    if count == 0:
        return '0'
    if count <= 2:
        return '1-2'
    if count <= 5:
        return '3-5'
    return '6+'
