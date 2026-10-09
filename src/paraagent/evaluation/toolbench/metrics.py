import os
import json
import csv
import re
import sys

def normalize_text(text: str) -> str:
    return re.sub('[^a-z0-9]+', '', str(text).lower())
MANUAL_API_ALIAS_MAP = {normalize_text('Latest Anime API : Get Latest  Anime'): 'Latest Anime API : Search Anime', normalize_text('Bulk Whatsapp Validator : Validate WA number'): 'Bulk Whatsapp Validator : Validate whatsapp number', normalize_text('LocationIQ : Nearest'): 'LocationIQ : listOfAllNearbyPoIsExceptGyms', normalize_text('LocationIQ : Matrix'): 'LocationIQ : fixedSourceDestination', normalize_text('Wayfair : products/detail'): 'Wayfair : products/detail (Deprecated)', normalize_text('Yahoo Finance : index'): 'Yahoo Finance : Trend', normalize_text('YTConvert : Url Download'): 'YTConvert : /download/mp4', normalize_text('JobSearch : /api/v1/Jobs/Search'): 'JobSearch : /api/v2/Jobs/Search', normalize_text('LocationIQ : reverse'): 'LocationIQ : generalUsage', normalize_text('INDIAN FUEL : /fuel/data/{city}'): 'INDIAN FUEL : City wise data', normalize_text('Airbnb Search : Search Property By GEO'): 'Airbnb Search : properties/search-by-geo', normalize_text('Shazam API : charts/get-top-songs-in-city'): 'Shazam api : Top music in City'}

def canonicalize_gold_relevant_api(tool_name: str, api_name: str) -> str:
    return f'{tool_name} : {api_name}'

def canonicalize_model_tool_name(tool_name: str) -> str:
    tool_name = str(tool_name).strip()
    if not tool_name or tool_name == 'Finish':
        return tool_name
    if '_for_' in tool_name:
        api_name, parent_tool = tool_name.split('_for_', 1)
        return f"{parent_tool.replace('_', ' ')} : {api_name.replace('_', ' ')}"
    return tool_name.replace('_', ' ')

def resolve_manual_alias(name: str) -> str:
    return MANUAL_API_ALIAS_MAP.get(normalize_text(name), name)

def extract_called_tools(answer_details) -> list[str]:
    called_tools = []

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        if node.get('role') == 'tool':
            message = node.get('message', {})
            if isinstance(message, dict):
                tool_name = message.get('name')
                if tool_name:
                    called_tools.append(tool_name)
        for child in node.get('next', []) or []:
            walk(child)
    walk(answer_details)
    return called_tools

def load_relevant_api_map(relevant_api_dir: str, test_set: str) -> dict[str, list[list[str]]]:
    file_path = os.path.join(relevant_api_dir, f'{test_set}.json')
    data = json.load(open(file_path, 'r'))
    return {str(item['query_id']): item.get('relevant APIs', []) for item in data}

def load_toolbench_name_api_set(toolbench_name_api_path: str) -> set[str]:
    csv.field_size_limit(sys.maxsize)
    name_set = set()
    with open(toolbench_name_api_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            name = row.get('name')
            if name:
                name_set.add(normalize_text(name))
    return name_set

def compute_tool_hit_metrics(example: dict, gold_relevant_apis: list[list[str]], corpus_name_set: set[str]) -> dict:
    called_tool_names = extract_called_tools(example['answer']['answer_details'])
    called_tool_names = [name for name in called_tool_names if name and name != 'Finish']
    called_canonical = []
    called_seen = set()
    for name in called_tool_names:
        canonical_name = resolve_manual_alias(canonicalize_model_tool_name(name))
        if canonical_name not in called_seen:
            called_seen.add(canonical_name)
            called_canonical.append(canonical_name)
    gold_canonical = []
    gold_seen = set()
    gold_in_corpus = 0
    for pair in gold_relevant_apis:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        canonical_name = resolve_manual_alias(canonicalize_gold_relevant_api(pair[0], pair[1]))
        if canonical_name not in gold_seen:
            gold_seen.add(canonical_name)
            gold_canonical.append(canonical_name)
        if normalize_text(canonical_name) in corpus_name_set:
            gold_in_corpus += 1
    called_norm = {normalize_text(name) for name in called_canonical}
    gold_norm = {normalize_text(name) for name in gold_canonical}
    hit_count = len(called_norm & gold_norm)
    gold_count = len(gold_norm)
    return {'model_called_tools': called_canonical, 'gold_relevant_apis': gold_canonical, 'tool_hit_count': hit_count, 'tool_hit_ratio': hit_count / gold_count if gold_count else 0.0, 'gold_relevant_api_count': gold_count, 'gold_relevant_api_in_corpus_count': gold_in_corpus, 'gold_relevant_api_coverage_in_corpus': gold_in_corpus / gold_count if gold_count else 0.0}

def has_nonempty_final_answer(example: dict) -> bool:
    final_answer = example.get('answer', {}).get('final_answer', '')
    if isinstance(final_answer, str):
        return bool(final_answer.strip())
    return bool(final_answer)
