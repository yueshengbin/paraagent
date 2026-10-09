
import json
import re

def normalize_whitespace(text):
    return re.sub('\\s+', ' ', str(text or '').strip())

def lcs_length(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0
    prev = [0] * (len(b_tokens) + 1)
    for a in a_tokens:
        curr = [0]
        for j, b in enumerate(b_tokens, start=1):
            if a == b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]

def rouge_l_f1(reference, prediction):
    ref = normalize_whitespace(reference).split()
    pred = normalize_whitespace(prediction).split()
    if not ref or not pred:
        return 0.0
    lcs = lcs_length(ref, pred)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def _safe_json_loads(obj):
    if isinstance(obj, dict):
        return obj
    if not isinstance(obj, str):
        return {}
    try:
        return json.loads(obj)
    except Exception:
        return {}

def extract_pred_api_calls(answer_generation):
    pred_api_calls = []
    for item in answer_generation.get('called_apis', []) or []:
        tool_name = item.get('tool', '')
        if tool_name in ('Finish', 'search_tool', 'ToolSearcher'):
            continue
        param_dict = item.get('input', item.get('arguments', {}))
        result = item.get('output', item.get('response', {}))
        if isinstance(result, dict) and list(result.keys()) == ['result']:
            result = result['result']
        pred_api_calls.append({'api_name': tool_name, 'param_dict': _safe_json_loads(param_dict), 'result': _safe_json_loads(result)})
    return pred_api_calls

def extract_gt_level3_api_calls(query):
    gt_api_calls = []
    for api_call in query.get('apis', []) or []:
        if api_call.get('api_name') == 'ToolSearcher':
            continue
        raw_output = api_call.get('output', {}) or {}
        if isinstance(raw_output, dict) and 'output' in raw_output:
            result = raw_output['output']
        else:
            result = raw_output
        gt_api_calls.append({'api_name': api_call.get('api_name', ''), 'param_dict': api_call.get('input', api_call.get('param_dict', {})) or {}, 'result': result or {}})
    return gt_api_calls

def _params_match(pred, gt):

    return isinstance(pred, dict) and isinstance(gt, dict) and (pred == gt)

def evaluate_level3_sample(query, answer_generation):
    pred_api_calls = extract_pred_api_calls(answer_generation)
    gt_api_calls = extract_gt_level3_api_calls(query)
    used_pred_indices = set()
    item_gt_api_calls = len(gt_api_calls)
    item_correct_api_calls = 0
    error_statistics = {'NO_API_CALL': 0, 'API_NAME_MISMATCH': 0, 'HAS_EXCEPTION': 0, 'INPUT_MISMATCH': 0, 'OUTPUT_MISMATCH': 0, 'INVALID_INPUT_PARAMETER': 0, 'KEY_ERROR': 0, 'FAILED_PARSE_API_CALL': 0, 'MISS_INPUT_ARGUMENT': 0}
    for gt_call in gt_api_calls:
        gt_name = gt_call['api_name']
        gt_input = gt_call['param_dict']
        candidate_indices = [idx for idx, pred_call in enumerate(pred_api_calls) if idx not in used_pred_indices and pred_call['api_name'] == gt_name]
        matched = False
        if candidate_indices:
            for idx in candidate_indices:
                pred_args = pred_api_calls[idx]['param_dict']
                if _params_match(pred_args, gt_input):
                    item_correct_api_calls += 1
                    used_pred_indices.add(idx)
                    matched = True
                    break
            if not matched:
                error_statistics['INPUT_MISMATCH'] += 1
        elif len(pred_api_calls) == 0:
            error_statistics['NO_API_CALL'] += 1
        else:
            error_statistics['API_NAME_MISMATCH'] += 1
    final_answer = answer_generation.get('final_answer', '')
    gt_response = query.get('response', '')
    rouge_score = rouge_l_f1(gt_response, final_answer) if gt_response else 0.0
    success = 1.0 if item_gt_api_calls > 0 and item_correct_api_calls == item_gt_api_calls else 0.0
    first_api = pred_api_calls[0]['api_name'] if pred_api_calls else ''
    return {'query': query.get('requirement', query.get('query', '')), 'expected_api': None, 'first_called_api': first_api, 'api_accuracy': item_correct_api_calls / item_gt_api_calls if item_gt_api_calls > 0 else 0.0, 'gt_api_calls': item_gt_api_calls, 'correct_api_calls': item_correct_api_calls, 'success': success, 'rouge_l': rouge_score, 'inference_time': answer_generation.get('inference_time', None), 'is_search_sample': True, 'error_statistics': error_statistics}
