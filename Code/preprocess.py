import re
import csv
import json
import os
import bisect
import traceback
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

TOUCH_MERGE_THRESHOLD_MS = 150

BATCH_TASKS = [
    {
        "INPUT_FILE": r"../Data/dataset/case_1.systrace",
        "OUTPUT_BASE_DIR": r"../Data/preprocessed/test/case_1",
        "TARGET_PIDS": [
            ('11106', 6),
            ('1458',4)
        ],
        "group_key": "animation_1_24",
    }
]

DRAIN_MATCHER_CONFIG = {
    "TEMPLATE_CSV": r"../Data/template/template_results_case_1.csv",
    "DEPTH": 4,
    "SIMILARITY_THRESHOLD": 0.7,
    "MAX_CHILDREN": 100,
}


def replace_symbols_to_space(raw_str: str) -> str:
    if not raw_str:
        return ""
    cleaned = re.sub(r'->|_|:|\||=|\[|\]|\(|\)|-|\{|\}|,|\'|"|!|;', ' ', raw_str)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


class Logcluster:
    def __init__(self, logTemplate="", logIDL=None):
        self.logTemplate = logTemplate
        self.logIDL = logIDL if logIDL is not None else []


class Node:
    def __init__(self, childD=None, depth=0, digitOrtoken=None):
        self.childD = childD if childD is not None else dict()
        self.depth = depth
        self.digitOrtoken = digitOrtoken


class DrainMatcher:
    def __init__(self, depth=4, st=0.4, maxChild=100, rex=None):
        self.depth = depth - 2
        self.st = st
        self.maxChild = maxChild
        self.rex = rex if rex else []
        self.rootNode = Node()
        self.logCluL = []

    def hasNumbers(self, s):
        return any(ch.isdigit() for ch in s)

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, "<*>", line)
        return line

    def addSeqToPrefixTree(self, rn, logClust):
        seqLen = len(logClust.logTemplate)
        if seqLen not in rn.childD:
            firstLayerNode = Node(depth=1, digitOrtoken=seqLen)
            rn.childD[seqLen] = firstLayerNode
        else:
            firstLayerNode = rn.childD[seqLen]

        parentn = firstLayerNode
        currentDepth = 1
        for token in logClust.logTemplate:
            if currentDepth >= self.depth or currentDepth > seqLen:
                if len(parentn.childD) == 0:
                    parentn.childD = [logClust]
                else:
                    parentn.childD.append(logClust)
                break

            if token not in parentn.childD:
                if not self.hasNumbers(token):
                    if "<*>" in parentn.childD:
                        if len(parentn.childD) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD["<*>"]
                    else:
                        if len(parentn.childD) + 1 < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        elif len(parentn.childD) + 1 == self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken="<*>")
                            parentn.childD["<*>"] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD["<*>"]
                else:
                    if "<*>" not in parentn.childD:
                        newNode = Node(depth=currentDepth + 1, digitOrtoken="<*>")
                        parentn.childD["<*>"] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.childD["<*>"]
            else:
                parentn = parentn.childD[token]

            currentDepth += 1

    def seqDist(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        simTokens = 0
        numOfPar = 0
        for t1, t2 in zip(seq1, seq2):
            if t1 == "<*>":
                numOfPar += 1
                simTokens += 1
                continue
            if t1 == t2:
                simTokens += 1
        return float(simTokens) / len(seq1), numOfPar

    def fastMatch(self, logClustL, seq):
        retLogClust = None
        maxSim = -1
        maxNumOfPara = -1
        maxClust = None
        for logClust in logClustL:
            curSim, curNumOfPara = self.seqDist(logClust.logTemplate, seq)
            if curSim > maxSim or (curSim == maxSim and curNumOfPara > maxNumOfPara):
                maxSim = curSim
                maxNumOfPara = curNumOfPara
                maxClust = logClust
        if maxSim >= self.st:
            retLogClust = maxClust
        return retLogClust

    def treeSearch(self, rn, seq):
        seqLen = len(seq)
        if seqLen not in rn.childD:
            return None
        parentn = rn.childD[seqLen]
        currentDepth = 1
        for token in seq:
            if currentDepth >= self.depth or currentDepth > seqLen:
                break
            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif "<*>" in parentn.childD:
                parentn = parentn.childD["<*>"]
            else:
                return None
            currentDepth += 1
        return self.fastMatch(parentn.childD, seq)

    def load_templates_from_csv(self, csv_path):
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 2:
                    continue
                template_tokens = str(row[1]).strip().split()
                if not template_tokens:
                    continue
                logClust = Logcluster(logTemplate=template_tokens, logIDL=[])
                logClust.template_id = str(row[0])
                logClust.template_text = " ".join(template_tokens)
                self.logCluL.append(logClust)
                self.addSeqToPrefixTree(self.rootNode, logClust)

    def match(self, log_line: str) -> dict:
        seq = self.preprocess(log_line).strip().split()
        if not seq:
            return {"matched": False, "template_id": None, "template": None}
        matchCluster = self.treeSearch(self.rootNode, seq)
        if matchCluster is None:
            return {"matched": False, "template_id": None, "template": None}
        return {
            "matched": True,
            "template_id": getattr(matchCluster, "template_id", None),
            "template": getattr(matchCluster, "template_text", " ".join(matchCluster.logTemplate)),
        }


class CallChainGroupedAnalyzer:
    def __init__(
        self,
        target_pids: List[Tuple[str, int]] = None,
        touch_merge_threshold_ms: float = 50.0,
        group_key: str = "group",
        drain_matcher=None,
    ):
        self.target_pids_config = {}
        if target_pids:
            for pid, max_depth in target_pids:
                self.target_pids_config[str(pid)] = max_depth

        self.touch_merge_threshold_sec = touch_merge_threshold_ms / 1000.0
        self.group_key = group_key
        self.thread_stacks = defaultdict(list)
        self.thread_chains = defaultdict(list)
        self.thread_name_map = {}
        self.chain_start_time = {}
        self.skip_counter = defaultdict(int)
        self.touch_events_raw = []
        self.touch_events_merged = []
        self.data_start_time = None
        self.drain_matcher = drain_matcher

    def parse_systrace_line(self, line: str) -> Optional[Dict[str, str]]:
        pattern = r'^\s*(.+?)-(\d+)\s+\(\s*(\d+)\)\s+\[(\d+)\]\s+[\.\w]+\s+([\d\.]+):\s+(.+)$'
        match = re.match(pattern, line)
        if not match:
            return None
        return {
            'task_name': match.group(1).strip(),
            'pid': match.group(2),
            'tgid': match.group(3),
            'timestamp': match.group(5),
            'function': match.group(6).strip()
        }

    def parse_function_field(self, function: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not function.startswith('print:'):
            return None, None, None
        content = function[7:].strip()
        parts = content.split('|', 2)
        if len(parts) < 2:
            return None, None, None
        return parts[0], parts[1], (parts[2] if len(parts) >= 3 else "")

    def process_systrace_file(self, systrace_file: str):
        current_chain_spans = defaultdict(list)

        with open(systrace_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('#') or not line.strip():
                    continue

                parsed = self.parse_systrace_line(line)
                if not parsed:
                    continue

                pid = parsed['pid']
                task_name = parsed['task_name']
                timestamp = float(parsed['timestamp'])
                function = parsed['function']

                if self.data_start_time is None:
                    self.data_start_time = timestamp

                if pid not in self.thread_name_map:
                    self.thread_name_map[pid] = task_name

                event_type, _, func_name = self.parse_function_field(function)
                if not event_type:
                    continue

                if func_name.startswith('H:touchEventDispatch'):
                    if event_type == 'S':
                        self.touch_events_raw.append({
                            'timestamp': timestamp,
                            'task_name': task_name
                        })
                    continue

                if self.target_pids_config and pid not in self.target_pids_config:
                    continue

                if event_type not in ['B', 'E']:
                    continue

                max_stack_depth = self.target_pids_config.get(pid, 0)

                if event_type == 'B':
                    current_stack_depth = len(self.thread_stacks[pid])
                    if max_stack_depth == 0 or current_stack_depth < max_stack_depth:
                        if len(self.thread_stacks[pid]) == 0:
                            self.chain_start_time[pid] = timestamp
                        self.thread_stacks[pid].append({
                            'func_name': func_name,
                            'start_time': timestamp,
                        })
                    else:
                        self.skip_counter[pid] += 1

                elif event_type == 'E':
                    if self.skip_counter[pid] > 0:
                        self.skip_counter[pid] -= 1
                        continue

                    if len(self.thread_stacks[pid]) == 0:
                        continue

                    span_info = self.thread_stacks[pid].pop()
                    f_name = span_info['func_name']
                    start_t = span_info['start_time']
                    duration = timestamp - start_t

                    preprocessed = replace_symbols_to_space(f_name)
                    match_result = self.drain_matcher.match(preprocessed)
                    if match_result["matched"] and match_result["template"]:
                        cleaned = match_result["template"].replace('<*>', '')
                        f_name_cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                    else:
                        f_name_cleaned = preprocessed.strip()

                    if f_name_cleaned:
                        current_chain_spans[pid].append({
                            'slice_name': f_name_cleaned,
                            'duration': duration,
                            'start_time': start_t,
                        })

                    if len(self.thread_stacks[pid]) == 0:
                        if current_chain_spans[pid]:
                            c_start = self.chain_start_time.get(pid, timestamp)
                            self.thread_chains[pid].append({
                                'spans_list': current_chain_spans[pid].copy(),
                                'start_time': c_start,
                                'end_time': timestamp,
                                'duration': timestamp - c_start,
                                'task_name': task_name,
                                'pid': pid
                            })
                            current_chain_spans[pid] = []
                            if pid in self.chain_start_time:
                                del self.chain_start_time[pid]

    def merge_touch_events(self):
        if not self.touch_events_raw:
            return
        events = self.touch_events_raw
        merged = []
        group_start = events[0]
        last_event = events[0]
        for i in range(1, len(events)):
            current_event = events[i]
            if current_event['timestamp'] - last_event['timestamp'] > self.touch_merge_threshold_sec:
                merged.append(group_start)
                group_start = current_event
            last_event = current_event
        merged.append(group_start)
        self.touch_events_merged = [e['timestamp'] for e in merged]

    def calculate_inter_chain_intervals(self):
        for pid, chains in self.thread_chains.items():
            chains.sort(key=lambda x: x['start_time'])
            for i, chain_info in enumerate(chains):
                if i == 0:
                    chain_info['inter_chain_interval'] = None
                else:
                    chain_info['inter_chain_interval'] = chain_info['start_time'] - chains[i - 1]['end_time']

    def group_chains_by_touch_events(self) -> Dict[str, List[Dict]]:
        groups = {}
        for idx, touch_time in enumerate(self.touch_events_merged):
            groups[f"{self.group_key}_touch_{idx + 1}_{touch_time:.6f}"] = []
        groups[f'{self.group_key}_before_first_touch'] = [] if self.touch_events_merged else []
        if not self.touch_events_merged:
            groups['all_chains'] = []

        for pid, chains in self.thread_chains.items():
            for chain_info in chains:
                if not self.touch_events_merged:
                    groups['all_chains'].append(chain_info)
                else:
                    idx = bisect.bisect_right(self.touch_events_merged, chain_info['start_time'])
                    if idx == 0:
                        groups[f'{self.group_key}_before_first_touch'].append(chain_info)
                    else:
                        t_time = self.touch_events_merged[idx - 1]
                        group_name = f"{self.group_key}_touch_{idx}_{t_time:.6f}"
                        groups[group_name].append(chain_info)
        return groups

    def save_groups_to_files(self, groups: Dict[str, List[Dict]], output_base_dir: str):
        os.makedirs(output_base_dir, exist_ok=True)
        for group_name, group_chains in groups.items():
            if not group_chains:
                continue
            group_dir = os.path.join(output_base_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)
            json_data = []
            seen_pids = set()
            for chain_info in group_chains:
                trace_sequence = " ".join(
                    s['slice_name']
                    for s in sorted(chain_info['spans_list'], key=lambda x: x['start_time'])
                )
                pid = chain_info['pid']
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    pre_interval_str = "0.000"
                else:
                    interval = chain_info.get('inter_chain_interval')
                    pre_interval_str = f"{interval * 1000:.3f}" if interval is not None else "N/A"
                json_data.append({
                    "thread_name": chain_info['task_name'],
                    "thread_id": pid,
                    "start_time": f"{chain_info['start_time']:.6f}",
                    "duration": f"{chain_info['duration'] * 1000:.3f}",
                    "pre_interval": pre_interval_str,
                    "trace_sequence": trace_sequence,
                    "label": "0",
                    "window_name": group_name
                })
            output_json_path = os.path.join(group_dir, f"{group_name}_chains.json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)


def validate_task_config(task: dict, index: int):
    required_keys = ["INPUT_FILE", "OUTPUT_BASE_DIR", "TARGET_PIDS", "group_key"]
    for key in required_keys:
        if key not in task:
            raise ValueError(f"Task {index} missing required field: {key}")
    if not os.path.isfile(task["INPUT_FILE"]):
        raise FileNotFoundError(f"Task {index} input file not found: {task['INPUT_FILE']}")
    if not isinstance(task["TARGET_PIDS"], list):
        raise TypeError(f"Task {index} TARGET_PIDS must be a list")
    if not isinstance(task["group_key"], str) or not task["group_key"].strip():
        raise ValueError(f"Task {index} group_key must not be empty")


def run_single_task(task: dict, drain_matcher):
    analyzer = CallChainGroupedAnalyzer(
        target_pids=task["TARGET_PIDS"],
        touch_merge_threshold_ms=TOUCH_MERGE_THRESHOLD_MS,
        group_key=task["group_key"],
        drain_matcher=drain_matcher,
    )
    analyzer.process_systrace_file(task["INPUT_FILE"])
    analyzer.merge_touch_events()
    analyzer.calculate_inter_chain_intervals()
    groups = analyzer.group_chains_by_touch_events()
    analyzer.save_groups_to_files(groups, task["OUTPUT_BASE_DIR"])


def main():
    matcher = DrainMatcher(
        depth=DRAIN_MATCHER_CONFIG["DEPTH"],
        st=DRAIN_MATCHER_CONFIG["SIMILARITY_THRESHOLD"],
        maxChild=DRAIN_MATCHER_CONFIG["MAX_CHILDREN"],
    )
    matcher.load_templates_from_csv(DRAIN_MATCHER_CONFIG["TEMPLATE_CSV"])

    for idx, task in enumerate(BATCH_TASKS, 1):
        try:
            validate_task_config(task, idx)
            run_single_task(task, matcher)
        except Exception as e:
            print(f"Task {idx} failed: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()