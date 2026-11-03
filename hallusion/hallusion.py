from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any

_YES_RE = re.compile(r"^\s*yes\b", re.IGNORECASE)
_NO_RE  = re.compile(r"^\s*no\b",  re.IGNORECASE)


def parse_first_token(text: str) -> int:
    if text is None:
        return 2
    text = text.strip()
    if _YES_RE.match(text):
        return 1
    if _NO_RE.match(text):
        return 0
    return 2


def is_correct(parsed: int, gt_answer: int, is_vs_no_visual: bool) -> int:
    if parsed == gt_answer:
        return 1
    if parsed == 2:
        if is_vs_no_visual:
            return 1
        return 0
    return 0 


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    for key in ("questions", "data", "samples"):
        if isinstance(obj.get(key), list):
            return obj[key]
    raise ValueError(f"Cannot parse question file: {path}")


def merge_gt(answers: list[dict], questions: list[dict]) -> list[dict]:
    lookup: dict[tuple, dict] = {}
    for q in questions:
        key = (
            str(q.get("set_id", "")),
            str(q.get("figure_id", "")),
            str(q.get("question_id", "")),
        )
        lookup[key] = q

    for a in answers:
        key = (
            str(a.get("set_id", "")),
            str(a.get("figure_id", "")),
            str(a.get("question_id", "")),
        )
        q = lookup.get(key, {})
        for field in ("category", "visual_input", "gt_answer", "filename",
                      "set_id", "figure_id", "question_id"):
            if field not in a and field in q:
                a[field] = q[field]
    return answers

def _is_vs_no_visual(sample: dict) -> bool:
    return (
        str(sample.get("category", "")).upper() == "VS"
        and str(sample.get("visual_input", "1")) == "0"
    )

def _difficulty(sample: dict) -> str:
    cat = str(sample.get("category", "")).upper()
    if cat == "VS":
        return "easy" if str(sample.get("visual_input", "1")) == "0" else "hard"
    else:  # VD
        fig_id = int(sample.get("figure_id", 0))
        return "easy" if fig_id == 0 else "hard"


def evaluate(samples: list[dict]) -> dict[str, Any]:
    for s in samples:
        raw_text = s.get("text", s.get("answer", ""))
        s["_parsed"]    = parse_first_token(str(raw_text))
        s["_gt"]        = int(s.get("gt_answer", -1))
        s["_no_visual"] = _is_vs_no_visual(s)
        s["_correct"]   = is_correct(s["_parsed"], s["_gt"], s["_no_visual"])
        s["_difficulty"]= _difficulty(s)
        s["_category"]  = str(s.get("category", "")).upper()

    aAcc_all  = _mean_correct(samples)
    aAcc_vd   = _mean_correct([s for s in samples if s["_category"] == "VD"])
    aAcc_vs   = _mean_correct([s for s in samples if s["_category"] == "VS"])
    aAcc_easy = _mean_correct([s for s in samples if s["_difficulty"] == "easy"])
    aAcc_hard = _mean_correct([s for s in samples if s["_difficulty"] == "hard"])

    fig_groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in samples:
        if s["_category"] == "VS" and str(s.get("figure_id", "")) == "0":
            continue
        key = (
            s["_category"],
            str(s.get("subcategory", "")),
            str(s.get("set_id", "")),
            str(s.get("figure_id", "")),
        )
        fig_groups[key].append(s)

    fAcc_all, fAcc_vd, fAcc_vs, fAcc_easy, fAcc_hard = _figure_acc(fig_groups)

    q_groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in samples:
        key = (
            s["_category"],
            str(s.get("subcategory", "")),
            str(s.get("set_id", "")),
            str(s.get("question_id", "")),
        )
        q_groups[key].append(s)

    qAcc_all, qAcc_vd, qAcc_vs, qAcc_easy, qAcc_hard = _question_pair_acc(q_groups)

    dy, rfp = _bias_tests(samples)
    dy_vd,  rfp_vd  = _bias_tests([s for s in samples if s["_category"] == "VD"])
    dy_vs,  rfp_vs  = _bias_tests([s for s in samples if s["_category"] == "VS"])

    parse_errors = _parse_error_stats(samples)

    failure_counts = _diagnostic_test(samples)

    return dict(
        n_total=len(samples),
        aAcc=dict(all=aAcc_all, VD=aAcc_vd, VS=aAcc_vs, easy=aAcc_easy, hard=aAcc_hard),
        fAcc=dict(all=fAcc_all, VD=fAcc_vd, VS=fAcc_vs, easy=fAcc_easy, hard=fAcc_hard),
        qAcc=dict(all=qAcc_all, VD=qAcc_vd, VS=qAcc_vs, easy=qAcc_easy, hard=qAcc_hard),
        bias=dict(
            yes_pct_diff_all=dy, fp_ratio_all=rfp,
            yes_pct_diff_VD=dy_vd, fp_ratio_VD=rfp_vd,
            yes_pct_diff_VS=dy_vs, fp_ratio_VS=rfp_vs,
        ),
        parse_errors=parse_errors,
        failure_modes=failure_counts,
    )


def _mean_correct(samples: list[dict]) -> float | None:
    if not samples:
        return None
    return sum(s["_correct"] for s in samples) / len(samples)


def _figure_acc(fig_groups: dict[tuple, list[dict]]):
    def _acc(groups):
        if not groups:
            return None
        correct = sum(1 for g in groups.values() if all(s["_correct"] for s in g))
        return correct / len(groups)

    all_fg   = fig_groups
    vd_fg    = {k: v for k, v in fig_groups.items() if v[0]["_category"] == "VD"}
    vs_fg    = {k: v for k, v in fig_groups.items() if v[0]["_category"] == "VS"}
    easy_fg  = {k: v for k, v in fig_groups.items() if v[0]["_difficulty"] == "easy"}
    hard_fg  = {k: v for k, v in fig_groups.items() if v[0]["_difficulty"] == "hard"}

    return _acc(all_fg), _acc(vd_fg), _acc(vs_fg), _acc(easy_fg), _acc(hard_fg)


def _question_pair_acc(q_groups: dict[tuple, list[dict]]):
    def _acc(groups):
        if not groups:
            return None
        correct = sum(1 for g in groups.values() if all(s["_correct"] for s in g))
        return correct / len(groups)

    vd_q    = {k: v for k, v in q_groups.items() if v[0]["_category"] == "VD"}
    vs_q    = {k: v for k, v in q_groups.items() if v[0]["_category"] == "VS"}

    easy_q: dict[tuple, list[dict]] = {}
    hard_q: dict[tuple, list[dict]] = {}
    for k, v in q_groups.items():
        easy_samples = [s for s in v if str(s.get("visual_input", "")) != "2"]
        hard_samples = [s for s in v if str(s.get("visual_input", "")) == "2"]
        if easy_samples:
            easy_q[k] = easy_samples
        if hard_samples:
            hard_q[k] = hard_samples

    return _acc(q_groups), _acc(vd_q), _acc(vs_q), _acc(easy_q), _acc(hard_q)


def _bias_tests(samples: list[dict]) -> tuple[float | None, float | None]:
    if not samples:
        return None, None

    n = len(samples)
    pred_yes = sum(1 for s in samples if s["_parsed"] == 1)
    gt_yes   = sum(1 for s in samples if s["_gt"] == 1)
    dy = (pred_yes - gt_yes) / n

    wrong = [s for s in samples if not s["_correct"]]
    rfp = (sum(1 for s in wrong if s["_parsed"] == 1) / len(wrong)) if wrong else None

    return dy, rfp


def _parse_error_stats(samples: list[dict]) -> dict[str, dict[str, float | int | None]]:
    def _stats(group: list[dict]) -> dict[str, float | int | None]:
        total = len(group)
        unparseable = sum(1 for s in group if s["_parsed"] == 2)
        penalized = sum(1 for s in group if s["_parsed"] == 2 and not s["_no_visual"])
        accepted_vs_no_visual = sum(
            1 for s in group if s["_parsed"] == 2 and s["_no_visual"]
        )
        return dict(
            total=total,
            count=unparseable,
            pct=(unparseable / total) if total else None,
            penalized_count=penalized,
            accepted_vs_no_visual_count=accepted_vs_no_visual,
        )

    vd = [s for s in samples if s["_category"] == "VD"]
    vs = [s for s in samples if s["_category"] == "VS"]
    easy = [s for s in samples if s["_difficulty"] == "easy"]
    hard = [s for s in samples if s["_difficulty"] == "hard"]

    return dict(
        all=_stats(samples),
        VD=_stats(vd),
        VS=_stats(vs),
        easy=_stats(easy),
        hard=_stats(hard),
    )

def _diagnostic_test(samples: list[dict]) -> dict[str, Any]:
    pairs: dict[tuple, list[dict]] = defaultdict(list)
    for s in samples:
        key = (
            s["_category"],
            str(s.get("subcategory", "")),
            str(s.get("set_id", "")),
            str(s.get("question_id", "")),
        )
        pairs[key].append(s)

    counts = dict(language_hallucination=0, visual_illusion=0,
                  mixed_uncertain=0, total_failed_pairs=0)

    for (_, _, _, _), group in pairs.items():
        cat = group[0]["_category"]

        if cat == "VS":
            ref_samples  = [s for s in group if s["_no_visual"]]
            edit_samples = [s for s in group if not s["_no_visual"]]
        else:
            ref_samples  = [s for s in group if int(s.get("figure_id", 0)) == 0]
            edit_samples = [s for s in group if int(s.get("figure_id", 0)) != 0]

        all_correct = all(s["_correct"] for s in group)
        if all_correct:
            continue

        counts["total_failed_pairs"] += 1

        ref_correct  = all(s["_correct"] for s in ref_samples)  if ref_samples  else None
        edit_correct = all(s["_correct"] for s in edit_samples) if edit_samples else None

        if cat == "VS" and ref_samples:
            ref_s = ref_samples[0]
            if not ref_s["_correct"] and ref_s["_parsed"] != 2:
                counts["language_hallucination"] += 1
                continue

        if ref_correct and not edit_correct and edit_samples:
            ref_parsed  = ref_samples[0]["_parsed"]  if ref_samples  else None
            edit_parsed = edit_samples[0]["_parsed"]
            if ref_parsed is not None and ref_parsed == edit_parsed:
                counts["language_hallucination"] += 1
            else:
                counts["visual_illusion"] += 1
        elif not ref_correct and not edit_correct:
            counts["mixed_uncertain"] += 1
        elif ref_correct is None or edit_correct is None:
            counts["mixed_uncertain"] += 1
        else:
            counts["visual_illusion"] += 1

    total = counts["total_failed_pairs"]
    if total > 0:
        counts["pct_language_hallucination"] = counts["language_hallucination"] / total
        counts["pct_visual_illusion"]        = counts["visual_illusion"]        / total
        counts["pct_mixed_uncertain"]        = counts["mixed_uncertain"]        / total
    else:
        counts["pct_language_hallucination"] = None
        counts["pct_visual_illusion"]        = None
        counts["pct_mixed_uncertain"]        = None

    return counts

def _pct(v: float | None) -> str:
    return f"{v * 100:.2f}%" if v is not None else "N/A"


def print_results(results: dict[str, Any]) -> None:
    sep  = "=" * 72
    sep2 = "-" * 72

    print(sep)
    print(f"  HallusionBench Evaluation Results   (n={results['n_total']})")
    print(sep)

    header = f"{'Metric':<10} {'All':>10} {'VD':>10} {'VS':>10} {'Easy':>10} {'Hard':>10}"
    print(header)
    print(sep2)
    for metric in ("aAcc", "fAcc", "qAcc"):
        d = results[metric]
        row = (f"{metric:<10} {_pct(d['all']):>10} {_pct(d['VD']):>10} "
               f"{_pct(d['VS']):>10} {_pct(d['easy']):>10} {_pct(d['hard']):>10}")
        print(row)
    print(sep2)
    print()

    b = results["bias"]
    print("Bias Tests")
    print(sep2)
    print(f"  {'Yes% Diff (dy)':<30}  All={b['yes_pct_diff_all']:+.4f}  "
          f"VD={b['yes_pct_diff_VD'] if b['yes_pct_diff_VD'] is not None else 'N/A'}  "
          f"VS={b['yes_pct_diff_VS'] if b['yes_pct_diff_VS'] is not None else 'N/A'}")
    print(f"  {'FP Ratio (rfp)':<30}  All={_pct(b['fp_ratio_all'])}  "
          f"VD={_pct(b['fp_ratio_VD'])}  VS={_pct(b['fp_ratio_VS'])}")
    print()

    p = results["parse_errors"]
    print("Answer Parsing")
    print(sep2)
    print(
        f"  {'Unparseable rate':<30}  All={_pct(p['all']['pct'])}  "
        f"VD={_pct(p['VD']['pct'])}  VS={_pct(p['VS']['pct'])}"
    )
    print(
        f"  {'Unparseable count':<30}  All={p['all']['count']}/{p['all']['total']}  "
        f"VD={p['VD']['count']}/{p['VD']['total']}  VS={p['VS']['count']}/{p['VS']['total']}"
    )
    print(
        f"  {'Penalized as incorrect':<30}  "
        f"{p['all']['penalized_count']}"
    )
    print(
        f"  {'Accepted (VS no-visual)':<30}  "
        f"{p['all']['accepted_vs_no_visual_count']}"
    )
    print()

    fm = results["failure_modes"]
    print("Failure-Mode Breakdown (Diagnostic Test)")
    print(sep2)
    print(f"  Total failed question-pairs  : {fm['total_failed_pairs']}")
    print(f"  Language Hallucination       : {fm['language_hallucination']}  "
          f"({_pct(fm['pct_language_hallucination'])})")
    print(f"  Visual Illusion              : {fm['visual_illusion']}  "
          f"({_pct(fm['pct_visual_illusion'])})")
    print(f"  Mixed / Uncertain            : {fm['mixed_uncertain']}  "
          f"({_pct(fm['pct_mixed_uncertain'])})")
    print(sep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HallusionBench evaluation (first-token, no LLM judge).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--answers_file",
        type=str,
        required=True,
        help="JSONL produced by the answer generation script.",
    )
    parser.add_argument(
        "--question_file",
        type=str,
        default=None,
        help=(
            "Original HallusionBench.json — if provided, missing gt_answer / "
            "category / visual_input fields are merged into the answers."
        ),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to write the full results dict as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    answers_path = os.path.expanduser(args.answers_file)
    if not os.path.exists(answers_path):
        sys.exit(f"[ERROR] answers file not found: {answers_path}")

    print(f"Loading answers from {answers_path} …")
    samples = load_jsonl(answers_path)

    if args.question_file:
        qfile = os.path.expanduser(args.question_file)
        if os.path.exists(qfile):
            print(f"Merging ground-truth from {qfile} …")
            questions = load_json(qfile)
            samples = merge_gt(samples, questions)
        else:
            print(f"[WARN] question_file not found: {qfile} — skipping merge")

    required = {"category", "visual_input", "set_id", "figure_id",
                "question_id", "gt_answer"}
    missing = required - set(samples[0].keys()) if samples else set()
    if missing:
        sys.exit(
            f"[ERROR] Missing fields in answers: {missing}\n"
            "Pass --question_file to merge ground-truth annotations."
        )

    results = evaluate(samples)
    print_results(results)

    if args.output_json:
        out = os.path.expanduser(args.output_json)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {out}")


if __name__ == "__main__":
    main()