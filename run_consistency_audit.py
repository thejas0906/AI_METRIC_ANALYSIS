import json
import csv
import re
from pathlib import Path

# 1. Read config.py
print("=== 1. Reading config.py ===")
with open("config.py", "r", encoding="utf-8") as f:
    cfg_text = f.read()

def get_config_val(name, text):
    m = re.search(rf"^{name}\s*=\s*(.+)$", text, re.MULTILINE)
    if m:
        val = m.group(1).split("#")[0].strip()
        return eval(val) if val.startswith('"') or val.startswith("'") or val.replace(".","").isdigit() else val
    return None

css_sup = get_config_val("CSS_SUPPORTED_THRESHOLD", cfg_text)
css_insuf = get_config_val("CSS_INSUFFICIENT_THRESHOLD", cfg_text)
contra_thresh = get_config_val("CONTRADICTION_THRESHOLD", cfg_text)
contra_mode = get_config_val("CONTRADICTION_MODE", cfg_text)

print(f"CSS_SUPPORTED_THRESHOLD    = {css_sup}")
print(f"CSS_INSUFFICIENT_THRESHOLD = {css_insuf}")
print(f"CONTRADICTION_THRESHOLD    = {contra_thresh}")
print(f"CONTRADICTION_MODE         = {contra_mode}")

# 2. Read results/multiclaim_metrics.csv
print("\n=== 2. Reading results/multiclaim_metrics.csv ===")
csv_metrics = {}
with open("results/multiclaim_metrics.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        csv_metrics[row["Metric"]] = float(row["Value"])
        print(f"  {row['Metric']}: {row['Value']}")

# 3. Read results/multiclaim_results.json
print("\n=== 3. Reading results/multiclaim_results.json ===")
with open("results/multiclaim_results.json", "r", encoding="utf-8") as f:
    mr = json.load(f)

json_metrics = mr["metrics"]["ground_truth_metrics"]
print("Metrics in JSON:")
for k, v in json_metrics.items():
    print(f"  {k}: {v}")

total_claims = 0
supported_count = 0
contradicted_count = 0
unverifiable_count = 0

for rep in mr["response_reports"]:
    total_claims += rep["total_claims"]
    supported_count += rep["supported_claims"]
    contradicted_count += rep["contradicted_claims"]
    unverifiable_count += rep["unverifiable_claims"]

print(f"\nResponse Reports Aggregated Counts:")
print(f"  Total Claims:       {total_claims}")
print(f"  SUPPORTED count:    {supported_count}")
print(f"  CONTRADICTED count: {contradicted_count}")
print(f"  UNVERIFIABLE count: {unverifiable_count}")

# 4. Check correction outcomes
outcomes = mr["correction_outcomes"]
print(f"\nCorrection Outcomes Count: {len(outcomes)}")

tp = 0
fp = 0
fn = 0
tn = 0

for o in outcomes:
    gt = o["ground_truth"] # True if hallucinated
    pred = o["predicted"]   # True if detected as hallucinated
    if gt and pred:
        tp += 1
    elif not gt and pred:
        fp += 1
    elif gt and not pred:
        fn += 1
    else:
        tn += 1

print(f"Outcomes Confusion Matrix:")
print(f"  TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
print(f"  Total = {tp + fp + fn + tn}")

calc_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
calc_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
calc_f1 = 2 * calc_prec * calc_rec / (calc_prec + calc_rec) if (calc_prec + calc_rec) > 0 else 0
calc_hda = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

print(f"\nCalculated from Outcomes:")
print(f"  Precision: {calc_prec:.8f} (vs CSV: {csv_metrics.get('Precision'):.8f}, JSON: {json_metrics['precision']:.8f})")
print(f"  Recall:    {calc_rec:.8f} (vs CSV: {csv_metrics.get('Recall'):.8f}, JSON: {json_metrics['recall']:.8f})")
print(f"  F1 Score:  {calc_f1:.8f} (vs CSV: {csv_metrics.get('F1 Score'):.8f}, JSON: {json_metrics['f1_score']:.8f})")
print(f"  HDA:       {calc_hda:.8f} (vs CSV: {csv_metrics.get('Hallucination Detection Accuracy (HDA)'):.8f}, JSON: {json_metrics['accuracy']:.8f})")

# Check all other json files in results/
print("\n=== Checking other result files in results/ ===")
for p in Path("results").glob("*.json"):
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and "response_reports" in d:
            tot = sum(r["total_claims"] for r in d["response_reports"])
            sup = sum(r["supported_claims"] for r in d["response_reports"])
            con = sum(r["contradicted_claims"] for r in d["response_reports"])
            unv = sum(r["unverifiable_claims"] for r in d["response_reports"])
            print(f"  {p.name}: Total={tot}, Sup={sup}, Con={con}, Unv={unv}")
    except Exception as e:
        pass
