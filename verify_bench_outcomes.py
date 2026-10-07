import json
import csv

# 1. Load multiclaim_metrics.csv
print("=== LOADING multiclaim_metrics.csv ===")
with open("results/multiclaim_metrics.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"  {row['Metric']}: {row['Value']}")

# 2. Load multiclaim_results.json
print("\n=== LOADING multiclaim_results.json ===")
with open("results/multiclaim_results.json", "r", encoding="utf-8") as f:
    mr = json.load(f)

print(f"Num samples: {mr['num_samples']}")
gt_metrics = mr["metrics"]["ground_truth_metrics"]
print(f"Ground truth metrics: Accuracy={gt_metrics['accuracy']:.4f}, Prec={gt_metrics['precision']:.4f}, Rec={gt_metrics['recall']:.4f}, F1={gt_metrics['f1_score']:.4f}, CPR={gt_metrics['cpr_gt']:.4f}, UMR={gt_metrics['umr_gt']:.4f}")

# Count response reports
total_claims = sum(r["total_claims"] for r in mr["response_reports"])
sup_claims = sum(r["supported_claims"] for r in mr["response_reports"])
contra_claims = sum(r["contradicted_claims"] for r in mr["response_reports"])
unv_claims = sum(r["unverifiable_claims"] for r in mr["response_reports"])

print(f"Total claims in response reports: {total_claims}")
print(f"Final claim labels: SUPPORTED={sup_claims}, CONTRADICTED={contra_claims}, UNVERIFIABLE={unv_claims}")

# Check correction outcomes for TP, FP, FN, TN
tp = 0
fp = 0
fn = 0
tn = 0

for co in mr["correction_outcomes"]:
    gt = co["ground_truth"] # True if hallucinated (CONTRADICTED)
    pred = co["predicted"]   # True if predicted hallucinated (CONTRADICTED)
    if gt and pred:
        tp += 1
    elif not gt and pred:
        fp += 1
    elif gt and not pred:
        fn += 1
    else:
        tn += 1

print(f"\nBinary Hallucination Confusion Matrix:")
print(f"  TP: {tp}")
print(f"  FP: {fp}")
print(f"  FN: {fn}")
print(f"  TN: {tn}")
prec = tp / (tp + fp) if (tp + fp) > 0 else 0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
print(f"Calculated: Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
