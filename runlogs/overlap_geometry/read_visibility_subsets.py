import os, yaml

D = "/nfs/home/charlie/InCoP/opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42"
TAGS = {"LOW": "data_hospital", "HIGH": "data_hospital_highoverlap"}
NOISE = [("", 0.0), ("noise_t0p05_r0p05", .05), ("noise_t0p1_r0p1", .10),
         ("noise_t0p15_r0p15", .15), ("noise_t0p2_r0p2", .20),
         ("noise_t0p3_r0p3", .30), ("noise_t0p4_r0p4", .40),
         ("noise_t0p6_r0p6", .60), ("noise_t0p8_r0p8", .80)]
SUB = ["ego_only", "shared", "partner_only"]

def load(tag, noise):
    mid = f"_{noise}" if noise else ""
    p = os.path.join(D, f"eval_all_isaac_intermediate_{tag}_epoch15{mid}"
                        "_score0p25_nms0p15_global_sort.yaml")
    return yaml.safe_load(open(p)) if os.path.exists(p) else None

print("=== GT composition at zero noise (the overlap measurement itself) ===")
print(f"{'split':6s} {'total GT':>9s} " + " ".join(f"{s:>14s}" for s in SUB))
for name, tag in TAGS.items():
    y = load(tag, "")
    if y is None:
        print(f"{name:6s} MISSING"); continue
    d = y["visibility_subset_metrics"]["distribution"]
    tot = d["total_union_gt"]
    cells = " ".join(f"{d[s]['gt_count']:6d} ({d[s]['gt_ratio']*100:4.1f}%)" for s in SUB)
    print(f"{name:6s} {tot:9d} {cells}")

print()
print("=== subset mAP@0.3, and retention vs own clean baseline ===")
for name, tag in TAGS.items():
    base = {}
    print(f"-- {name} overlap")
    print(f"{'pos':>5s} " + " ".join(f"{s:>22s}" for s in SUB))
    for noise, std in NOISE:
        y = load(tag, noise)
        if y is None:
            print(f"{std:5.2f}  (missing)"); continue
        s = y["visibility_subset_metrics"]["subsets"]
        row = []
        for sub in SUB:
            v = s[sub]["detection"]["mAP@0.3"]
            if std == 0.0:
                base[sub] = v
            ret = 100.0 * v / base[sub] if base.get(sub) else float("nan")
            row.append(f"{v:.4f} ({ret:5.1f}%)")
        print(f"{std:5.2f} " + " ".join(f"{c:>22s}" for c in row))
    print()
