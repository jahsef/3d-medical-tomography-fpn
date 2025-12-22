import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def vanilla_cornernet_pos(pred, alpha=2.0):
    """Original CornerNet pos loss: -(1-p)^α * log(p)"""
    pred = np.clip(pred, 1e-3, 1 - 1e-3)
    return -((1 - pred) ** alpha) * np.log(pred)

def fuzzy_cornernet_pos(target, pred, alpha=2.0):
    """Modified pos loss: -(t-p)^α * log(p)"""
    pred = np.clip(pred, 1e-3, 1 - 1e-3)
    return -((target - pred) ** alpha) * np.log(pred)
    
def cornernet_neg(target, pred, alpha=2.0, beta=4.0):
    """CornerNet neg loss: -(1-t)^β * p^α * log(1-p)"""
    pred = np.clip(pred, 1e-3, 1 - 1e-3)
    return -((1 - target) ** beta) * (pred ** alpha) * np.log(1 - pred)


def combined_loss(target, pred, alpha=2.0, beta=4.0):
    """Generalized BCE: at alpha=1, beta=1 this is essentially BCE."""
    pred = np.clip(pred, 1e-3, 1 - 1e-3)

    focal_weighting = abs(target - pred) ** alpha
    pos_weight = (target ** beta) * focal_weighting * np.log(pred)

    high_conf_fp_weighting = pred ** alpha
    neg_weight = ((1 - target) ** beta) * high_conf_fp_weighting * np.log(1 - pred)

    return -(pos_weight + neg_weight)

def weighted_bce(target, pred, pos_weight=10.0):
    """Weighted BCE: (1 + t*w) * BCE"""
    pred = np.clip(pred, 1e-3, 1 - 1e-3)
    weights = 1.0 + (target * pos_weight)
    bce = -(target * np.log(pred) + (1 - target) * np.log(1 - pred))
    return weights * bce

def bce_focal(target, pred, gamma=2.0):
    """BCE Focal: |t-p|^γ * BCE"""
    pred = np.clip(pred, 1e-3, 1 - 1e-3)
    bce = -(target * np.log(pred) + (1 - target) * np.log(1 - pred))
    focal = np.abs(target - pred) ** gamma * bce
    return focal

def mae_focal(target, pred, gamma=2.0):
    """MAE Focal: |t-p|^γ * MAE"""
    mae = np.abs(pred - target)
    focal = np.abs(target - pred) ** gamma * mae
    return focal

# Configurable error offsets (positive = wrong direction)
CORRECT_OFFSET = 0.01
MILD_WRONG_OFFSET = 0.2
VERY_WRONG_OFFSET = 0.5

# Test scenarios
scenarios = [
    ("Peak 1.0 correct", 1.0, 1.0 - CORRECT_OFFSET),
    ("Peak 1.0 mild wrong", 1.0, 1.0 - MILD_WRONG_OFFSET),
    ("Peak 1.0 very wrong", 1.0, 1.0 - VERY_WRONG_OFFSET),
    
    ("Peak 0.8 correct", 0.8, 0.8 - CORRECT_OFFSET),
    ("Peak 0.8 mild wrong", 0.8, 0.8 - MILD_WRONG_OFFSET),
    ("Peak 0.8 very wrong", 0.8, 0.8 - VERY_WRONG_OFFSET),
    
    ("Falloff 0.7 correct", 0.7, 0.7 - CORRECT_OFFSET),
    ("Falloff 0.7 mild wrong", 0.7, 0.7 - MILD_WRONG_OFFSET),
    ("Falloff 0.7 very wrong", 0.7, 0.7 - VERY_WRONG_OFFSET),
    
    ("Background correct", 0.0, 0.0 + CORRECT_OFFSET),
    ("Background mild wrong", 0.0, 0.0 + MILD_WRONG_OFFSET),
    ("Background very wrong", 0.0, 0.0 + VERY_WRONG_OFFSET),
]

results = []
raw_losses = {'Vanilla': [], 'Fuzzy': [], 'Combined': [], 'WBCE': [], 'BCEFocal': [], 'MAEFocal': []}

for name, target, pred in scenarios:
    if target >= 0.75:
        fuzzy_loss = fuzzy_cornernet_pos(target, pred)
        vanilla_loss = vanilla_cornernet_pos(pred)
        loss_type = "pos"
    else:
        fuzzy_loss = cornernet_neg(target, pred)
        vanilla_loss = cornernet_neg(target, pred)
        loss_type = "neg"

    combined = combined_loss(target, pred)
    wbce = weighted_bce(target, pred)
    bce_f = bce_focal(target, pred)
    mae_f = mae_focal(target, pred)

    raw_losses['Vanilla'].append(vanilla_loss)
    raw_losses['Fuzzy'].append(fuzzy_loss)
    raw_losses['Combined'].append(combined)
    raw_losses['WBCE'].append(wbce)
    raw_losses['BCEFocal'].append(bce_f)
    raw_losses['MAEFocal'].append(mae_f)

    fuzzy_vanilla_ratio = f"{fuzzy_loss/vanilla_loss:.2f}x" if target >= 0.75 else "-"

    results.append({
        'Scenario': name,
        'Target': target,
        'Pred': pred,
        'Type': loss_type,
        'Vanilla': f"{vanilla_loss:.4f}",
        'Fuzzy': f"{fuzzy_loss:.4f}",
        'Combined': f"{combined:.4f}",
        'WBCE': f"{wbce:.4f}",
        'BCEFocal': f"{bce_f:.4f}",
        'MAEFocal': f"{mae_f:.4f}",
        'Fuzzy/Vanilla': fuzzy_vanilla_ratio,
    })

df = pd.DataFrame(results)
print("\n" + "="*100)
print("LOSS COMPARISON: Multiple Loss Functions")
print("="*100)
print(f"Error offsets: correct={CORRECT_OFFSET}, mild={MILD_WRONG_OFFSET}, very={VERY_WRONG_OFFSET}")
print("="*100)
print(df.to_string(index=False))
print("\nKey insights:")
print("- Vanilla/Fuzzy: CornerNet variants (piecewise pos/neg)")
print("- Combined: Non-piecewise formulation")
print("- WBCE: Weighted BCE (pos_weight=10)")
print("- BCEFocal: |t-p|^2 * BCE")
print("- MAEFocal: |t-p|^2 * MAE")

# Normalize each loss to 0-1 range and plot
normalized = {}
# visualize = ['Vanilla', 'Fuzzy', 'Combined', 'WBCE', 'BCEFocal', "MAEFocal"]
visualize = ['Fuzzy','Combined']
for loss_name, values in raw_losses.items():
    if loss_name in visualize:
        arr = np.array(values)
        min_val, max_val = arr.min(), arr.max()
        normalized[loss_name] = (arr - min_val) / (max_val - min_val) if max_val > min_val else arr

scenario_names = [s[0] for s in scenarios]
x = np.arange(len(scenario_names))

fig, ax = plt.subplots(figsize=(14, 6))
width = 0.15
for i, (loss_name, values) in enumerate(normalized.items()):
    ax.bar(x + i * width, values, width, label=loss_name)

ax.set_xlabel('Scenario')
ax.set_ylabel('Normalized Loss (0-1)')
ax.set_title('Loss Comparison (0-1 Normalized per Loss Function)')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(scenario_names, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.show()