#%%
import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_diabetes_hospital
from matplotlib import pyplot as plt
from sklearn import set_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, mean_absolute_error)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skada import MultiLinearMongeAlignment
from skada._mapping import OTMapping
from skada.utils import source_target_split

#%%
# Load the dataset

df = fetch_diabetes_hospital(as_frame=True)
X_df = df["frame"]
y = X_df["readmit_binary"].to_numpy()

# remove Unknown gender for consistency
X_df = X_df.loc[X_df["gender"] != 'Unknown/Invalid']
y = y[X_df.index]

sample_domain = X_df["gender"].to_numpy()

# Drop target and sensitive feature to get the feature matrix, also drop the other "readmittance" columns
X = X_df.drop(columns=["gender", "readmit_binary", "readmitted", "readmit_30_days"])

# Take 10% of data while preserving the distribution
X, _, y, _, sample_domain, _ = train_test_split(
    X, y, sample_domain, test_size=0.9, random_state=42, stratify=sample_domain
)
    
#%%
# Encode the categorical features
for col in X.columns:
    if isinstance(X[col][X[col].first_valid_index()], str):
        cat = X[col].unique()

        # replace the value by its index in the value list
        X[col] = X[col].map({v: i for i, v in enumerate(cat)})

# Normalize the features
X = StandardScaler().fit_transform(X)

# Re-label domains, 
sample_domain = np.where(sample_domain == "Male", 1, -1)


#%%
X_source, X_target, y_source, y_target = source_target_split(X, y, sample_domain=sample_domain)
print(f"Source domain size: {X_source.shape[0]}")
print(f"Target domain size: {X_target.shape[0]}")

print(f"Source domain re-admittance mean: {y_source.mean()}")
print(f"Target domain re-admittance mean: {y_target.mean()}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(y_source, bins=[-0.5, 0.5, 1.5], alpha=0.5, label="Male (source)", rwidth=0.8)
plt.hist(y_target, bins=[-0.5, 0.5, 1.5], alpha=0.5, label="Female (target)", rwidth=0.8)
plt.legend(framealpha=0.5)
plt.xticks([0, 1])
plt.title("Readmittance distribution before adaptation")
plt.xlabel("Readmittance (0: No, 1: Yes)")
# %%
# Train classifier on source, evaluate performance and plot it

clf = RandomForestClassifier(n_estimators=5, random_state=31415)
clf.fit(X_source, y_source)

# Predictions for source and target
y_pred_source = clf.predict(X_source)
y_pred_target = clf.predict(X_target)

# Evaluate performance using accuracy
accuracy_source = accuracy_score(y_source, y_pred_source)
accuracy_target = accuracy_score(y_target, y_pred_target)

# Plot predictions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cm_source = ConfusionMatrixDisplay(
    confusion_matrix(
        y_source, y_pred_source, 
    )
)

cm_source.plot(ax=ax1)
ax1.set_title(f"Source (Accuracy={accuracy_source:.2f})")

cm_target = ConfusionMatrixDisplay(
    confusion_matrix(
        y_target, y_pred_target, 
    )
)

cm_target.plot(ax=ax2)
ax2.set_title(f"Target (Accuracy={accuracy_target:.2f})")

plt.tight_layout()
plt.show()

#%%
# ----------------------------------
# Build OTDA pipeline for classification

clf_otda = MultiLinearMongeAlignment(
    RandomForestClassifier(n_estimators=5, random_state=31415),
)

# modify y such that for the target domain there are only nan
y_for_fit = np.where(sample_domain == 1, y, np.nan)
clf_otda.fit(X, y_for_fit, sample_domain=sample_domain)

#%%
# Predict
y_pred_source_ot = clf_otda.predict(X_source)
y_pred_target_ot = clf_otda.predict(X_target)

# Evaluate performance using accuracy
accuracy_source_OT = accuracy_score(y_source, y_pred_source_ot)
accuracy_target_OT = accuracy_score(y_target, y_pred_target_ot)

print(f"Accuracy - Source: {accuracy_source_OT:.2f}")
print(f"Accuracy - Target: {accuracy_target_OT:.2f}")

# Plot predictions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cm_source = ConfusionMatrixDisplay(
    confusion_matrix(
        y_source, y_pred_source_ot, 
    )
)

cm_source.plot(ax=ax1)
ax1.set_title(f"Source (Accuracy={accuracy_source_OT:.2f})")

cm_target = ConfusionMatrixDisplay(
    confusion_matrix(
        y_target, y_pred_target_ot, 
    )
)

cm_target.plot(ax=ax2)
ax2.set_title(f"Target (Accuracy={accuracy_target_OT:.2f})")

plt.tight_layout()
plt.show()

# %%


def compute_demographic_parity_difference(y_pred, sensitive_attr):
    """Compute the demographic parity difference between two groups."""
    group_1 = y_pred[sensitive_attr == 1]
    group_2 = y_pred[sensitive_attr == -1]

    p1 = np.mean(group_1)
    p2 = np.mean(group_2)

    # Scale the difference by the overall mean prediction
    return np.abs(p1 - p2)


def compute_error_difference(y_pred, y_true, sensitive_attr):
    group_1 = y_pred[sensitive_attr == 1]
    group_2 = y_pred[sensitive_attr == -1]

    group_1_true = y_true[sensitive_attr == 1]
    group_2_true = y_true[sensitive_attr == -1]

    error_1 = np.linalg.norm(group_1 - group_1_true)
    error_2 = np.linalg.norm(group_2 - group_2_true)

    return np.abs(error_1 - error_2)


# %%
# Concatenate source predictions (before OTDA) with target predictions (before OTDA)
y_stacked_no_ot = np.concatenate([y_pred_source, y_pred_target])

# Concatenate source predictions (before OTDA) with target predictions (after OTDA)
y_stacked_ot = np.concatenate([y_pred_source, y_pred_target_ot])

# %%

print(
    "Demographic parity difference before OTDA:",
    compute_demographic_parity_difference(
        y_stacked_no_ot, sensitive_attr=sample_domain
    ),
)
print(
    "Demographic parity difference after OTDA:",
    compute_demographic_parity_difference(y_stacked_ot, sensitive_attr=sample_domain),
)

# compute percrentage improvement in demographic parity difference
improvement_dp = (
    compute_demographic_parity_difference(y_stacked_no_ot, sensitive_attr=sample_domain)
    - compute_demographic_parity_difference(y_stacked_ot, sensitive_attr=sample_domain)
) / compute_demographic_parity_difference(y_stacked_no_ot, sensitive_attr=sample_domain)

print(f"Percentage improvement in demographic parity difference: {improvement_dp:.2%}")
# %%

print(
    "Error difference before OTDA:",
    compute_error_difference(y_stacked_no_ot, y, sensitive_attr=sample_domain),
)
print(
    "Error difference after OTDA:",
    compute_error_difference(y_stacked_ot, y, sensitive_attr=sample_domain),
)
# %%
