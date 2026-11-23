# %% Setup
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score)

try:
    BASE = Path(__file__).resolve().parent   # folder this file is in
except NameError:
    BASE = Path.cwd()
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

extra_text = []   # lines that get written to results.txt at the end

# %% Load data
df = pd.read_csv(BASE / "Video_Games_Sales_Cleaned.csv")
print("Full dataset:", df.shape)

# %% Keep only PS2, Xbox, GameCube games from 2000-2006
sub = df[df["console"].isin(["PS2", "XB", "GC"])].copy()
sub = sub[sub["release_year"].between(2000, 2006)].copy()
sub["release_year"] = sub["release_year"].astype(int)
print("After filtering:", sub.shape)

# %% Cleaning
# 22 games have 0 sales, which is probably missing data, so drop them
sub = sub[sub["total_sales"] > 0].copy()

# Label: 1 = hit (more than 1 million copies), 0 = not a hit
sub["success"] = (sub["total_sales"] > 1).astype(int)

# 78% of critic scores are exactly 7.3 (the dataset filled missing scores with the median).
# Treat 7.3 as "no real score" and add a flag saying whether a real score exists.
sub["has_critic_score"] = (sub["critic_score"] != 7.3).astype(int)
sub.loc[sub["has_critic_score"] == 0, "critic_score"] = np.nan

print("Rows:", len(sub), "| hits:", sub["success"].sum(), f"({sub['success'].mean():.1%})")
print(sub["success"].value_counts())
print(sub.groupby("console")["success"].agg(["count", "mean"]).round(3))
print(sub.groupby("genre")["success"].agg(["count", "mean"]).round(3))
print("Real critic scores:", sub["has_critic_score"].sum(), "of", len(sub))

# Hit rate by release year (release_year was an important feature, so check why)
year_table = sub.groupby("release_year")["success"].agg(["count", "mean"]).round(3)
print(year_table)
extra_text.append("Hit rate by release year:\n" + year_table.to_string())

# %% Class distribution plot
ax = sub["success"].value_counts().sort_index().plot(kind="bar", color=["gray", "seagreen"])
ax.set_xticks([0, 1])
ax.set_xticklabels(["Not hit (<=1M)", "Hit (>1M)"], rotation=0)
ax.set_xlabel("")
ax.set_ylabel("Number of games")
ax.set_title("Class distribution")
plt.savefig(OUT / "01_class_distribution.png", dpi=200, bbox_inches="tight")
plt.show()

# %% EDA plot 1: hit rate by console and genre
by_console = sub.groupby("console")["success"].mean()
by_genre = sub.groupby("genre")["success"].agg(["count", "mean"])
by_genre = by_genre[by_genre["count"] >= 30]["mean"].sort_values()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
by_console.plot(kind="bar", ax=ax[0], rot=0, title="Hit rate by console")
ax[0].set_ylabel("Share of games that are hits")
ax[0].set_xlabel("")
by_genre.plot(kind="barh", ax=ax[1], title="Hit rate by genre (genres with 30+ games)")
ax[1].set_ylabel("")
plt.tight_layout()
plt.savefig(OUT / "02_eda_console_genre.png", dpi=200, bbox_inches="tight")
plt.show()

# %% EDA plot 2: critic score for hits vs non-hits (real scores only)
real = sub[sub["has_critic_score"] == 1]
real.boxplot(column="critic_score", by="success", figsize=(5, 4))
plt.title("Critic score by outcome (real scores only)")
plt.suptitle("")
plt.xticks([1, 2], ["Not hit", "Hit"])
plt.xlabel("")
plt.savefig(OUT / "03_eda_critic_score.png", dpi=200, bbox_inches="tight")
plt.show()

# %% Encoding and train/test split
# One-hot encode console and genre (turn each category into a 0/1 column)
X = pd.get_dummies(
    sub[["console", "genre", "release_year", "critic_score", "has_critic_score"]],
    columns=["console", "genre"], dtype=int)
y = sub["success"]
print("Number of features:", X.shape[1])

# 80/20 split. stratify=y keeps the same hit rate in train and test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape, "hits:", y_train.sum())
print("Test: ", X_test.shape, "hits:", y_test.sum())

# For the limitations slide: a game released on more than one console can land in both sets
titles = sub["title"].str.lower().str.strip()
train_titles = set(titles.loc[X_train.index])
overlap_mask = titles.loc[X_test.index].isin(train_titles).values
n_overlap_titles = len(train_titles & set(titles.loc[X_test.index]))
print("Titles in both train and test:", n_overlap_titles)
print("Test rows whose title is also in train:", overlap_mask.sum(), "of", len(X_test))
extra_text.append(f"Titles in both train and test: {n_overlap_titles}\n"
                  f"Test rows whose title is also in train: {overlap_mask.sum()} of {len(X_test)}")

# %% Model 1: Logistic Regression
# The pipeline fills missing critic scores with the median, scales the features,
# then fits the model. Inside GridSearchCV this is redone on each fold, so
# nothing from the validation fold leaks into the preprocessing.
lr_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])
lr_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],            # regularization strength (smaller = stronger)
    "model__class_weight": ["balanced", None]       # does weighting the rare hit class help?
}
lr_search = GridSearchCV(lr_pipe, lr_grid, cv=5, scoring="f1", n_jobs=1)
lr_search.fit(X_train, y_train)
print("LR best params:", lr_search.best_params_)
print("LR best CV F1: %.3f" % lr_search.best_score_)

res = pd.DataFrame(lr_search.cv_results_)
res["cw"] = res["param_model__class_weight"].astype(str)
plt.figure(figsize=(7, 4))
for cw in res["cw"].unique():
    s = res[res["cw"] == cw].sort_values("param_model__C")
    plt.plot(s["param_model__C"], s["mean_test_score"], marker="o", label=f"class_weight={cw}")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("CV F1 (hit class)")
plt.title("Logistic Regression: CV F1 vs C")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(OUT / "04_lr_tuning.png", dpi=200, bbox_inches="tight")
plt.show()

# %% Model 2: Random Forest (no scaling needed for trees)
rf_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
])
rf_grid = {
    "model__n_estimators": [50, 100, 200],   # number of trees
    "model__max_depth": [3, 5, 10, None]     # how deep each tree can grow
}
rf_search = GridSearchCV(rf_pipe, rf_grid, cv=5, scoring="f1", n_jobs=1)
rf_search.fit(X_train, y_train)
print("RF best params:", rf_search.best_params_)
print("RF best CV F1: %.3f" % rf_search.best_score_)

res_rf = pd.DataFrame(rf_search.cv_results_)
res_rf["depth"] = res_rf["param_model__max_depth"].astype(str)
plt.figure(figsize=(7, 4))
for d in res_rf["depth"].unique():
    s = res_rf[res_rf["depth"] == d].sort_values("param_model__n_estimators")
    plt.plot(s["param_model__n_estimators"], s["mean_test_score"], marker="o", label=f"max_depth={d}")
plt.xlabel("n_estimators")
plt.ylabel("CV F1 (hit class)")
plt.title("Random Forest: CV F1 vs number of trees")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))   # legend outside so it doesn't cover a line
plt.grid(alpha=0.3)
plt.savefig(OUT / "05_rf_tuning.png", dpi=200, bbox_inches="tight")
plt.show()

# %% Train and test results for both models
lr_best = lr_search.best_estimator_
rf_best = rf_search.best_estimator_
report_text = []

def show_results(name, model, tag):
    for split, Xs, ys in [("train", X_train, y_train), ("test", X_test, y_test)]:
        pred = model.predict(Xs)
        report = classification_report(ys, pred, target_names=["Not hit", "Hit"], zero_division=0)
        print(f"\n{name} - {split}\n{report}")
        report_text.append(f"{name} - {split}\n{report}")
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(ys, pred, display_labels=["Not hit", "Hit"],
                                                ax=ax, colorbar=False)
        ax.set_title(f"{name} - {split}")
        plt.savefig(OUT / f"{tag}_{split}.png", dpi=200, bbox_inches="tight")
        plt.show()

show_results("Logistic Regression", lr_best, "06_lr_confusion")
show_results("Random Forest", rf_best, "07_rf_confusion")

# %% Compare the two models on the test set
scores = {}
for name, model in [("Logistic Regression", lr_best), ("Random Forest", rf_best)]:
    pred = model.predict(X_test)
    scores[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
    }
compare = pd.DataFrame(scores).T
print("\nTest-set comparison:\n", compare.round(3))
baseline = 1 - y_test.mean()
print("Accuracy if we always predicted 'Not hit': %.3f" % baseline)

compare.plot(kind="bar", figsize=(8, 4), ylim=(0, 1), rot=0, title="Test-set metrics (Hit = positive class)")
plt.ylabel("Score")
plt.savefig(OUT / "08_model_comparison.png", dpi=200, bbox_inches="tight")
plt.show()

# %% Sanity check: score only the test games whose title is NOT in the training set
new_mask = ~overlap_mask
X_new, y_new = X_test.loc[new_mask], y_test.loc[new_mask]
line = f"Test rows with an unseen title: {len(X_new)} | hits: {y_new.sum()}"
print("\n" + line)
extra_text.append("\n" + line)
for name, model in [("Logistic Regression", lr_best), ("Random Forest", rf_best)]:
    f1_new = f1_score(y_new, model.predict(X_new), zero_division=0)
    line = f"{name} F1 on unseen titles: {f1_new:.3f}"
    print(line)
    extra_text.append(line)

# %% Which features mattered most? (Random Forest)
imp = pd.Series(rf_best.named_steps["model"].feature_importances_, index=X.columns)
imp.sort_values().tail(10).plot(kind="barh", figsize=(7, 4), title="Random Forest: top 10 features")
plt.tight_layout()
plt.savefig(OUT / "09_feature_importance.png", dpi=200, bbox_inches="tight")
plt.show()

# %% Save text results
with open(OUT / "results.txt", "w") as f:
    f.write(f"LR best params: {lr_search.best_params_}, CV F1 {lr_search.best_score_:.3f}\n")
    f.write(f"RF best params: {rf_search.best_params_}, CV F1 {rf_search.best_score_:.3f}\n\n")
    f.write("\n\n".join(report_text))
    f.write("\n\nTest-set comparison:\n" + compare.round(3).to_string())
    f.write(f"\nAlways-'Not hit' accuracy on test: {baseline:.3f}\n\n")
    f.write("\n".join(extra_text))
print("Saved results to", OUT)