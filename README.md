# Predicting Video Game Hits (PS2, Xbox, GameCube)

This project tries to predict whether a PS2, Xbox, or GameCube game released between 2000 and 2006 sold more than 1 million copies. I used two models, Logistic Regression and Random Forest, and compared them.

## Dataset

I used the cleaned file from this Kaggle dataset: https://www.kaggle.com/datasets/bhushandivekar/video-game-sales-and-industry-data-1980-2024

The file is called `Video_Games_Sales_Cleaned.csv`. It has 18,832 games and 8 columns. After I kept only PS2, Xbox, and GameCube games from 2000 to 2006 and dropped 22 games with 0 sales, I had 2,804 games left. 289 of them are hits (10.3%).

## What I did

- Made the label: 1 if the game sold more than 1 million copies, 0 if not.
- Found that about 78% of the critic scores were exactly 7.3, which looks like the dataset filled in missing scores with the median. I treated those as missing and added a column that says if a game has a real score.
- One-hot encoded console and genre, then did an 80/20 train/test split.
- Used 5-fold cross-validation with GridSearchCV to tune both models, scored on F1 for the hit class.
- Logistic Regression: tuned `C` and `class_weight`.
- Random Forest: tuned `n_estimators` and `max_depth`.

## Results

| Model | Test F1 | Precision | Recall |
|---|---|---|---|
| Logistic Regression | 0.309 | 0.194 | 0.759 |
| Random Forest | 0.333 | 0.230 | 0.603 |

Random Forest was slightly better on F1, but the difference is small. There are only 58 hits in the test set, so the scores are noisy. Also, some games came out on more than one console, so the same title can be in both the train and test sets. That probably makes the scores a bit too optimistic.

## How to run

Put `Video_Games_Sales_Cleaned.csv` in the same folder as the script, then run:

```bash
pip install pandas numpy matplotlib scikit-learn
python finalproject.py
```

The plots will pop up one at a time, so close each window to keep going. Everything also gets saved to the `outputs/` folder, including the plots and `results.txt` with the classification reports.

## Files

- `finalproject.py` is the full script.
- `Video_Games_Sales_Cleaned.csv` is the data.
- `outputs/` has the saved plots and results.
