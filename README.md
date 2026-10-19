# Predicting and Preventing Student Dropout with Apache Spark

Final project – Business Data Analytics with Apache Spark

## 1. The business problem

Every student who drops out is a loss for the university (tuition revenue, reputation, funding) and for the student. Universities have support programs (tutoring, scholarships, counselling), but their budget is limited: they cannot follow every student closely.

**Our question:**

> With a limited support budget, how can a university identify the students most at risk of dropping out as early as the first semester, and which levers should it act on first?

- **Decision-maker:** academic affairs and the student success department
- **Decision:** which students to support first, and where to invest (scholarships, on-campus jobs, tutoring)
- **Success metric:** share of future dropouts reached when supporting only 20% of students

## 2. Dataset

- File: `student_dropout_risk_2026.csv`
- **100,000 students, 34 columns**: profile (age, income level, scholarship, first-generation student), academic results (admission grade, GPA, failed units), engagement (attendance, online platform logins, assignments, missed deadlines), personal life (part-time job, working hours, stress, sleep)
- Target: `dropout` (1 = the student dropped out). **32% of students drop out.**
- No missing values and no duplicates. The data is very likely synthetic (see limitations).

## 3. How we analysed it

All the analysis is done with Apache Spark (PySpark) in `student_dropout_spark.ipynb`.

1. **Data quality check** (DataFrame API): schema, missing values, duplicates, target distribution.
2. **Exploration with Spark SQL**: dropout rate by GPA, attendance, income, scholarship, job, stress, field of study, gender…
3. **Prediction model with Spark MLlib**: a Pipeline (StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler) followed by a classifier. We compared:
   - Logistic regression vs Random Forest
   - an **early-warning model** (first-semester data only, without the second-semester GPA) vs the full model
   - a model with and without gender and nationality (ethics check)
   - Evaluation on a 70/30 train/test split with AUC, accuracy, precision and recall.
4. **Budget targeting**: students are ranked by predicted risk, and we measure how many future dropouts are reached when supporting the top 10%, 20%, 30%… (gain curve).
5. **Action levers** (Spark SQL): effect of scholarships at the same income level, and effect of working hours within each income level.

## 4. Main findings

**Who drops out?**

| Factor | Dropout rate |
|---|---|
| First-semester GPA < 2.0 vs ≥ 3.5 | 74% vs 2% |
| Attendance < 60% vs ≥ 90% | 53% vs 16% |
| Low vs high family income | 43% vs 20% |
| Tuition not up to date vs up to date | 44% vs 29% |
| Stress level 1 vs 10 | 17% vs 44% |
| Field of study, gender, study mode, international status | ~32% everywhere: **no real effect** |

Students receiving mental health support drop out slightly *more* (34% vs 31%). This does not mean support is harmful: struggling students are the ones who seek it. **Correlation is not causation.**

**The model**

| Model | AUC |
|---|---|
| **Logistic regression – early warning (selected)** | **0.900** |
| Random Forest – early warning | 0.881 |
| Logistic regression – full (with semester 2 GPA) | 0.904 |
| Early warning + gender and nationality | 0.900 |

- At-risk students can be identified **as early as the first semester**: the second-semester GPA adds almost nothing.
- Removing gender and nationality costs nothing in accuracy, so we leave them out.
- Strongest predictors: first-semester GPA, attendance, unpaid tuition, online platform activity, missed deadlines, failed units, working hours, stress.

**Targeting with a limited budget**

| Students supported (highest risk first) | Dropouts reached |
|---|---|
| 10% | 30% |
| **20%** | **54%** |
| 30% | 72% |
| 50% | 91% |

**By supporting only 20% of students, the university reaches more than half of future dropouts** (vs 20% with random targeting).

**Levers**

- **Scholarships:** among low-income students, dropout is 49% without a scholarship vs 40% with one (−9 points), vs only −4 points for high-income students. **10,393 low-income students have no scholarship.**
- **Student work:** dropout rises steadily with weekly working hours: 25% (no job) → 32% (1–10 h) → 38% (11–20 h) → 45% (21–30 h) → 52% (31–40 h). The same increase appears at every income level, so work itself weighs on success.

## 5. Our recommendations

1. **Run the early-warning score every semester** and classify students into three risk levels:

| Risk level | Action |
|---|---|
| High (> 60%) | One-to-one meeting, tutoring, priority scholarship review, schedule adjustment if heavy job |
| Medium (30–60%) | Attendance monitoring, automatic alert if attendance drops |
| Low (< 30%) | No specific action |

2. **Target scholarships at low-income students**, where they are associated with the largest drop in dropout.
3. **Limit the impact of student work**: offer on-campus jobs with capped hours and flexible schedules for students working more than 20 hours a week.
4. **Monitor attendance continuously**: it is available long before exam results and is one of the strongest warning signs.

**Why Spark?** The same pipeline (Spark SQL + MLlib) processes 100,000 students in seconds and would scale to millions of rows (several universities, several years) without changing the code.

## 6. Limitations

- The dataset is very likely synthetic: results should be validated on real university data.
- The scholarship and work effects are **correlations**, not proven causes. An experiment (e.g. granting scholarships to a test group) would measure the real effect.
- The risk score must be used to **help** students, never to exclude them or filter admissions.

## 7. AI usage

We used an AI assistant (Claude, by Anthropic) during this project. This section describes what it was used for and what we did ourselves.

**What the AI was used for**

- Rephrasing the assignment and listing what had to be delivered.
- Proposing several possible business questions for our dataset, with a first look at the data to check that each one was supported by a real signal.
- Writing a first version of the PySpark notebook (data loading, Spark SQL queries, MLlib pipeline, gain curve) and of this README.
- Explaining Spark concepts we were not sure about: what MLlib is, why we fix a random seed, what `printSchema()` checks, and the limits of Databricks serverless compute (no `cache()`, no RDDs).
- Translating the notebook comments and explanations into English.

**What we did ourselves**

- Chose the dataset.
- Chose the business question among the proposals and defined the scope of the project.
- Ran the notebook on Databricks, fixed the errors we met (for example a missing cell after editing), and simplified the code (fewer charts, no custom styling or helper functions) so that it stays readable and we can explain every cell.
- Reviewed the results and the recommendations before presenting them.

**How we checked the AI's output**

- The whole notebook was run end to end on Databricks: all figures quoted in this README come from that run, not from the AI's text.
- We asked for an explanation of every part we did not understand instead of copying code blindly.
- We kept the limitations explicit (synthetic data, correlation vs causation) rather than presenting the results as stronger than they are.

## 8. How to run the project

### Option A – Databricks (Free Edition)

1. **Catalog** > `workspace` > `default` > **Create** > **Volume** named `projet_spark`, then **Upload to this volume** and add `student_dropout_risk_2026.csv`.
2. **Workspace** > **Import** > `student_dropout_spark.ipynb`.
3. Attach the notebook to **Serverless** compute (environment version 4 or later, required for MLlib).
4. Click **Run all**.

If the volume has another name, update `DATABRICKS_PATH` in the third code cell.

### Option B – Local

Requirements: Python 3.10–3.13 and Java 17 or 21.

```bash
pip install pyspark==4.0.4 pandas matplotlib pyarrow jupyter
jupyter notebook student_dropout_spark.ipynb
```

Keep the CSV in the same folder as the notebook, then run all cells.

The random seed is fixed (`SEED = 42`) for reproducibility. Results may vary very slightly between environments because of how Spark partitions the data.

## Repository structure

```
├── README.md
├── student_dropout_spark.ipynb    # Spark analysis (PySpark, Spark SQL, MLlib)
└── student_dropout_risk_2026.csv  # dataset (100,000 rows)
```
