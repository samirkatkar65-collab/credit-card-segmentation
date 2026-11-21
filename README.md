# Credit Card Usage Segmentation

Capstone project: segment credit card customers by spending and payment behaviour (unsupervised learning).

## Course tools used (from the academic calendar)

| Course session | Tool / topic | Where it is used in this project |
|---|---|---|
| #3-4 Fundamentals & Advanced Excel | Excel | `outputs/segmentation_results.xlsx` (comparison, profile, labelled customers). Build pivot tables from the "Customers + segment" sheet |
| #7-8 SQL basics & advanced | SQL (SQLite) | Notebook section 12: GROUP BY, CASE, subquery, window function (RANK) on `outputs/credit_card.db` |
| #9-12 Python basics, loops, lists, dicts | Python | Whole project (`segment_names` dictionary, loops over k and algorithms) |
| #13 NumPy | NumPy | log transform, scaling, distance arrays |
| #14 Pandas & data visualization | Pandas, Matplotlib, Seaborn | Cleaning, groupby profiling, all charts |
| #15-16 Statistics & hypothesis testing | SciPy | Section 13: Kruskal-Wallis test + effect size per feature |
| #19 Power BI / #20 Tableau | Power BI, Tableau | Load `outputs/powerbi_tableau_data.csv` and build the dashboard (see below) |
| #25, #29 Logistic Regression, SVM, KNN | Logistic, SVM, KNN vs Random Forest | Section 16: compare four classifiers that predict the segment |
| #9-12 Python basics | Functions, error handling, loops, tuples, lists, dicts, sets | Section 17 |
| #23-24 ML basics, Multiple Linear Regression | Predict PAYMENTS, R2/RMSE/MAE | Section 18 |
| #30 ARIMA / SARIMA | Section 19: synthetic demo. **Section 21b: REAL** daily credit card complaints (CFPB, `complaints.csv`), ARIMA vs SARIMA vs weekly baseline | Sections 19 and 21b (needs `python -m pip install statsmodels`) |
| #31-32 Deep learning | Multi-layer neural network, training curves | Section 20 |
| #33-34 NLP | Section 21: text derived from numbers (demo). **Section 21c: REAL** CFPB complaint text, TF-IDF clustering and relief classifier | Sections 21 and 21c |
| #37-39 GenAI / LLMs | Prompt templates, output checks, fine-tuning explained | Section 22 |
| #26 Decision trees & ensembles | Decision Tree, Random Forest | Section 14: explain and predict the segments |
| #27 Model evaluation & validation | Cross-validation, ARI | Section 15: 5-fold CV, confusion matrix, clustering stability |
| #28 Unsupervised learning | K-Means, Hierarchical, GMM, DBSCAN | Sections 5-8 (core of the project) |
| #40 Gradio deployment | Gradio | `app_gradio.py`: enter a customer, get the segment and suggested action |

Honest limits: GenAI/LLMs (#37-39) are used through prompt templates only (no LLM is trained or fine-tuned). Sections 19 (ARIMA) and 21 (NLP) are demonstrations on synthetic/derived data; the real-data versions are Sections 21b and 21c.

## Run
```
pip install -r requirements.txt
jupyter notebook Credit_Card_Segmentation.ipynb     # or open in VS Code and Run All
python app_gradio.py                                # after the notebook has run once
```

## Power BI / Tableau dashboard (suggested pages)
1. Overview: customers per segment (donut), average balance / purchases / cash advance by segment (bars).
2. Behaviour: scatter of PURCHASES vs CASH_ADVANCE coloured by Segment_Name.
3. Risk: PRC_FULL_PAYMENT and BALANCE by segment, with a slicer on Segment_Name.
