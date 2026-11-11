# AI Disclosure

## Statement of use

During development of this assignment I used  generative AI
**Gemini**, as a support tool for debugging, structural planning, and document
templating. I did not treat the assistant's output as evidence of experimental
results. I executed the pipeline inspected the
generated outputs and  verified the calculations used in
the final evaluation.

## Structured usage log

| Area | How Gemini was used | My verification and responsibility |
|---|---|---|
| Assignment constraints and data leakage | I asked Gemini to review strict preprocessing requirements and help identify leakage risks. The resulting structural fix ensured that each scaler was fitted only on its corresponding training subset and then applied to validation and test subsets. | I executed the notebooks independently and checked that the fixed test set was reserved before the repeated training/validation splits. I also verified that separate scalers were used for each dataset and seed. |
| ROC-AUC implementation | I asked Gemini to help correct the ROC-AUC logic. The implementation uses a temporary binary target for Breast Cancer, with malignant class 0 mapped to 1 and the probability of original class 0 selected. Wine uses class probabilities with `multi_class="ovr"` and `average="macro"`. | I independently ran the pipeline and inspected the ROC-AUC outputs and ROC figures. I verified that hard class predictions were not used for ROC-AUC calculation. |
| Configuration selection | I asked Gemini to help structure the required model-selection tie-breaker. The implemented order is mean validation macro-F1, mean validation accuracy, lower validation macro-F1 standard deviation, and alphabetical `configuration_id`. | I checked the sorting columns and order in the pipeline and verified the selected configurations against the generated summary table. |
| CSV exports | I asked Gemini for boilerplate pandas export structure to ensure that exactly six CSV files were produced with the required row counts. | I independently executed both notebooks and verified the exported files and counts: 132 classifier runs, 44 classifier summaries, 2 classifier test rows, 12 K-Means runs, 4 K-Means summaries, and 4 K-Means test rows. |
| Drafting document templates | I asked Gemini to draft structural templates for the report and README, including section headings, methodological descriptions, and placeholders for tables and figures. | I reviewed the templates, supplied the project context, and remain responsible for checking, completing, and explaining all submitted content. |

