# Problem Statement And Topic Choice

## Why this topic was chosen

The topic "Predictive Health: Lifestyle and Chronic Disease" was chosen because it is relevant, explainable, and suitable for a full data science workflow.

- Chronic diseases such as diabetes and hypertension are common public-health problems.
- Many risk factors are related to lifestyle, which makes the problem actionable.
- The modeling goal is easy to explain to a non-technical audience: estimate disease risk from lifestyle and demographic patterns.
- The NHIS dataset already contains both predictors and disease outcomes, so the topic can be studied directly instead of using weak proxy labels.

## Main problem statement

The primary project question is:

> Can lifestyle and basic demographic factors predict whether an adult has ever been diagnosed with diabetes in the 2024 NHIS Sample Adult dataset?

## Why diabetes was selected as the main target

- It matches the assignment topic closely because diabetes is strongly linked to body weight, activity, smoking, diet-related conditions, and socioeconomic environment.
- The target already exists in the dataset as `DIBEV_A`.
- The screening stage showed that diabetes is predictive enough for a serious assignment while still keeping the story lifestyle-centered.
- It is easier to interpret than some alternative outcomes because the main predictors have strong public-health meaning.

## Why hypertension was kept as a secondary comparison

- Hypertension also scored well during screening.
- It is useful as a benchmark because it shows what happens when age and demographics carry even more of the signal.
- Comparing diabetes and hypertension strengthens the final discussion.

