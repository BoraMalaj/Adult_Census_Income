This hands on was completed as part of Group 1 assignment where the 
assigned task was Fairness Analysis in Machine Learning. The dataset 
used is the Adult Census Income dataset which contains 32561 records 
with 15 features including protected attributes such as gender, race, 
age and native country. The target variable is income classified as 
either below or above 50K dollars per year. Exploratory data analysis 
revealed that females and minority racial groups are significantly 
underrepresented in the dataset. Two models were trained without 
fairness considerations — Logistic Regression and Decision Tree — 
and their performance was evaluated separately for each demographic 
group. Significant performance gaps were found across gender and 
race groups indicating model bias. A reweighting mitigation technique 
was applied by assigning higher sample weights to underrepresented 
groups including females, minority races and elderly individuals. 
Models were retrained and fairness metrics were compared before and 
after mitigation showing improved equity across demographic groups 
with a minor tradeoff in overall accuracy.
