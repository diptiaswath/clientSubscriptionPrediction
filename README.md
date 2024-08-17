Client Subscription Prediction

Objective

The primary goal of this project is to develop a predictive model for banking client subscriptions that balances high predictive accuracy with minimizing the number of contacts during marketing campaigns. This approach is expected to enhance resource allocation, reduce unnecessary outreach, and improve client engagement.

Key Performance Metrics:

To achieve this objective, the model should:
1. Accurately Predict Client Subscriptions: Ensure high predictive accuracy and correctly identify clients who will subscribe.
2. Minimize Unnecessary Contacts: Reduce false positives to limit the number of clients contacted unnecessarily.

The model should strike a balance between high precision to reduce unnecessary contacts and high recall to ensure successful subscriptions are correctly identified. 

The goal with these metrics is to retain or improve the performance metrics such as, F1 score, PR AUC and ROC AUC score, that were observed before any hyperparameter tuning or decision threshold adjustments, ensuring the model continues to perform well in terms of accuracy and efficiency within the marketing context.

Project Scope:

In this project, we intend to compare the performance of classifiers: K-Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines. We use a multivariate dataset related to marketing bank products via telephone to predict whether a client will subscribe to a term deposit (variable y). This dataset falls under the business subject area and focuses on predicting client subscriptions based on various features. This dataset is based on "Bank Marketing" UCI dataset at: http://archive.ics.uci.edu/ml/datasets/Bank+Marketing.

Executive Summary 

Key Factors Considered: 

1. Optimizing Resource Allocation: The model should accurately identify high-potential clients, allowing targeted marketing efforts and making campaigns more cost-effective.
2. Reducing False Positives: Enhancing model precision helps lower the number of clients contacted who are unlikely to subscribe, avoiding unnecessary outreach and streamlining the marketing process.
3. Enhancing Client Experience: Fewer, more relevant communications improve client experience, reduce fatigue, and increase marketing effectiveness.
4. Model Performance and Trade-Offs: The model needs to balance a high F1 score, which reflects accurate subscription predictions, with minimizing ineffective contacts. This balance ensures marketing efforts are both accurate and efficient.
5. Feature Importance: Key features are expected to be identified as these are crucial for refining predictions and minimizing unnecessary contacts, ultimately enhancing model's performance metrics.


Key Insights and Implications:

The input dataset is highly imbalanced, with 88.73% of instances labeled as 'no' and only 11.27% as 'yes' for positive client subscriptions. This imbalance impacts the model's ability to accurately predict the minority 'yes' class. To address this issue, technique such as SMOTE resampling was employed to enhance the model's performance and ensure effective identification of both majority and minority classes.

SMOTE resampling was applied to both the entire dataset and just the training data to compare model performance. Applying SMOTE to the entire dataset improved the test scores but led to high training and validation scores, indicating overfitting and poor generalization. In contrast, applying SMOTE only to the training data provided a more realistic estimate of performance on the test dataset, suggesting better generalization and less overfitting. Given this observation, the rest of the findings are based on using SMOTE resampling exclusively on the training data.

Additionally, data analysis revealed that there were no significant linear relationships between the predictor input features, suggesting that the relationships among features are complex and non-linear. This insight was factored into the final model recommendation.

Evaluation Summary:

After evaluating the performance of Logistic Regression, Decision Tree, and Support Vector Classifier (SVC) models through iterative hyperparameter tuning and decision threshold adjustments, the following findings stand out based on the need to accurately predict client subscriptions and minimize unnecessary contacts:
1. Best Performing Models:
    * Hyperparameter-Tuned Decision Tree: Achieved the highest F1 score, which is essential for balancing precision and recall—key for both accurately predicting client subscriptions and minimizing unnecessary contacts. This model also has the advantage of faster training and prediction times, making it suitable for real-time applications.
    * Basic SVC Model: Initially demonstrated the best balance between F1 score, PR AUC, and ROC AUC, contributing to accurate predictions. However, fine-tuning the decision threshold resulted in a lower F1 score, and its longer training time makes it less ideal for deployment.
    * Hyperparameter-Tuned Logistic Regression: While offering the best PR AUC and ROC AUC scores, which reflect strong performance in distinguishing between classes, its F1 score is slightly lower than the Decision Tree. It still performs well in reducing false positives but is slower to train and predict.
2. Impact of Threshold Tuning:
    * Fine-tuning the decision thresholds for all models to maximize the F1 score resulted in lower F1 scores, suggesting that this step does not improve performance in terms of reducing false positives or enhancing predictive accuracy and should be discarded.
3. Training Speed and Efficiency:
    * The Decision Tree model is the fastest to train and predict, an important consideration when scaling up predictions for large datasets or when quick decisions are needed for client subscription campaigns.
    * The basic SVC model takes considerably longer to train and predict, making it less feasible for time-sensitive applications, despite its initial strong performance in client subscription prediction.
    * The Logistic Regression model performs well in terms of PR AUC and ROC AUC but lags behind in training speed compared to the Decision Tree.
4. Interpretability:
    * Both the Decision Tree and Logistic Regression models offer clear insights into feature importance and decision-making processes, which is helpful for understanding how the model is predicting client subscriptions. The SVC model, while effective in some cases, lacks intuitive interpretability, making it harder to explain predictions to stakeholders.
5. Feature Importance in Decision Tree vs. Logistic Regression:
    * Decision Tree Model: Features like "duration" (0.410) and "euribor3m" (0.346) are straightforward indicators of client subscription likelihood. The model’s structure helps minimize unnecessary contacts by focusing on these critical factors, ensuring that marketing efforts are effectively targeted.
    * Logistic Regression Model: Provides detailed coefficients for features such as "Cons. Price Index" (2.707556) and "Month_Oct" (2.433611), which indicate how various factors influence subscription probability. Positive coefficients signal high likelihood, while negative coefficients help avoid less promising contacts. This nuanced view aids in optimizing marketing strategies by targeting high-potential clients and reducing unnecessary outreach.

Recommendation: 

Deploy the hyperparameter-tuned Decision Tree model in Production. It provides the best balance between F1 score, computational efficiency, and accurate prediction of client subscriptions while minimizing unnecessary contacts. Although the Logistic Regression model has better PR AUC and ROC AUC scores, the Decision Tree’s superior speed and F1 score with its ability to handle complex and non-linear relationships among predictors make it the more practical choice for deployment. The Decision Tree’s focus on critical features like "duration" and "euribor3m," alongside the Logistic Regression model’s detailed coefficient insights, ensures alignment with both high predictive performance and effective contact reduction strategies.

Feature Importance in hyperparameter-tuned Decision Tree v.s. Logistic Regression

To help optimize marketing strategies, focus on the key features and interactions that drive client subscriptions by targeting high-potential clients and reducing unnecessary outreach.

To optimize marketing strategies, focus on the key features and interactions driving client subscriptions:

1. Importance and Coefficient Interpretation:

* Decision Tree Model:

	Top Features:
		Duration (0.410): Highest importance, indicating that the length of the call is a strong predictor of client subscription likelihood.
		Euribor3m (0.346): Significant predictor, reflecting the impact of the 3-month Euribor rate on subscription probability.
		Month_Apr (0.051) and Cons. Conf. Index (0.051): Moderate importance, showing some influence based on the month of contact and consumer confidence.
	Predictive Performance: The model’s ability to accurately classify clients and minimize unnecessary contacts is driven by its high-importance features. It efficiently directs marketing efforts by making clear decisions based on feature splits.
	Minimizing Contacts: By focusing on features like "duration" and "euribor3m," the Decision Tree helps to identify clients more likely to subscribe, thereby reducing the number of unnecessary contacts.

* Logistic Regression Model:

	Top Features with Positive Coefficients:
		Cons. Price Index (2.707556): High positive coefficient, indicating a strong positive effect on the likelihood of subscription. Higher values of this feature significantly increase the probability of subscription.
		Month_Oct (2.433611): High positive coefficient, showing that contacts made in October are strongly associated with a higher likelihood of subscription.
		Cons. Conf. Index^6 (2.380486): A high positive coefficient on this transformed feature suggests that very high levels of consumer confidence have a strong positive effect on subscription likelihood.
		Cons. Price Index^2 (1.593425): Indicates that higher squared values of the cons. price index contribute positively, showing that increases in this feature’s squared term further raise the likelihood of subscription.
	Top Features with Negative Coefficients:
		Cons. Price Index^6 (-2.886025): High negative coefficient, suggesting that very high values of this feature decrease the likelihood of subscription. Extreme values in this feature can lower subscription probabilities.
		Month_May (-1.733885): Negative coefficient shows that contacts made in May are associated with a lower probability of subscription, indicating this month is less favorable for successful subscriptions.
		Duration^2 Cons. Price Index^4 (-1.525438): Complex interaction where higher values in both the duration and cons. price index raised to the fourth power lead to a decreased likelihood of subscription, suggesting that very long calls combined with high cons. price index values can negatively impact subscription chances.
	Predictive Performance: The model’s coefficients provide insights into how each feature and its transformations influence subscription probabilities. For instance:
		Cons. Price Index (2.707556) and Month_Oct (2.433611) have strong positive effects, meaning clients with higher cons. price index values or those contacted in October are more likely to subscribe.
		Cons. Price Index^2 (1.593425) further amplifies the effect of this feature, indicating that increases in the cons. price index have a compounded positive impact on subscription probability.
	On the other hand:
		Cons. Price Index^6 (-2.886025) and Month_May (-1.733885) show strong negative effects, helping to identify and avoid less promising clients, thus improving targeting efficiency.
		Duration^2 Cons. Price Index^4 (-1.525438) demonstrates that extreme values in both features combined reduce subscription likelihood, guiding the model to avoid these less favorable scenarios.


