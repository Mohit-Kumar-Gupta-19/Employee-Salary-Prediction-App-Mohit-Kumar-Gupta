# **💼 Employee Salary Prediction & Classification**

## **✨ Project Overview**

This project presents an advanced machine learning solution designed to predict whether an individual's annual income exceeds $50,000 or not, based on various demographic and professional attributes. Leveraging the UCI Adult Census Dataset, this application demonstrates a complete end-to-end machine learning pipeline, from data preprocessing and exploratory data analysis to model training, evaluation, and deployment as an interactive web application.

The core of this project is a robust classification model that provides accurate insights into salary brackets, which can be valuable for socio-economic analysis, policy-making, or even personal career planning.

**Live Demo:** [Access the deployed application here\!](https://employee-salary-prediction-app-mohit-kumar-gupta-djgp4pmgtkau9.streamlit.app/)

## **🚀 Features**

* **Interactive Web Application:** Built with Streamlit, providing a user-friendly interface for individual predictions.  
* **Batch Prediction:** Upload a CSV file for bulk predictions and download the results.  
* **Comprehensive Data Preprocessing:** Handles missing values, outliers, and categorical feature encoding.  
* **Multiple Model Evaluation:** Compares the performance of various classification algorithms (Logistic Regression, Random Forest, K-Nearest Neighbors, Support Vector Machine, Gradient Boosting).  
* **Best Model Selection:** Automatically identifies and utilizes the best-performing model (Gradient Boosting Classifier) for predictions.  
* **Detailed Performance Metrics:** Displays accuracy, precision, recall, F1-score, and a confusion matrix for model transparency.  
* **Clear Visualizations:** Uses matplotlib and seaborn for insightful data visualizations and model performance comparison.

## **📊 Dataset**

The project utilizes the **UCI Adult Census Dataset**, which contains over 48,000 instances with 14 attributes. These attributes include:

* age: Age of the individual.  
* workclass: Type of employer (e.g., Private, Self-emp-not-inc, Federal-gov).  
* fnlwgt: Final weight (census-specific, represents the number of people the census believes the entry represents).  
* education: Highest level of education achieved (e.g., Bachelors, HS-grad, Some-college).  
* educational-num: Numerical representation of education level.  
* marital-status: Marital status (e.g., Married-civ-spouse, Never-married, Divorced).  
* occupation: Type of occupation (e.g., Tech-support, Craft-repair, Exec-managerial).  
* relationship: Relationship within the household (e.g., Husband, Own-child, Not-in-family).  
* race: Race of the individual (e.g., White, Black, Asian-Pac-Islander).  
* gender: Gender of the individual (Male, Female).  
* capital-gain: Capital gains from investments.  
* capital-loss: Capital losses from investments.  
* hours-per-week: Number of hours worked per week.  
* native-country: Country of origin.  
* income: Target variable, indicating if income is \>50K or \<=50K.

## **🛠 Technology Stack**

* **Python:** Core programming language.  
* **Pandas:** For data manipulation and analysis.  
* **NumPy:** For numerical operations.  
* **Scikit-learn:** For machine learning model development, training, and evaluation.  
* **Matplotlib & Seaborn:** For data visualization.  
* **Streamlit:** For creating the interactive web application.  
* **Joblib:** For saving and loading the trained machine learning model.

## **⚙️ Installation & Setup**

To run this project locally, follow these steps:

1. **Clone the repository:**  
   git clone https://github.com/Mohit-Kumar-Gupta-19/Employee-Salary-Prediction-App-Mohit-Kumar-Gupta.git  
   cd Employee-Salary-Prediction-App-Mohit-Kumar-Gupta

2. **Create a virtual environment (recommended):**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows: \`venv\\Scripts\\activate\`

3. **Install the required libraries:**  
   pip install \-r requirements.txt

   *(If requirements.txt is not provided, you can generate it using pip freeze \> requirements.txt after installing the necessary libraries manually, or install them one by one: pandas, scikit-learn, streamlit, matplotlib, seaborn, joblib)*  
4. Download the dataset:  
   Ensure you have the adult.csv (or adult 3.csv if that's the primary) dataset in the root directory of your project. You can typically find this dataset on the UCI Machine Learning Repository.  
5. **Run the Streamlit application:**  
   streamlit run app.py

   This command will open the application in your web browser.

## **📈 Model Performance**

The project evaluates several machine learning algorithms to determine the most effective one for salary classification. The **Gradient Boosting Classifier** emerged as the best-performing model, achieving a high accuracy score on the test set.

| Algorithm | Accuracy | Status |
| :---- | :---- | :---- |
| Gradient Boosting | 85.71% | ✅ Best |
| Random Forest | 85.34% | 🥈 Second |
| Logistic Regression | 79.64% | 🥉 Third |
| SVM | 78.84% | 4th |
| KNN | 77.04% | 5th |

## **🤝 Contribution**

This project was crafted with dedication by **Mohit Kumar Gupta** as a contribution to the **Edunet Foundation IBM Internship Project**.

## **📄 License**

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

Feel free to explore the code, contribute, or use it as a reference for your own machine learning projects\!
