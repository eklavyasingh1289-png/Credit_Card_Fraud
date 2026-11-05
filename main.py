import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,r2_score,confusion_matrix,roc_auc_score,precision_recall_curve

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

df = pd.read_csv("creditcard_csv.csv")
df["Class"] = df["Class"].str.strip().str.strip("'").map({'1':1,"0":0})

x = df.iloc[:,0:30]
y = df.iloc[:,-1]

X_train,X_test,Y_train,Y_test = train_test_split(x,y, test_size=.15, random_state=42, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_test  = scaler.transform(X_test)

smote = SMOTE(random_state=42)
x_train_sm, Y_train_sm = smote.fit_resample(x_train,Y_train)

model = XGBClassifier(n_estimators=400, max_depth=5, random_state=42)
model.fit(x_train_sm, Y_train_sm)
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

print("Accuracy:",accuracy_score(Y_test,y_pred))
print("ROC-AUC:", roc_auc_score(Y_test, y_prob))

cm = confusion_matrix(Y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()