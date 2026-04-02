"""
generate_model.py — run ONCE to create model.pkl
Run:  python generate_model.py
"""
import pickle, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(2024)
N = 6000
tv = np.random.randint(50,5000,N).astype(float)
T  = np.random.uniform(15,50,N)
H  = np.random.uniform(15,98,N)
pd_= np.random.randint(200,55000,N).astype(float)
hr = np.random.randint(0,24,N).astype(float)
we = np.random.randint(0,2,N).astype(float)
wc = np.random.randint(0,6,N).astype(float)
rq = np.random.randint(0,5,N).astype(float)
X  = np.column_stack([tv,T,H,pd_,hr,we,wc,rq])
sc = (tv/5000*40 + np.clip((T-20)/30,0,1)*20 + H/100*10 + pd_/55000*15
      + np.where((hr>=7)&(hr<=9),12,0) + np.where((hr>=17)&(hr<=19),12,0)
      + (1-we)*5 + np.where(wc>=3,8,0) + (4-rq)/4*10
      + np.random.normal(0,5,N))
lbl = np.where(sc>=62,"HIGH",np.where(sc>=38,"MEDIUM","LOW"))
X_tr,X_te,y_tr,y_te = train_test_split(X,lbl,test_size=.2,random_state=42,stratify=lbl)
model = GradientBoostingClassifier(n_estimators=200,max_depth=5,learning_rate=.08,random_state=42)
model.fit(X_tr,y_tr)
acc = accuracy_score(y_te,model.predict(X_te))
print(f"Accuracy: {acc*100:.1f}%")
print(classification_report(y_te,model.predict(X_te)))
with open("model.pkl","wb") as f: pickle.dump(model,f)
print("model.pkl saved.")
