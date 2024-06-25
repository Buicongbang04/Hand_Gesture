import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=0, shuffle=True)

model = RandomForestClassifier()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_score = accuracy_score(Y_test, Y_pred)

print(f"Accuracy: {acc_score * 100:.2f}%")
pickle.dump({'model': model}, open('./model.pickle', 'wb'))