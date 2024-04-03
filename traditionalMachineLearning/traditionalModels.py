# Random Forest
# CNN
# RNN

# load model from pickle file
with open(model_pkl_file, "rb") as file:
    model = pickle.load(file)

# evaluate model
y_predict = model.predict(X_test)

# check results
print(classification_report(y_test, y_predict))
