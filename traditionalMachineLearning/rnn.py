import pickle

model_pkl_file = "rnn_classifier_model.pkl"

with open(model_pkl_file, "wb") as file:
    pickle.dump(model, file)
