import pickle

model_path = input('model path: ', 's')
model = pickle.load(open(model_path), 'rb')

