import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from keras.models import Sequential
from keras.layers import Dense, Dropout


def choose_from_possible_models(full_name, possible_models):
	if full_name in possible_models:
		return full_name
	else:
		possible_models = sorted(possible_models, key=lambda name: len(name), reverse = True)
		for name in possible_models:
			if name in full_name:
				return name
	raise NotImplementedError("Requested model named " + full_name + " does not exist! Or The model name was not added in possible_models list.")

def get_ensemble_network_model(full_name, input_dim, output_dim = 1):
	model = Sequential()

	possible_models = ["nowdeepn_model"]
	name = choose_from_possible_models(full_name, possible_models)

	if "nowdeepn_model" == name:
		print("Chosen model: ensemble_model_3")
		model = Sequential()
		model.add(Dense(200, input_dim=input_dim, activation='relu'))
		model.add(Dense(2000, activation='relu'))
		model.add(Dropout(0.1))
		model.add(Dense(500, activation='relu'))
		model.add(Dense(500, activation='relu'))
		model.add(Dense(500, activation='relu'))
		model.add(Dense(500, activation='relu'))
		model.add(Dense(500, activation='relu'))
		model.add(Dense(100, activation='relu'))
		model.add(Dense(100, activation='relu'))
		model.add(Dense(100, activation='relu'))
		model.add(Dense(100, activation='relu'))
		model.add(Dense(100, activation='relu'))

	else:
		raise NotImplementedError("Request ensemble network model does not exist!! (yet)")

	model.add(Dense(output_dim, activation='linear'))

	return model