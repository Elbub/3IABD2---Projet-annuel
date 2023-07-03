import lib


images_dimensions = 50 * 50 * 3

inputs_folders = ["./database/resized_dataset/asie_sud_est","./database/resized_dataset/rome_grece"]
# inputs_folders = ["./database/resized_dataset/amerique_sud","./database/resized_dataset/asie_sud_est","./database/resized_dataset/rome_grece"]

inputs, labels = lib.read_dataset(inputs_folders)
# print(inputs[:10], "...", inputs[-10:])
# print("--------------")
# print(labels[:10], "...", labels[-10:])
# print("--------------")


layers = [images_dimensions, len(inputs_folders)]

mlp_model = lib.generate_multi_layer_perceptron_model(layers)
print(mlp_model)


learning_rate = 0.02
number_of_epochs = 1

mlp_model = lib.train_multi_layer_perceptron_model(True, # is_classification
                                                   layers,
                                                   inputs,
                                                   labels,
                                                   mlp_model,
                                                   learning_rate,
                                                   number_of_epochs)
print(mlp_model)


predicted_dataset = lib.predict_with_multi_layer_perceptron_model(True,
                                                                  layers,
                                                                  inputs,
                                                                  mlp_model)
print(predicted_dataset)