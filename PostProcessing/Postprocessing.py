from datetime import date


def save_params(save_file_path, func_args):
    f = open(save_file_path, 'a')
    f.write("\nMODEL TRAINED WITH THE FOLLOWING PARAMETERS AND HYPERPARAMETERS\n\n")
    f.write("n_iter : " + str(func_args[3]["n_iter"]) + '\n')
    f.write("learning_rate : " + str(func_args[3]["learning_rate"]) + '\n')
    f.close()


def save_model(model, save_file_path, params_saved=None):
    f = open(save_file_path, 'w')  # create files if does not exist, writing in the file anyway
    f.write("Date : {}".format(date.today().strftime("%d/%m/%Y")) + "\n\n")
    f.write("MODEL\n\n")
    f.write("Parameter Theta value after training : ")
    f.write(str(model) + '\n')
    f.close()

    if params_saved:
        save_params(save_file_path, params_saved)
