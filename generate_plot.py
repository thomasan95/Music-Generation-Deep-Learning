import argparse
import utilities as utils
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, default="", help="Path to pickle file of losses")
parser.add_argument("-s", "--save_file", type=str, default="", help="file to save as. Will save under results")
parser.add_argument("-uc", "--update_check", type=int, default=1000, help="how many iterations per update")
parser.add_argument("-t", "--title", type=str, default="", help="Specify title of plot")
args = parser.parse_args()


def generate_plot(file, save_file="", title=""):
    '''
    Function for visualizing training and validation loss over training cycle

    File must be given for the function to run. If no save_file given, then file will be saved
    with same name as File.

    If save_file is given, the plot will be saved inside results with the given name

    :param file: if ran stand alone it is from args.file, path to specified file
    :type file: str
    :param save_file: if ran stand alone, it is from args.save_file, name to save the file as
    :type save_file: str
    :param title: title of the plot to generate
    :type title: str
    :return: None
    '''

    assert isinstance(file, str)
    assert isinstance(save_file, str)

    if file == "":
        print("Please specify file path")
        return -1

    assert (file[-2:] == '.p')  # Must specify that the file is a pickled file with the .p!

    losses = utils.load_files(file)
    epochs_train = len(losses['train'])
    epochs_val = len(losses['valid'])

    if epochs_train == epochs_val:
        plt.plot(list(range(epochs_train)), losses['train'], label='Train Loss')
        plt.plot(list(range(epochs_val)), losses['valid'], label='Valid Loss')
        plt.xlabel("Every %d Training Iterations" % args.update_check)
        plt.ylabel("Loss")
        if title == "":
            plt.title("Loss over Training Cycle")
        else:
            plt.title(title)
        plt.legend(loc='upper right')

    else:
        plt.subplot(2, 1, 1)
        plt.plot(list(range(epochs_train)), losses['train'], label='Train Loss')
        plt.xlabel("Training Iterations")
        plt.ylabel("Loss")
        plt.title("Train Loss over Training Cycle")

        plt.subplot(2, 1, 2)
        plt.plot(list(range(epochs_val)), losses['valid'], label='Valid Loss')
        plt.xlabel("Every %d Training Iterations" % args.update_check)
        plt.ylabel("Loss")
        plt.title("Valid Loss over Training Cycle")
        plt.legend(loc='upper right')
    if save_file == "":
        save_file = file[:-2]
        plt.savefig(save_file+'.png')
    else:
        plt.savefig('results/' + save_file + '.png')

    plt.show()


if __name__ == "__main__":
    generate_plot(args.file, args.save_file, args.title)
