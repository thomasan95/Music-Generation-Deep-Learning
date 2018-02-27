import argparse
import utilities as utils
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, default="", help="Path to pickle file of losses")
parser.add_argument("-s", "--save_file", type=str, default="losses", help="file to save as. Will save under results")
parser.add_argument("-uc", "--update_check", type=int, default=1000, help="how many iterations per update")
args = parser.parse_args()


def generate_plot(file, save_file):
    '''
    Function for visualizing training and validation loss over training cycle
    :param file: if ran stand alone it is from args.file, path to specified file
    :type file: str
    :param save_file: if ran stand alone, it is from args.save_file, name to save the file as
    :type save_file: str
    :return: None
    '''
    if file == "":
        print("Please specify file path")
        return -1

    losses = utils.load_files(file)
    epochs_train = len(losses['train'])
    epochs_val = len(losses['valid'])

    if epochs_train == epochs_val:
        plt.plot(list(range(epochs_train)), losses['train'], label='Train Loss')
        plt.plot(list(range(epochs_val)), losses['valid'], label='Valid Loss')
        plt.xlabel("Every %d Training Iterations" % args.update_check)
        plt.ylabel("Loss")
        plt.title("Loss over Training Cycle")
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

    plt.savefig('results/' + save_file + '.png')
    plt.show()


if __name__ == "__main__":
    generate_plot(args.file, args.save_file)
