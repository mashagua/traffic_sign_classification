import pickle
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(23)

def load_traffic_sign_data(training_file):
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    X_train, y_train = train['features'], train['labels']

    return X_train, y_train

def show_classes_distribution(n_classes,y_train):
    train_distribution=np.zeros(n_classes)
    for c in range(n_classes):
        train_distribution[c]=np.sum(y_train==c)/(y_train.shape[0])
    fig, ax = plt.subplots()
    col_width = 1
    bar_train = ax.bar(np.arange(n_classes), train_distribution, width=col_width, color='r')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Class Label')
    ax.set_title('Distribution')
    ax.set_xticks(np.arange(0, n_classes, 5) + col_width)
    ax.set_xticklabels(['{:02d}'.format(c) for c in range(0, n_classes, 5)])
    plt.show()

if __name__ == "__main__":
    X_train, y_train = load_traffic_sign_data('data/train.p')
    n_train=y_train.shape[0]
    print("the number of training data= ",n_train)
    print('the size of image =',X_train[0].shape)
    print('the number of classes ',np.unique(y_train).shape[0])
    show_classes_distribution(np.unique(y_train).shape[0], y_train)