import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(tp, fn, fp, tn):
    """
    Plot a confusion matrix from four values.
    
    Matrix layout:
            Pred 0   Pred 1
    True 0     TN       FP
    True 1     FN       TP
    """

    cm = np.array([[tn, fp],
                   [fn, tp]])

    fig, ax = plt.subplots()

    # Show matrix
    cax = ax.imshow(cm, cmap="Blues")

    # Colorbar
    plt.colorbar(cax)

    # Axis labels
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Correct tick positions
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    # Correct tick labels
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])

    # Add text values in cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()

# plot_confusion_matrix(27342, 30147, 10326, 194925)
#plot_confusion_matrix(41105, 9881, 22221, 189533)
#plot_confusion_matrix(21244, 3536, 2253, 113127)

plot_confusion_matrix(22745, 29640, 0, 192895)

'''[[1338383    4837]
 [ 142165  266615]]'''
 
'''
[[113127 2253]
[3536   21244]]
'''

'''
Accuracy : 0.8792
Precision: 1.0000
Recall   : 0.4342
F1       : 0.6055

Confusion Matrix:
 [[192895      0]
 [ 29640  22745]]
'''