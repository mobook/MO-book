
def scatter_comparison(X, y, y_pred):
    """
    Creates scatter plots comparing actual and predicted outcomes for both training and test sets.

    Parameters:
    X : DataFrame
        Feature matrix as a DataFrame.
    y : Series
        Actual target vector as a Series.
    y_pred : Series
        Predicted target vector as a Series.

    Returns:
    None
    """

    xmin, ymin = X.min()
    xmax, ymax = X.max()
    xlim = [xmin - 0.05 * (xmax - xmin), xmax + 0.05 * (xmax - xmin)]
    ylim = [ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin)]

    # Plot training and test sets
    labels = ["genuine", "counterfeit"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    scatter_labeled_data(X, y, labels, ["g", "r"], ax=ax[0], xlim=xlim, ylim=ylim, title="Actual")
    scatter_labeled_data(X, y_pred, labels, ["c", "m"], ax=ax[1], xlim=xlim, ylim=ylim, title="Prediction")

    # Plot actual positives and actual negatives
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    scatter_labeled_data(X[y > 0], y_pred[y > 0], ["true positive", "false negative"], 
                         ["c", "m"], xlim=xlim, ylim=ylim, ax=ax[0], title="Actual Positives")
    scatter_labeled_data(X[y < 0], y_pred[y < 0], ["false positive", "true negative"], 
                         ["c", "m"], xlim=xlim, ylim=ylim, ax=ax[1], title="Actual Negatives")

    
