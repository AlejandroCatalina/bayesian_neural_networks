import matplotlib
matplotlib.use("Agg")
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg


def plot_heldout_prediction(input_val,
                            y_val,
                            mu_val,
                            sigma_val,
                            fname=None,
                            n=1,
                            title=""):
    """Save a PNG plot visualizing posterior uncertainty on heldout data.
  Args:
    input_val: input locations of heldout data.
    y_val: heldout target.
    mu_val: predictive mean.
    sigma_val: predictive standard deviation.
    fname: Python `str` filename to save the plot to, or None to show.
    title: Python `str` title for the plot.
  """
    fig = figure.Figure(figsize=(9, 3 * n))
    canvas = backend_agg.FigureCanvasAgg(fig)
    for i in range(n):
        ax = fig.add_subplot(n, i + 1, 1)
        ax.plot(input_val, y_val, label='True data')
        ax.plot(input_val, mu_val, label='Predictive mean')
        lower = mu_val - 1.96 * sigma_val
        upper = mu_val + 1.96 * sigma_val
        ax.fill_between(
            input_val, lower, upper, label='95% confidence interval')

    plt.legend()
    fig.suptitle(title)
    fig.tight_layout()

    if fname is not None:
        canvas.print_figure(fname, format="png")
        print("saved {}".format(fname))
