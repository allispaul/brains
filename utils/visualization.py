"""Tools for data visualization."""

def visualize_spectrogram(item, *, figsize=(15, 12), label_spacing=6, colorbars=True):
    """Visualize a spectrogram.
    
    Args:
      item: Spectrogram to visualize, as a np.array or torch.Tensor of shape
        [400, *].
      figsize (default (10, 8)): Size of matplotlib figure.
      label_spacing (default 6): Spacing between y-axis labels.
      colorbars (default True): Whether to include colorbars.
    """
    # Thanks to https://www.kaggle.com/code/alejopaullier/hms-efficientnetb0-pytorch-train#%7C-Utils-%E2%86%91
    item = torch.clamp(item, 1e-4, 1e7)
    item = torch.log(item)
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axsflat = axs.flatten()
    regions = ["LL", "RL", "LP", "RP"]
    freqs = [0.59, 0.78, 0.98, 1.17, 1.37, 1.56, 1.76, 1.95, 2.15, 2.34, 2.54,
             2.73, 2.93, 3.13, 3.32, 3.52, 3.71, 3.91, 4.1, 4.3, 4.49, 4.69,
             4.88, 5.08, 5.27, 5.47, 5.66, 5.86, 6.05, 6.25, 6.45, 6.64, 6.84,
             7.03, 7.23, 7.42, 7.62, 7.81, 8.01, 8.2, 8.4, 8.59, 8.79, 8.98,
             9.18, 9.38, 9.57, 9.77, 9.96, 10.16, 10.35, 10.55, 10.74, 10.94,
             11.13, 11.33, 11.52, 11.72, 11.91, 12.11, 12.3, 12.5, 12.7, 12.89,
             13.09, 13.28, 13.48, 13.67, 13.87, 14.06, 14.26, 14.45, 14.65, 14.84,
             15.04, 15.23, 15.43, 15.63, 15.82, 16.02, 16.21, 16.41, 16.6, 16.8,
             16.99, 17.19, 17.38, 17.58, 17.77, 17.97, 18.16, 18.36, 18.55, 18.75,
             18.95, 19.14, 19.34, 19.53, 19.73,]
    for i in range(4):
        img = axsflat[i].imshow(item[i*100:(i+1)*100], aspect="auto", origin="lower")
        axsflat[i].set_title(regions[i])
        axsflat[i].set_yticks(np.arange(0, len(freqs), label_spacing))
        axsflat[i].set_yticklabels(freqs[::label_spacing])
        if colorbars:
            cbar = fig.colorbar(img, ax=axsflat[i])
            cbar.set_label('Log(Value)')
        axsflat[i].set_ylabel("Frequency (Hz)")
        axsflat[i].set_xlabel("Time")
    plt.tight_layout()