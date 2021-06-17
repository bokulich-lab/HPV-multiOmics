import math
import seaborn as sns
from matplotlib.pylab import plt
import matplotlib as mpl
import re


def biplot(features, coeff, coeff_md, metabolite_colors, color=None,
           feature_labels=None, coeff_labels=None, minradius=1, whitelist=[]):
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = features.iloc[:, 0]
    ys = features.iloc[:, 1]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    ax.scatter(xs * scalex, ys * scaley, c=color, marker='.', s=100, alpha=1)
    if feature_labels is not None:
        for x, y, label in zip(xs, ys, feature_labels):
            # is point beyond radius minradius of origin?
            # or on whitelist of taxa of interest
            if math.sqrt(x ** 2 + y ** 2) > minradius:
                if whitelist is not None and label not in whitelist:
                    continue
                if x > 0:
                    tx = 0.1
                    ha = 'left'
                else:
                    tx = -0.1
                    ha = 'right'
                if y > 0:
                    ty = 0.1
                    va = 'bottom'
                else:
                    ty = -0.1
                    va = 'top'
                ax.annotate(label, (x, y), xytext=(tx, ty),
                            textcoords="offset points",
                            horizontalalignment=ha,
                            verticalalignment=va, alpha=0.7)

    # plot constraining variables
    groupnames = set()
    for i in range(coeff.shape[0]):
        # assign label color
        groupname = coeff_md.loc[coeff.index[i]]
        groupnames.add(groupname)
        colorit = metabolite_colors[groupname]
        # is point beyond radius minradius of origin?
        if (math.sqrt(coeff.iloc[i, 0] ** 2 + coeff.iloc[i, 1] ** 2)
                > minradius):
            scalex = 1.0/(coeff[0].max() - coeff[0].min())
            scaley = 1.0/(coeff[1].max() - coeff[1].min())
            plt.arrow(0, 0, coeff.iloc[i, 0] * scalex,
                      coeff.iloc[i, 1] * scaley,
                      color='grey', alpha=0.4,
                      linewidth=0.75, head_width=0.01)
            if coeff_labels is not None:
                plt.text(coeff.iloc[i, 0] * scalex * 1.15,
                         coeff.iloc[i, 1] * scaley * 1.15,
                         coeff_labels[i],
                         color=colorit, ha='center', va='center',
                         fontsize=8, alpha=0.7)

    # add metabolite color legend
    # make white blobs so that only text gets plotted
    blobs = [mpl.patches.Patch(facecolor='w', edgecolor='w')
             for i in groupnames]
    legend1 = plt.legend(blobs, metabolite_colors.keys(), title=coeff_md.name,
                         ncol=1, bbox_to_anchor=(0.92, 0.4),
                         bbox_transform=plt.gcf().transFigure)
    plt.gca().add_artist(legend1)
    # color legend text according to label color
    for text in legend1.get_texts():
        text.set_color(metabolite_colors[text.get_text()])

    # plt.xlim(xs.min() * 1.5, xs.max() * 1.5)
    # plt.ylim(-1.5,1.5)
    plt.xlabel("Axis 1")
    plt.ylabel("Axis 2")

    plt.grid()
    return fig


def _format_taxonomy(taxa, level):
    '''
    taxa: array-like or pd.Series of semicolon-delimited taxonomy strings.
    '''
    new_taxa = []
    for i in taxa:
        t = re.sub('[kpcofgs]__', '', i).replace(
            '_', ' ').rstrip('; ').split(';')
        t = [i.strip(' ') for i in t if len(i) > 2]
        try:
            new_taxa.append(t[level])
        except IndexError:
            new_name = 'Unknown {0}'.format(t[-1].strip('[]'))
            if len(t[-1]) < level:
                new_name += ' ({0})'.format(t[-2].strip('[]'))
            new_taxa.append(new_name)
    return new_taxa


def biplot_from_rhapsody(cca_res, taxa, metabolite_md, metabolite_colors,
                         palette="Set2", level=6, minradius=1, label_level=3,
                         feature_labels=None, mincount=10, whitelist=[]):
    # relabel features
    taxa = taxa['Taxon'].reindex(cca_res.features.index)
    if feature_labels is not None:
        feature_labels = _format_taxonomy(taxa, level)

    # set color scheme
    classes = taxa.apply(lambda x: x.split(';')[:label_level][-1])
    color_labels = classes.value_counts()
    color_labels = color_labels.where(color_labels >= mincount).dropna()
    # List of RGB triplets
    # add an additional value for "other"
    rgb_values = sns.color_palette(palette, len(color_labels) + 1)
    # Map label to RGB
    palette = dict(zip(list(color_labels.index) + ['Other'], rgb_values))
    # convert low-abundance taxa to "other"
    classes = classes.apply(
        lambda x: 'Other' if x not in color_labels.index else x)
    # Finally use the mapped values
    colors = classes.map(palette)

    coeff_labels = cca_res.samples.index

    fig = biplot(cca_res.features, cca_res.samples, metabolite_md,
                 metabolite_colors, color=colors,
                 feature_labels=feature_labels,
                 coeff_labels=coeff_labels, minradius=minradius,
                 whitelist=whitelist)

    recs = [mpl.patches.Rectangle((0, 0), 1, 1, fc=palette[k])
            for k in palette.keys()]
    plt.legend(recs, list(color_labels.index) +
               ['Other'], loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    return fig
