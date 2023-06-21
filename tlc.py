

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_pdf import PdfPages
import csv
import io
import os


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def MultiPie(mydf: pd, mytitle) -> plt:
    # défini le nb de colonnes
    if mydf.shape[1] == 1:
        ncols = 1
    else:
        ncols = 3

    # calcul du nb de ligne
    if mydf.shape[1] % ncols == 0:
        nrows = int(mydf.shape[1]/ncols)
    else:
        nrows = int(mydf.shape[1]/ncols)+1

    fig, axs = plt.subplots(figsize=(20, 20),
                            nrows=nrows,
                            ncols=ncols,
                            squeeze=False)
    fig.subplots_adjust(top=0.8)

    colors = ['#49DC3A',
              '#D0F741',
              '#5A45C3',
              '#3B7EBA',
              '#FFDA43',
              '#FFB143',
              '#FC424B',
              '#C634AF']
    for ax, (title, values) in zip(axs.flat, mydf.items()):

        ax.pie(values, labels=[k[0] for k, v in mydf.iterrows()],
               autopct=lambda p: f"{p:.2f}%  ({(p * sum(values)/100):,.0f})" if p > 0 else '',
               wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
               textprops={'color': "#8E8E90", 'weight': 'bold',
                          'fontsize': '14'},
               colors=colors)

        ax.set_title(mytitle, weight='bold',
                     size='medium',
                     position=(0.5, 1.1),
                     horizontalalignment='center',
                     verticalalignment='center')

    # récupère la légende et la place sur la figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,  bbox_to_anchor=(0, 0.8))

    # supprime les derniers graphes vides
    if mydf.shape[1] % ncols != 0:
        for n in range(ncols, mydf.shape[1] % ncols, -1):
            fig.delaxes(axs[nrows-1, n-1])

    fig.tight_layout(pad=2.0)
    # fig.show()
    return fig


def QPie(question, maxpage, pdf):
    mydf = df[occurences[question]['column']].value_counts().sort_index(ascending=True).to_frame()
    myfig = MultiPie(mydf, f"{question}: {occurences[question].get('title')}")
    pdf.savefig(myfig, bbox_inches="tight")


def labels_radar(columns: list) -> list:
    labels = []
    for str in columns:
        deb = str.find('>')
        label = multiligne(str[deb+1:], 30)
        labels.append(label)
    return labels


def multiligne(texte: str, size: int) -> str:
    mots = texte.split()
    resultat = ""
    ligne = ""
    for mot in mots:
        if len(ligne + mot) <= size:
            ligne += mot + " "
        else:
            resultat += ligne.strip() + "\n"
            ligne = mot + " "
    resultat += ligne.strip()
    return resultat


def Plot_bar(question, maxpage, pdf):
    df[occurences[question]['column']].replace(-999.0, np.NaN, inplace=True)
    categories = labels_radar(occurences[question]['column'])
    values = df[occurences[question]['column']].mean().tolist()

    fig = plt.figure(figsize=(10, 5))

    colors = ['#D0F741',
              '#49DC3A',
              '#3B7EBA',
              '#5A45C3',
              '#FFB143',
              '#FFDA43',
              '#C634AF',
              '#FC424B']

    # creating the bar plot
    plt.bar(categories,
            values,
            color=colors[1],
            width=0.4)

    plt.ylim((0, 5))
    plt.ylabel("Rating")
    plt.title(f"{question}: {occurences[question].get('title')}")
    plt.show()
    pdf.savefig(fig, bbox_inches="tight")


def generate_charts(change) -> str:
    """
    génère les graphique et retourne le nom de fichier en retour
    """
    global df
    global occurences

    if type(change['new']) is dict:
        infos = change['new']
        infos = list(infos.values())[0]
        input_file_name = infos['metadata']['name']
    elif type(change['new']) is tuple:
        infos = change['new'][0]
        input_file_name = infos['name']

    _temp = os.path.splitext(input_file_name)
    pdf_file_name = f"public.{_temp[0]}.pdf"

    content = infos['content']
    if type(content) is memoryview:
        content = bytes(content)
    content = io.StringIO(content.decode('utf-8'))

    try:
        headers = pd.read_csv(content,
                              index_col=0,
                              nrows=0,
                              encoding='utf-8-sig').columns.tolist()
        if 'Course' in headers:
            course_col = 'Course'
        else:
            course_col = 'Cours'
    except:
        raise UserWarning("\nProblem reading the file. Check the name of the .CSV file\nEnter the correct file name in the previous step.\nProblème de lecture du fichier. Vérifier le nom du fichier .CSV\nSaisir le bon nom de fichier à l'étape précédente.")

    content.seek(0)
    try:
        dialect = csv.Sniffer().sniff(content.read(1024), [',', ';'])
    except:
        raise UserWarning("Problème de lecture du fichier. Vérifier le nom du fichier .CSV\nSaisir le bon nom de fichier à l'étape précédente.")

    content.seek(0)
    df = pd.read_csv(content,
                     encoding='utf-8-sig',
                     delimiter=dialect.delimiter,
                     on_bad_lines='skip')
    # print(df.head())

    if len(df.columns) <= 2:
        raise UserWarning(f'\nProblem reading the CSV file. Inconsistent number of columns ({df.columns}).\nProblème de lecture du fichier CSV. Nombre de colonnes incohérent ({df.columns})')

    # on extrait les questions que l'on met dans un dictionnaire
    questions = [col[:3] for col in df.columns if re.search(r"^Q\d{2}", col)]

    # on cherche le nombre de colonne associé à une question
    occurences = {}
    for q in questions:
        if q in occurences:
            occurences[q]['nb'] += 1
        else:
            occurences[q] = {}
            occurences[q]['nb'] = 1

    # on garde que les questions
    for col in df.columns:
        if re.search(r"^Q\d{2}", col):
            indice_fin = col.find("-")
            if indice_fin == -1:
                indice_fin = len(col)
            occurences[col[:3]]['title'] = col[4:indice_fin]
            if 'column' in occurences[col[:3]]:
                occurences[col[:3]]['column'].append(col)
            else:
                occurences[col[:3]]['column'] = [col]

    for element in occurences.keys():
        if occurences[f"{element}"]['nb'] == 1:
            name = occurences[f"{element}"]['column'][0]
            """try:
                if df.dtypes[name] == 'O':
                    match = df[df[name].str.match(r"^[0-9]* :.*$") is True]
                else:
                    # la colonne est numérique
                    match = [1]
            except Exception as e:
                print(occurences)
                print(name)
                print(df[name].dtypes)
                print(df[name])
                raise (e)"""
            if df.dtypes[name] == 'O':
                match = df[df[name].str.match(r"^[0-9]* :.*$") == True]
            else:
                # la colonne est numérique
                match = [1]

            if len(match) > 0:
                occurences[element]['type'] = 'pie'
            else:
                occurences[element]['type'] = 'comments'
        elif occurences[f"{element}"]['nb'] == 2:
            occurences[element]['type'] = 'bar'
        else:
            occurences[element]['type'] = 'radar'

    # print(occurences)

    #
    # Crée un fichier pdf
    #
    pdf = PdfPages(pdf_file_name)

    for q in occurences:
        if occurences[q]['type'] == 'pie':
            # on dessine un pie
            QPie(q, 9, pdf)

        elif occurences[q]['type'] == 'bar':
            Plot_bar(q, 9, pdf)

        elif occurences[q]['type'] == 'radar':
            # remplace les -999 par NaN
            # print(occurences[q])
            df[occurences[q]['column']] = df[occurences[q]['column']].replace(-999.0, np.NaN)

            categories = labels_radar(occurences[q]['column'])
            val1 = (f"{q}: {occurences[q]['title']}",
                    [df[occurences[q]['column']].mean().tolist()])
            mydata = [categories, val1]
            N = len(mydata[0])

            theta = radar_factory(N, frame='polygon')
            spoke_labels = mydata.pop(0)
            fig, axs = plt.subplots(figsize=(20, 20),
                                    nrows=2,
                                    ncols=1,
                                    subplot_kw=dict(projection='radar'))
            fig.subplots_adjust(wspace=0.25,
                                hspace=0.25,
                                top=0.85,
                                bottom=0.1)
            colors = ['#D0F741',
                      '#49DC3A',
                      '#3B7EBA',
                      '#5A45C3',
                      '#FFB143',
                      '#FFDA43',
                      '#C634AF',
                      '#FC424B']

            for ax, (title, case_data) in zip(axs.flat, mydata):
                ax.set_rgrids([1, 2, 3, 4, 5])  # ici calculer le max de la colonne
                ax.set_title(title,
                             weight='bold',
                             size='medium',
                             position=(0.5, 1.1),
                             horizontalalignment='center',
                             verticalalignment='center')

            for d, color in zip(case_data, colors):
                ax.plot(theta, d, color=color)
                ax.fill(theta, d, facecolor=color, alpha=0.25)
                ax.set_varlabels(spoke_labels)
                ax.set_ylim(bottom=0, top=4)

            plt.yticks([1, 2, 3, 4], color="grey", size=10)

            # supprime le deuxième (vide)
            fig.delaxes(axs[1])

            fig.tight_layout(pad=2.0)
            plt.show()
            fig.savefig(f"{title}.svg")
            pdf.savefig(fig, bbox_inches="tight")

    pdf.close()
    return pdf_file_name
