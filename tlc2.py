

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
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from svglib.svglib import svg2rlg


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
    myfig.savefig(f"{remplacer_caracteres(occurences[question].get('title'))}.svg")
    pdf.savefig(myfig, bbox_inches="tight")
    plt.close(myfig)
    occurences[question]['imgs'] = [f"{remplacer_caracteres(occurences[question].get('title'))}.svg"]


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


def remplacer_caracteres(chaine) -> str:
    # Créer une expression régulière qui correspond à espace, virgule ou deux points
    pattern = r'[ ,:;/]'
    # Remplacer tous les correspondances par '_'
    nouvelle_chaine = re.sub(pattern, '_', chaine)
    return nouvelle_chaine


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


def create_pdf(pages=[], file_name='output.pdf'):
    """
    Create a PDF file with pages that contain SVG images resized to fit the page.

    Args:
        pages (list): A list of tuples, where each tuple contains the title of the page and a list of SVG file paths.
        file_name: PDF file name to create

    Returns:
        None
    """
    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(file_name, pagesize=A4,
                            bottomMargin=0,
                            topMargin=.5*inch)
    width, height = A4
    elements = []

    for _type, page_title, data in pages:
        if (_type == 'img') & (data is not None):
            elements.append(Paragraph(page_title, styles["Heading1"]))
            elements.append(Spacer(1, 12))
            if len(data) > 1:
                available_height = (height - (2 * inch)) / len(data)
            else:
                available_height = height - (2 * inch)

            available_width = width - (2 * inch)
            # print(page_title)
            # print(f"width:{width} height:{height} - avail. W:{available_width} avail. H:{available_height}")
            
            for idx, svg_file in enumerate(data):
                drawing = svg2rlg(svg_file)                    
                scale_factor = min(available_width / drawing.width, available_height / drawing.height)
                # print(f"drawing.height:{drawing.height} Scale:{scale_factor} - new w:{drawing.width * scale_factor} new H:{drawing.height * scale_factor}")
                drawing.width *= scale_factor
                drawing.height *= scale_factor
                drawing.scale(scale_factor, scale_factor)
                elements.append(KeepTogether(drawing))
                if idx < len(svg_file):
                    elements.append(Spacer(1, 12))
            # Add a page break
            elements.append(PageBreak())
        elif _type == 'txt':
            elements.append(Paragraph(page_title, styles["Heading1"]))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(data, styles["Normal"]))
            # Add a page break
            elements.append(PageBreak())

    doc.build(elements)


def generate_charts(change) -> str:
    """
    génère les graphique et retourne le nom de fichier en retour
    """
    global df
    global occurences

    #if type(change['new']) is dict: # 
    if isinstance(change['new'], dict):
        infos = change['new']
        infos = list(infos.values())[0]
        input_file_name = infos['metadata']['name']
    # elif type(change['new']) is tuple:
    elif isinstance(change['new'], tuple):
        infos = change['new'][0]
        input_file_name = infos['name']

    _temp = os.path.splitext(input_file_name)
    pdf_file_name = f"public.{_temp[0]}.pdf"

    content = infos['content']
    if isinstance(content, memoryview):
        content = bytes(content)
    content = io.StringIO(content.decode('utf-8'))

    # détermine la langue
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

    # cherche le séparateur
    content.seek(0)
    try:
        dialect = csv.Sniffer().sniff(content.read(1024), [',', ';'])
    except:
        raise UserWarning("Problème de lecture du fichier. Vérifier le nom du fichier .CSV\nSaisir le bon nom de fichier à l'étape précédente.")

    # charge le fichier
    content.seek(0)
    df = pd.read_csv(content,
                     encoding='utf-8-sig',
                     delimiter=dialect.delimiter,
                     on_bad_lines='skip')
    # print(df.head())

    if len(df.columns) <= 2:
        raise UserWarning(f'\nProblem reading the CSV file. Inconsistent number of columns ({df.columns}).\nProblème de lecture du fichier CSV. Nombre de colonnes incohérent ({df.columns})')

    # on extrait les questions que l'on met dans un dictionnaire
    # questions = [col[:3] for col in df.columns if re.search(r"^Q\d{2}", col)]
    questions = [re.match(r"^(Q\d*)_", elem).group(1) for elem in df.columns if re.match(r"^(Q\d*)_", elem)]

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
        if re.match(r"^(Q\d*)_", col):
            num_q = re.match(r"^(Q\d*)_", col).group(1)
            indice_fin = col.find("-")
            if indice_fin == -1:
                indice_fin = len(col)
            occurences[num_q]['title'] = col[len(num_q)+1:indice_fin]
            if 'column' in occurences[num_q]:
                occurences[num_q]['column'].append(col)
            else:
                occurences[num_q]['column'] = [col]

    # on détermine le type de graphique
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

    # on crée le graphique pour chaque question
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
            values = df[occurences[q]['column']].mean().tolist()
            title = occurences[q]['title']
                
            N = len(categories)

            theta = radar_factory(N, frame='polygon')
            fig1, axs = plt.subplots(figsize=(10, 10),
                                    subplot_kw=dict(projection='radar'))
            fig1.subplots_adjust(wspace=0.25,
                                hspace=0.25,
                                top=0.85,
                                bottom=0.1)
            
            # Définir l'échelle maximale en fonction de la valeur maximale des moyennes
            max_val = np.ceil(max(values))

            axs.set_rgrids(np.arange(1, max_val+1, 1))
            axs.set_ylim(0, max_val)
            
            axs.plot(theta, values, color='#D0F741')
            axs.fill(theta, values, facecolor='#D0F741', alpha=0.25)
            axs.set_varlabels(categories)
            
            axs.set_title(title,
                        weight='bold',
                        size='medium',
                        position=(0.5, 1.1),
                        horizontalalignment='center',
                        verticalalignment='center')
            
            plt.yticks(np.arange(1, max_val+1, 1), color="grey", size=10)

            fig1.tight_layout()
            # plt.show()
            fig1.savefig(f"{remplacer_caracteres(title)}.svg", format='svg')
            pdf.savefig(fig1, bbox_inches="tight")
            plt.close(fig1)
            

            # Ajout de l'histogramme des réponses
            #-------
            # calcul du nb de ligne
            nb_q = len(occurences[q]['column'])
            
            if nb_q % 4 == 0:
                nb_lignes = nb_q // 4
            else:
                nb_lignes = (nb_q // 4) + 1        
            
            nb_cols = 4 if nb_q > 4 else nb_q  # Limiter le nombre de colonnes à 3 max

            # Ajuster dynamiquement la taille de la figure en fonction du nombre de lignes
            fig_width = 20
            fig_height = 4 * nb_lignes

            # Créer une grille de subplots avec n lignes et 3 colonnes max
            fig2, axes = plt.subplots(nrows=nb_lignes, ncols=nb_cols, figsize=(fig_width, fig_height))

            for idx, col in enumerate(df[occurences[q]['column']].columns):

                val_counts = df[col].value_counts(sort=False).sort_index()

                # Calculer les indices des subplots
                row = idx // nb_cols
                col_idx = idx % nb_cols
                # print(f'nb_lignes:{nb_lignes}, nb_cols:{nb_cols}, row:{row}, col_idx:{col_idx}')

                # Gérer le cas où axes est un tableau 1D ou 2D
                if nb_lignes == 1:
                    ax = axes[col_idx]
                elif nb_cols == 1:
                    ax = axes[row]
                else:
                    ax = axes[row, col_idx]

                # Créer l'histogramme à partir des données val_counts
                ax.bar(val_counts.index, val_counts.values, width=0.8, color='blue')

                # Ajouter des étiquettes et un titre
                ax.set_xlabel(f'Answers (average:{df[col].mean():,.3f})')
                ax.set_ylabel('Occurrences')
                ax.set_title(f'...{col[-30:]}')

                # Définir les ticks sur l'axe des x en tant qu'entiers croissants à partir de 1
                max_val = int(val_counts.index.max())
                ax.set_xticks(range(1, max_val + 1))

                # Définir les valeurs de l'axe des y en tant qu'entiers
                current_ylim = ax.get_ylim()
                ax.set_ylim(bottom=0, top=current_ylim[1])
                ax.set_yticks(range(int(ax.get_ylim()[0]), int(ax.get_ylim()[1])+1, 1))

            
            # Supprimer les subplots vides si nécessaire
            if nb_q % nb_cols != 0:
                for idx in range(nb_q, nb_lignes * nb_cols):
                    fig2.delaxes(axes.flat[idx])


            # Ajuster la mise en page pour éviter les chevauchements
            plt.tight_layout()

            # Afficher le graphique
            # plt.show()
            fig2.savefig(f"{remplacer_caracteres(title)}_hist.svg")
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

            # Enregistre les infos pour pdf
            occurences[q]['imgs'] = [f"{remplacer_caracteres(title)}.svg", 
                                    f"{remplacer_caracteres(title)}_hist.svg"]
    pdf.close()

    pages =[]
    for q in occurences:
        if occurences[q].get('type') != 'comments':
            pages.append(('img', occurences[q]['title'], occurences[q].get('imgs')))
        elif occurences[q].get('type') == 'comments':
            valeurs_non_nulles_dropna = df[occurences[q]['column']].dropna()
            paragraph = ''
            for _string in valeurs_non_nulles_dropna.values:
                paragraph += f"{_string[0]}\n\n"
                
            pages.append(('txt', occurences[q]['title'], paragraph))

    create_pdf(pages=pages, file_name=pdf_file_name)

    return pdf_file_name

