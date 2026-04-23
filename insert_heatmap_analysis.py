"""
Script to insert a heatmap analysis markdown cell into the CVRP_brinks.ipynb notebook.
The cell is inserted right after the heatmap code cell (execution_count: 73)
and before the next code cell (execution_count: 74).
"""
import json
import sys

notebook_path = r"c:\Users\axelc\Desktop\cours\ECOLE-INGE\projet-graphes\projetRO\CVRP_brinks.ipynb"

# Read the notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the index of the heatmap code cell (execution_count: 73)
heatmap_cell_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code" and cell.get("execution_count") == 73:
        heatmap_cell_idx = i
        break

if heatmap_cell_idx is None:
    print("ERROR: Could not find heatmap code cell with execution_count=73")
    sys.exit(1)

print(f"Found heatmap code cell at index {heatmap_cell_idx}")

# Create the markdown analysis cell
analysis_cell = {
    "cell_type": "markdown",
    "id": "heatmap-analysis",
    "metadata": {},
    "source": [
        "### Analyse des heat maps de performance\n",
        "\n",
        "Cette analyse croisée présente, sous forme de cartes de chaleur, les gains (en %) de chaque algorithme d'amélioration par rapport à Clarke & Wright (CW), pour chaque combinaison de taille de réseau ($N$) et de capacité de véhicule ($Q$). L'échelle de couleur va du vert (gain nul ou faible) au rouge (gain élevé), permettant une lecture visuelle immédiate de la performance relative de chaque méthode.\n",
        "\n",
        "**1. Or-Opt vs CW : une amélioration marginale et uniforme**\n",
        "La première heat map (Gain Or-Opt/CW) révèle des gains quasi nuls sur l'ensemble de la grille ($N \\times Q$), oscillant entre 0,0 % et 0,8 % :\n",
        "* La couleur dominante est le vert foncé, signe que Or-Opt n'apporte qu'un raffinement cosmétique à la solution CW.\n",
        "* Le léger pic à 0,8 % s'observe sur un petit réseau ($N=20$) avec une capacité intermédiaire ($Q \\approx 300{-}400$), mais il reste négligeable en termes d'impact opérationnel.\n",
        "* Cela confirme que l'opérateur Or-Opt, bien qu'utile pour des ajustements locaux de sous-séquences, ne dispose pas d'un pouvoir de restructuration suffisant pour dépasser significativement CW.\n",
        "\n",
        "**2. Recuit Simulé (SA) vs CW : un potentiel concentré sur les petites instances**\n",
        "La deuxième heat map (Gain RS/CW) met en évidence un gradient net :\n",
        "* Pour $N=20$, les gains atteignent leur maximum (jusqu'à 3,8 % pour $Q=300$), avec une bande de couleurs chaudes clairement visible.\n",
        "* Dès que $N$ augmente ($N \\ge 50$), les gains chutent drastiquement, passant sous la barre de 1 % et tendant vers 0 % pour $N=100$.\n",
        "* La capacité $Q$ influence modérément les résultats sur les petites instances : les capacités intermédiaires ($Q = 200{-}300$) semblent offrir un espace d'exploration plus favorable au SA.\n",
        "* Ce comportement est cohérent : sur un petit réseau, l'espace de recherche est suffisamment restreint pour que le SA explore efficacement le voisinage, tandis que sur un grand réseau, le nombre d'itérations devient insuffisant pour exploiter la complexité combinatoire.\n",
        "\n",
        "**3. Colonie de fourmis (AC) vs CW : une instabilité structurelle alarmante**\n",
        "La troisième heat map (Gain AC/CW) est la plus frappante :\n",
        "* Elle affiche une dominante rouge/orange, indiquant paradoxalement des **gains négatifs** massifs, de l'ordre de $-3$% à $-8$% voire au-delà pour certaines configurations.\n",
        "* Contrairement aux autres métaheuristiques qui partent de la solution CW pour l'améliorer, la Colonie de fourmis (ACO) construit sa solution **de zéro**, ce qui explique qu'elle puisse produire des résultats **inférieurs** à CW.\n",
        "* La dégradation s'accentue fortement avec $N$ : pour $N \\ge 75$, les pertes dépassent systématiquement $-5$%.\n",
        "* Cela révèle que, sans un mécanisme d'initialisation basé sur CW, l'ACO est incapable de rivaliser avec l'heuristique de construction, en particulier sur les instances de grande taille.\n",
        "\n",
        "**4. Recherche Tabou vs CW : un profil comparable à Or-Opt**\n",
        "La quatrième heat map (Gain Tabu/CW) montre un profil très similaire à Or-Opt :\n",
        "* Les gains restent faibles, entre 0,0 % et 0,8 %, avec une dominante de vert foncé sur la quasi-totalité de la grille.\n",
        "* Les valeurs les plus élevées apparaissent pour $N=20$, mais elles restent modestes.\n",
        "* Ce résultat suggère que la recherche Tabou, dans sa configuration actuelle, atteint rapidement les limites de son voisinage d'exploration sans parvenir à s'éloigner significativement de la solution CW initiale.\n",
        "\n",
        "### Synthèse\n",
        "\n",
        "* **Dichotomie construction / amélioration** : Les algorithmes partant de la solution CW (Or-Opt, SA, Tabu) ne font que la raffiner, tandis que l'ACO, qui reconstruit sans s'appuyer sur CW, produit des résultats nettement inférieurs. Cela confirme la nécessité absolue d'une **approche hybride** (construction CW + métaheuristique d'amélioration).\n",
        "* **Influence de $N$ et $Q$** : La taille du réseau ($N$) est le facteur dominant dans la performance des métaheuristiques. L'influence de la capacité ($Q$) est secondaire et se manifeste principalement sur les petites instances.\n",
        "* **Recommandation opérationnelle** : Pour des instances de petite ou moyenne taille ($N \\le 50$), le Recuit Simulé est le meilleur candidat d'amélioration. Pour des instances plus grandes, les gains marginaux de toutes les métaheuristiques suggèrent que l'effort devrait se concentrer sur le **paramétrage** (augmentation du nombre d'itérations, réglage de la température) ou sur des méthodes hybrides plus avancées."
    ]
}

# Insert the cell right after the heatmap cell
nb["cells"].insert(heatmap_cell_idx + 1, analysis_cell)

# Write the notebook back
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Successfully inserted heatmap analysis cell at index {heatmap_cell_idx + 1}")
print(f"Total cells: {len(nb['cells'])}")
