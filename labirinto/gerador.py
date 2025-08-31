import random

import numpy


def gerar_labirinto_aleatorio(n, seed, entrada, saida):
    """
    Generate a random maze using depth first search algorithm.

    Parameters
    ----------
    n : int
        Size of the maze (number of cells in one dimension).
    seed : int
        Seed for random number generators to ensure reproducibility.
    entrada : str
        Position of the entrance. Must be one of: "cima", "baixo", "esquerda", "direita".
    saida : str
        Position of the exit. Must be one of: "cima", "baixo", "esquerda", "direita".

    Returns
    -------
    labirinto : numpy.ndarray
        Matrix representing the maze. 1 indicates open path, 0 indicates wall.
    tamanho : int
        Dimension of the maze matrix (2*n + 1).
    entrada : tuple
        Coordinates of the entrance cell.
    saida : tuple
        Coordinates of the exit cell.

    Raises
    ------
    ValueError
        If entrada or saida are not one of the allowed values.
    """

    random.seed(seed)
    numpy.random.seed(seed)

    labirinto = numpy.zeros((2 * n + 1, 2 * n + 1), dtype=int)

    direcoes = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    def dentro_limites(l, c):
        return 0 < l < 2 * n and 0 < c < 2 * n

    def dfs(l, c):
        labirinto[l, c] = 1

        dirs = direcoes[:]
        random.shuffle(dirs)

        for dl, dc in dirs:
            nl, nc = l + dl, c + dc
            if dentro_limites(nl, nc) and labirinto[nl, nc] == 0:
                labirinto[l + dl // 2, c + dc // 2] = 1
                dfs(nl, nc)

    dfs(1, 1)

    if entrada == "cima":
        entrada = (0, 1)
        labirinto[entrada] = 1
    elif entrada == "baixo":
        entrada = (2 * n, 2 * n - 1)
        labirinto[entrada] = 1
    elif entrada == "esquerda":
        entrada = (1, 0)
        labirinto[entrada] = 1
    elif entrada == "direita":
        entrada = (2 * n - 1, 2 * n)
        labirinto[entrada] = 1
    else:
        raise ValueError("entrada deve ser: 'cima', 'baixo', 'esquerda' ou 'direita'")

    if saida == "cima":
        saida = (0, 2 * n - 1)
        labirinto[saida] = 1
    elif saida == "baixo":
        saida = (2 * n, 1)
        labirinto[saida] = 1
    elif saida == "esquerda":
        saida = (2 * n - 1, 0)
        labirinto[saida] = 1
    elif saida == "direita":
        saida = (1, 2 * n)
        labirinto[saida] = 1
    else:
        raise ValueError("saida deve ser: 'cima', 'baixo', 'esquerda' ou 'direita'")
    return labirinto, 2 * n + 1, entrada, saida
