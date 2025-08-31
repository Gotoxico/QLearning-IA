import numpy

from labirinto.gerador import gerar_labirinto_aleatorio
from qlearning.agente import q_learning
from qlearning.qtable import criar_q_tabela


def encontrar_caminho():
    """
    Generate a random maze, train a QLearning agent, and find a path from the entrance
    to the exit using the learned QTable.

    This function creates a maze, initializes a QTable, runs the QLearning algorithm,
    and then reconstructs the path from the entrance to the exit by following the
    actions with the highest Q values.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The function prints the generated maze and the path found to the console.
    """

    labirinto, tamanho, entrada, saida = gerar_labirinto_aleatorio(
        5, 30, "cima", "baixo"
    )
    print("Labirinto criado:")
    print(labirinto)

    q_tabela = criar_q_tabela(tamanho)

    q_tabela = q_learning(q_tabela, labirinto, 0.1, 0.9, 500, entrada, saida)

    estado = entrada
    caminho = [estado]
    concluido = False
    max_passos = tamanho * tamanho

    while not concluido and len(caminho) < max_passos:
        i, j = estado
        q_values = [
            q_tabela[i][j].consultar_cima(),
            q_tabela[i][j].consultar_baixo(),
            q_tabela[i][j].consultar_esquerda(),
            q_tabela[i][j].consultar_direita(),
        ]
        acao = int(numpy.argmax(q_values))
        acoes = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        di, dj = acoes[acao]
        ni, nj = i + di, j + dj

        if 0 <= ni < tamanho and 0 <= nj < tamanho and labirinto[ni][nj] == 1:
            estado = (ni, nj)
        else:
            break

        caminho.append(estado)

        if estado == saida:
            concluido = True

    print("\nCaminho encontrado:")
    print(caminho)
