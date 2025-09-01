import numpy
import random


def q_learning(q_tabela, labirinto, alpha, gama, epsilon, episodios, entrada, saida):
    """
    Run the QLearning algorithm on the given maze environment.

    The agent starts at the entrance and explores the maze, updating the QTable
    according to the rewards received for each action. Rewards are defined as:
    100 for reaching the exit,
    -1 for valid moves that are not the exit,
    -10 for invalid moves (hitting a wall or going out of bounds).

    Parameters
    ----------
    q_tabela : list[list[Celula]]
        The QTable structure storing Q values for each action in each state.
    labirinto : numpy.ndarray
        Matrix representing the maze. 1 indicates open path, 0 indicates wall.
    alpha : float
        Learning rate that controls how much new information overrides old values.
    gama : float
        Discount factor for future rewards (0 <= gama <= 1).
    episodios : int
        Number of training episodes to run.
    entrada : tuple
        Coordinates of the entrance cell.
    saida : tuple
        Coordinates of the exit cell.

    Returns
    -------
    q_tabela : list[list[Celula]]
        Updated QTable after running the QLearning algorithm.
    """

    tamanho = len(labirinto)
    acoes = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    sucesso = 0

    for ep in range(episodios):
        print("Episodio: ", ep)
        estado = entrada
        concluido = False
        passos = 0
        max_passos = tamanho * tamanho * 4

        while not concluido and passos < max_passos:
            i, j = estado

            q_values = [
                q_tabela[i][j].consultar_cima(),
                q_tabela[i][j].consultar_baixo(),
                q_tabela[i][j].consultar_esquerda(),
                q_tabela[i][j].consultar_direita(),
            ]

            if random.uniform(0, 1) < epsilon:
                acao = random.randint(0, 3)
            else:
                acao = int(numpy.argmax(q_values))

            di, dj = acoes[acao]
            ni, nj = i + di, j + dj

            if 0 <= ni < tamanho and 0 <= nj < tamanho and labirinto[ni][nj] == 1:
                proximo_estado = (ni, nj)
                if proximo_estado == saida:
                    recompensa = 100
                    concluido = True  
                    sucesso += 1
                else:
                    recompensa = -1

            else:
                proximo_estado = estado
                recompensa = -10

            ni, nj = proximo_estado
            proximo_q = [
                q_tabela[ni][nj].consultar_cima(),
                q_tabela[ni][nj].consultar_baixo(),
                q_tabela[ni][nj].consultar_esquerda(),
                q_tabela[ni][nj].consultar_direita(),
            ]
            q_atual = q_values[acao]
            q_novo = q_atual + alpha * (recompensa + gama * max(proximo_q) - q_atual)

            if acao == 0:
                q_tabela[i][j].atualizar_cima(q_novo)
            elif acao == 1:
                q_tabela[i][j].atualizar_baixo(q_novo)
            elif acao == 2:
                q_tabela[i][j].atualizar_esquerda(q_novo)
            else:
                q_tabela[i][j].atualizar_direita(q_novo)

            estado = proximo_estado

    return q_tabela, sucesso
