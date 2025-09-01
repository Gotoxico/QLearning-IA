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

    labirinto, tamanho, entrada, saida = gerar_labirinto_aleatorio(5, 30, "cima", "baixo")
    print("Labirinto criado:")
    print(labirinto)

    q_tabela = criar_q_tabela(tamanho)

    alpha = 0.1
    gama = 0.9
    epsilon = 0.1
    episodios_treino = 500
    q_tabela, sucesso = q_learning(q_tabela, labirinto, alpha, gama, epsilon, episodios_treino, entrada, saida)
    print("Episódios de sucesso no treino:", sucesso)

    acoes = [(-1,0),(1,0),(0,-1),(0,1)]
    max_passos = tamanho * tamanho * 2
    estado = entrada
    caminho = [estado]
    concluido = False
    passos = 0

    while not concluido and passos < max_passos:
        i,j = estado
        q_values = [
            q_tabela[i][j].consultar_cima(),
            q_tabela[i][j].consultar_baixo(),
            q_tabela[i][j].consultar_esquerda(),
            q_tabela[i][j].consultar_direita()
        ]
        acao = int(numpy.argmax(q_values))
        di,dj = acoes[acao]
        ni,nj = i+di,j+dj

        if 0 <= ni < tamanho and 0 <= nj < tamanho and labirinto[ni][nj] == 1:
            estado = (ni,nj)
            caminho.append(estado)
            if estado == saida:
                concluido = True
        else:
            # Pare se a ação é inválida
            break

        passos += 1

    print("\nCaminho encontrado (teste da política):")
    print(caminho)
    print("\nSucesso real:", "Sim" if concluido else "Não")
