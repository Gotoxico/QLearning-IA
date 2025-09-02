import numpy as np
import matplotlib.pyplot as plt
from labirinto.gerador import gerar_labirinto_aleatorio
from qlearning.agente import q_learning
from qlearning.qtable import criar_q_tabela
import itertools

def encontrar_caminho():
    """
    Treina agentes Q-Learning em um labirinto para diferentes combinações
    de parâmetros (alpha, gamma, epsilon) e plota os caminhos encontrados.
    """

    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    gamas = [0, 0.5, 0.7, 0.9, 1]
    epsilons = [0.1, 0.5, 0.9]
    episodios_treino = [100, 500, 3000]
    acoes = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    n = 9
    combinations = list(itertools.product(alphas, gamas, epsilons))
    labirinto, tamanho, entrada, saida = gerar_labirinto_aleatorio(
        n, 30, "cima", "baixo"
    )
    print(f"\n=== Labirinto criado, n: {n} ===")
    plt.imshow(labirinto, cmap="Greys")
    plt.scatter(*entrada[::-1], c="green", label="Entrada")
    plt.scatter(*saida[::-1], c="red", label="Saída")
    plt.title(f"Labirinto n={n}")
    plt.axis("off")
    plt.legend()
    plt.show()


    for episodios in episodios_treino:

        fig, axes = plt.subplots(
            len(alphas), len(gamas) * len(epsilons),
            figsize=(20, 20)
        )
        if len(combinations) == 1:
            axes = np.array([[axes]])
        elif len(alphas) == 1:
            axes = np.array([axes])

        fig.suptitle(f"Labirinto n={n} - Caminhos encontrados (Q-Learning) - Episódios de treino: {episodios}", fontsize=18)

        for idx, (alpha, gama, epsilon) in enumerate(combinations):
            q_tabela = criar_q_tabela(tamanho)
            q_tabela, sucesso = q_learning(
                q_tabela, labirinto, alpha, gama, epsilon,
                episodios, entrada, saida
            )
            print(f"[α={alpha}, γ={gama}, ε={epsilon}] -> sucesso: {sucesso}")

            max_passos = tamanho * tamanho * 2
            estado = entrada
            caminho = [estado]
            concluido = False
            passos = 0

            while not concluido and passos < max_passos:
                i, j = estado
                q_values = [
                    q_tabela[i][j].consultar_cima(),
                    q_tabela[i][j].consultar_baixo(),
                    q_tabela[i][j].consultar_esquerda(),
                    q_tabela[i][j].consultar_direita()
                ]
                acao = int(np.argmax(q_values))
                di, dj = acoes[acao]
                ni, nj = i + di, j + dj

                if 0 <= ni < tamanho and 0 <= nj < tamanho and labirinto[ni][nj] == 1:
                    estado = (ni, nj)
                    caminho.append(estado)
                    if estado == saida:
                        concluido = True
                else:
                    break
                passos += 1

            CaminhoMatriz = np.copy(labirinto)
            for (ci, cj) in caminho:
                CaminhoMatriz[ci][cj] = 0.5  

            row = idx // (len(gamas) * len(epsilons))
            col = idx % (len(gamas) * len(epsilons))
            ax = axes[row, col]

            # Plot
            ax.imshow(labirinto, cmap="Greys", alpha=0.8)      # Labirinto em cinza
            ax.imshow(CaminhoMatriz, cmap="viridis", alpha=0.7) # Caminho em tons vibrantes
            ax.set_title(f"α={alpha}\nγ={gama}\nε={epsilon}", fontsize=9)
            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
