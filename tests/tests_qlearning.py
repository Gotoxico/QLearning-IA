import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy
from labirinto.gerador import gerar_labirinto_aleatorio
from qlearning.agente import q_learning
from qlearning.qtable import criar_q_tabela


def testar_alphas_gamas():
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    gamas = [0.5, 0.7, 0.9, 0.99]

    resultados = numpy.zeros((len(alphas), len(gamas)))

    for i, alpha in enumerate(alphas):
        for j, gama in enumerate(gamas):
            labirinto, tamanho, entrada, saida = gerar_labirinto_aleatorio(9, 30, "cima", "baixo")
            q_tabela = criar_q_tabela(tamanho)
            _, sucesso = q_learning(q_tabela, labirinto, alpha, gama, 0.1, 500, entrada, saida)
            resultados[i, j] = sucesso / 500

    plt.imshow(resultados, cmap="viridis", origin="lower", 
               extent=[min(gamas), max(gamas), min(alphas), max(alphas)],
               aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="Taxa de sucesso")
    plt.xlabel("Gama")
    plt.ylabel("Alpha")
    plt.xlim(min(gamas), max(gamas))
    plt.ylim(min(alphas), max(alphas))
    plt.title("Taxa de sucesso em função de Alpha e Gama")
    plt.show()

if __name__ == "__main__":
    testar_alphas_gamas()