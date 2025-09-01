import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import labirinto.gerador as lg
from qlearning.agente import q_learning
from qlearning.qtable import criar_q_tabela


def testar_tamanhos():
    tamanhos = range(1, 10)
    sucessos = []

    for n in tamanhos:
        labirinto, tamanho, entrada, saida = lg.gerar_labirinto_aleatorio(n, 30, "cima", "baixo")
        q_tabela = criar_q_tabela(tamanho)
        _, sucesso = q_learning(q_tabela, labirinto, 0.1, 0.9, 0.1, 500, entrada, saida)
        sucessos.append(sucesso / 500)  

    plt.plot(tamanhos, sucessos, marker="o")
    plt.xlabel("Tamanho do labirinto (n)")
    plt.ylabel("Taxa de sucesso")
    plt.title("Taxa de sucesso vs Tamanho do labirinto")
    plt.grid(True)
    plt.ylim(0,1.05)
    plt.show()

def testar_entradas_saidas():
    posicoes = ["cima", "baixo", "esquerda", "direita"]
    resultados = {}

    for entrada in posicoes:
        for saida in posicoes:
            if entrada == saida:
                continue
            labirinto, tamanho, e, s = lg.gerar_labirinto_aleatorio(5, 30, entrada, saida)
            q_tabela = criar_q_tabela(tamanho)
            _, sucesso = q_learning(q_tabela, labirinto, 0.1, 0.9, 0.1, 500, e, s)
            resultados[(entrada, saida)] = sucesso / 500

    for (entrada, saida), taxa in resultados.items():
        print(f"Entrada: {entrada}, Saída: {saida} → Sucesso: {taxa:.2f}")

if __name__ == "__main__":
    testar_tamanhos()
    # estar_entradas_saidas()


