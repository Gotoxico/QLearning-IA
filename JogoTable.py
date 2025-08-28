import numpy
import random

def gerarLabirintoAleatorio(n, seed, entrada, saida):
    random.seed(seed)
    numpy.random.seed(seed)

    labirinto = numpy.zeros((2*n+1,2*n+1), dtype=int)

    direcoes = [(-2,0), (2,0), (0,-2), (0,2)]

    def dentroLimites(l, c):
        return 0 < l < 2*n and 0 < c < 2*n
    
    def dfs(l, c):
        labirinto[l, c] = 1

        dirs = direcoes[:]
        random.shuffle(dirs)

        for dl, dc in dirs:
            nl, nc = l + dl, c + dc
            if dentroLimites(nl, nc) and labirinto[nl, nc] == 0:
                labirinto[l+dl//2, c+dc//2] = 1
                dfs(nl, nc)

    

    dfs(1,1)

    if entrada == "cima":
        entrada = (0, 1); labirinto[entrada] = 1
    elif entrada == "baixo":
        entrada = (2*n, 2*n-1); labirinto[entrada] = 1
    elif entrada == "esquerda":
        entrada = (1, 0); labirinto[entrada] = 1
    elif entrada == "direita":
        entrada = (2*n-1, 2*n); labirinto[entrada] = 1
    else:
        raise ValueError("entrada deve ser: 'cima', 'baixo', 'esquerda' ou 'direita'")

    if saida == "cima":
        saida = (0, 2*n-1); labirinto[saida] = 1
    elif saida == "baixo":
        saida = (2*n, 1); labirinto[saida] = 1
    elif saida == "esquerda":
        saida = (2*n-1, 0); labirinto[saida] = 1
    elif saida == "direita":
        saida = (1, 2*n); labirinto[saida] = 1
    else:
        raise ValueError("saida deve ser: 'cima', 'baixo', 'esquerda' ou 'direita'")
    return labirinto, 2*n+1, entrada, saida

class Celula:
    def __init__(self):
        self.esquerda = 0
        self.direita = 0
        self.cima = 0
        self.baixo = 0

    #Getters
    def consultarEsquerda(self):
        return self.esquerda
    
    def consultarDireita(self):
        return self.direita
    
    def consultarCima(self):
        return self.cima
    
    def consultarBaixo(self):
        return self.baixo
    

    #Setters
    def atualizarEsquerda(self, valor):
        self.esquerda = valor

    def atualizarDireita(self, valor):
        self.direita = valor

    def atualizarCima(self, valor):
        self.cima = valor

    def atualizarbaixo(self, valor):
        self.baixo = valor


def criarQTabela(tamanho):
    QTabela = [[Celula() for j in range(tamanho)] for i in range(tamanho)]
    return QTabela



def QLearning(QTabela, labirinto, alpha, gama, episodios, entrada, saida):
    tamanho = len(labirinto)
    acoes = [(-1,0), (1,0), (0,-1), (0,1)]

    for ep in range(episodios):
        print("Episodio: ", ep)
        estado = entrada
        concluido = False
        passos = 0
        maxPassos = tamanho * tamanho

        while not concluido and passos < maxPassos:
            i, j = estado

            qValues = [QTabela[i][j].consultarCima(), QTabela[i][j].consultarBaixo(), QTabela[i][j].consultarEsquerda(), QTabela[i][j].consultarDireita()]

            acao = int(numpy.argmax(qValues))

            di, dj = acoes[acao]
            ni, nj = i + di, j + dj

            if 0 <= ni < tamanho and 0 <= nj < tamanho and labirinto[ni][nj] == 1:
                proximoEstado = (ni,nj)
                if proximoEstado == saida:
                    recompensa = 100
                    concluido = True
                else:
                    recompensa = -1
            
            else:
                proximoEstado = estado
                recompensa = -10
            
            ni, nj = proximoEstado
            proximoQ = [QTabela[ni][nj].consultarCima(), QTabela[ni][nj].consultarBaixo(), QTabela[ni][nj].consultarEsquerda(), QTabela[ni][nj].consultarDireita()]
            QAtual = qValues[acao]
            QNovo = QAtual + alpha * (recompensa + gama * max(proximoQ) - QAtual)

            if acao == 0:
                QTabela[i][j].atualizarCima(QNovo)
            elif acao == 1:
                QTabela[i][j].atualizarbaixo(QNovo)
            elif acao == 2:
                QTabela[i][j].atualizarEsquerda(QNovo)
            else:
                QTabela[i][j].atualizarDireita(QNovo)

            estado = proximoEstado

    return QTabela



def encontrarCaminho():
    labirinto, tamanho, entrada, saida = gerarLabirintoAleatorio(5, 30, "cima", "baixo")
    print("Labirinto criado:")
    print(labirinto)

    QTabela = criarQTabela(tamanho)

    QTabela = QLearning(QTabela, labirinto, 0.1, 0.9, 500, entrada, saida)

    estado = entrada
    caminho = [estado]
    concluido = False
    maxPassos = tamanho * tamanho  

    while not concluido and len(caminho) < maxPassos:
        i, j = estado
        qValues = [
            QTabela[i][j].consultarCima(),
            QTabela[i][j].consultarBaixo(),
            QTabela[i][j].consultarEsquerda(),
            QTabela[i][j].consultarDireita()
        ]
        acao = int(numpy.argmax(qValues))
        acoes = [(-1,0),(1,0),(0,-1),(0,1)]
        di, dj = acoes[acao]
        ni, nj = i + di, j + dj

        if 0 <= ni < tamanho and 0 <= nj < tamanho and labirinto[ni][nj] == 1:
            estado = (ni,nj)
        else:
            break

        caminho.append(estado)

        if estado == saida:
            concluido = True

    print("\nCaminho encontrado:")
    print(caminho)


if __name__ == "__main__":
    encontrarCaminho()



