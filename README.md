# QLearning-IA

# 📂 Estrutura do Projeto

## **gerador**
Responsável por gerar labirintos que seguem a regra de formação `2*n + 1` para o tamanho lateral, garantindo sempre **paredes externas** e **caminhos internos coerentes**.

**Exemplo:**  
Para `n = 15`, o labirinto terá tamanho `31 x 31`.

---

## **utils**
Contém ferramentas para **testar caminhos aprendidos** e **ajustar hiperparâmetros** do algoritmo.  

Você pode modificar diretamente os parâmetros de treinamento aqui:

```python
alphas = [0.1, 0.3, 0.5, 0.7, 0.9]        # Taxa de aprendizado
gamas = [0, 0.5, 0.7, 0.9, 1]              # Fator de desconto
epsilons = [0.1, 0.5, 0.9]                 # Grau de exploração
episodios_treino = [100, 500, 3000, 10000] # Número de episódios

# Ações possíveis: cima, baixo, esquerda, direita
acoes = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Tamanho base do labirinto
n = 15

# Geração de todas as combinações de alpha, gamma e epsilon
combinations = list(itertools.product(alphas, gamas, epsilons))

# Exemplo de criação de um labirinto aleatório
labirinto, tamanho, entrada, saida = gerar_labirinto_aleatorio(
    n, 30, "cima", "baixo"
)

## **agente**
Classe principal do agente **Q-Learning**, responsável por:

- **Treinar** a Q-Table;  
- **Escolher ações** com base em:
  - **Exploração** (`epsilon`);
  - **Exploração baseada em política** (*policy*);
- **Atualizar** valores de **estado-ação** após cada episódio.

---

## **qtable**
Módulo responsável por:

- **Criar e gerenciar** a Q-Table;  
- **Armazenar** os valores de **recompensa esperada** para cada par *(estado, ação)*.