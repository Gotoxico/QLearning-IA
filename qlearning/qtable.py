class Celula:
    """
    Represent a single state in the QTable, storing Q values for four possible actions:
    left, right, up, and down.

    Attributes
    ----------
    esquerda : float
        Q value for the left action.
    direita : float
        Q value for the right action.
    cima : float
        Q value for the up action.
    baixo : float
        Q value for the down action.
    """

    def __init__(self):
        """
        Initialize a new Celula with all Q values set to 0.
        """
        self.esquerda = 0
        self.direita = 0
        self.cima = 0
        self.baixo = 0

    # Getters
    def consultar_esquerda(self):
        """
        Get the Q value for the left action.

        Returns
        -------
        float
            Q value for the left action.
        """
        return self.esquerda

    def consultar_direita(self):
        """
        Get the Q value for the right action.

        Returns
        -------
        float
            Q value for the right action.
        """
        return self.direita

    def consultar_cima(self):
        """
        Get the Q value for the up action.

        Returns
        -------
        float
            Q value for the up action.
        """
        return self.cima

    def consultar_baixo(self):
        """
        Get the Q value for the down action.

        Returns
        -------
        float
            Q value for the down action.
        """
        return self.baixo

    # Setters
    def atualizar_esquerda(self, valor):
        """
        Update the Q value for the left action.

        Parameters
        ----------
        valor : float
            New Q value for the left action.
        """
        self.esquerda = valor

    def atualizar_direita(self, valor):
        """
        Update the Q value for the right action.

        Parameters
        ----------
        valor : float
            New Q value for the right action.
        """
        self.direita = valor

    def atualizar_cima(self, valor):
        """
        Update the Q value for the up action.

        Parameters
        ----------
        valor : float
            New Q value for the up action.
        """
        self.cima = valor

    def atualizar_baixo(self, valor):
        """
        Update the Q value for the down action.

        Parameters
        ----------
        valor : float
            New Q value for the down action.
        """
        self.baixo = valor


def criar_q_tabela(tamanho):
    """
    Create a QTable structure initialized with Celula objects.

    The QTable is represented as a 2D list where each cell corresponds to a state
    in the maze, and each Celula object stores the Q values for the possible actions
    (up, down, left, right).

    Parameters
    ----------
    tamanho : int
        Dimension of the maze matrix.

    Returns
    -------
    q_tabela : list[list[Celula]]
        A 2D list of Celula objects representing the QTable.
    """

    q_tabela = [[Celula() for _ in range(tamanho)] for _ in range(tamanho)]
    return q_tabela
