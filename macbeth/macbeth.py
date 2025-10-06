import csv

class Macbeth:
    def __init__(self):
        self.criterias = []
        self.alternatives = []
        self.judgment_matrix = []
    
    def add_criteria(self, name, type="+"):
        self.criterias.append(Criteria(name, type))
        self._expand_matrix()

    def add_alternative(self, name):
        self.alternatives.append(Alternative(name))

    def _expand_matrix(self):
        n = len(self.criterias)
        for row in self.judgment_matrix:
            row.append(None)
        self.judgment_matrix.append([None] * n)

    def _swap_matrix_rows(self, i, j):
        """Troca as linhas i e j da matriz."""
        self.judgment_matrix[i], self.judgment_matrix[j] = (
            self.judgment_matrix[j],
            self.judgment_matrix[i],
        )

    def _swap_matrix_cols(self, i, j):
        """Troca as colunas i e j da matriz."""
        for row in self.judgment_matrix:
            row[i], row[j] = row[j], row[i]

    def show_criteria(self, detailed=False):
        """Exibe os critérios na ordem atual com suas posições.
        Se detailed=True, mostra também peso e tipo."""
        for i, c in enumerate(self.criteria, start=1):
            if detailed:
                print(f"{i}. {c.name} (weight={c.weight}, type={c.type})")
            else:
                print(f"{i}. {c.name}")

    def swap_criteria(self, name1, name2):
        """Troca a posição de dois critérios."""
        idx1, idx2 = None, None
        for i, c in enumerate(self.criterias):
            if c.name == name1:
                idx1 = i
            elif c.name == name2:
                idx2 = i
        if idx1 is None or idx2 is None:
            raise ValueError("One or both criterias not found")
        self.criterias[idx1], self.criterias[idx2] = self.criterias[idx2], self.criterias[idx1]
        self._swap_matrix_rows(idx1, idx2)
        self._swap_matrix_cols(idx1, idx2)

    def move_up(self, name):
        """Move o critério `name` uma posição para cima (se possível)."""
        for i, c in enumerate(self.criterias):
            if c.name == name:
                if i == 0:
                    return  # já está no topo
                self.criterias[i], self.criterias[i-1] = self.criterias[i-1], self.criterias[i]
                self._swap_matrix_rows(i, i-1)
                self._swap_matrix_cols(i, i-1)
                return
        raise ValueError(f"Criteria '{name}' not found.")

    def move_down(self, name):
        """Move o critério `name` uma posição para baixo (se possível)."""
        for i, c in enumerate(self.criterias):
            if c.name == name:
                if i == len(self.criterias) - 1:
                    return  # já está no fim
                self.criterias[i], self.criterias[i+1] = self.criterias[i+1], self.criterias[i]
                self._swap_matrix_rows(i, i+1)
                self._swap_matrix_cols(i, i+1)
                return
        raise ValueError(f"Criteria '{name}' not found.")
    
    def import_judments_from_csv(self, filepath):
        """Importa a matriz de julgamentos de um CSV."""
        with open(filepath, newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            data = list(reader) 

        n = len(data)
        if n != len(self.criterias):        # precisa ter mesmo número de critérios
            raise ValueError("Número de critérios não bate com a matriz.")

        self.judgment_matrix = [[float(x) for x in row] for row in data]
    
    def import_criterias_and_judments_from_csv(self, filepath):
        """Importa matriz de julgamentos de um CSV e recria critérios e matriz."""
        with open(filepath, newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)

        if not data or len(data) < 2:
            raise ValueError("Arquivo CSV vazio ou inválido.")

        col_names = [h.strip() for h in data[0][1:]]
        n = len(col_names)

        self.criterias = [Criteria(name) for name in col_names] # recria a lista com base nos nomes do CSV
            
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]

        for i, row in enumerate(data[1:]):  # ignora o cabeçalho
            row_name = row[0].strip()
            if row_name != col_names[i]:
                print(f"Aviso: critério da linha {i+1} ({row_name}) difere do cabeçalho ({col_names[i]})")
            for j, val in enumerate(row[1:]):
                try:
                    matrix[i][j] = float(val)
                except ValueError:
                    raise ValueError(f"Valor inválido na posição ({i},{j}): '{val}'")

        self.judgment_matrix = matrix

    def export_matrix_to_csv(self, filepath):
        """Exporta a matriz de julgamentos atual para um CSV com cabeçalho e nomes de critérios."""
        if not hasattr(self, "judgment_matrix") or not self.judgment_matrix:
            raise ValueError("No judgment matrix to export ")

        if not self.criterias:
            raise ValueError("Criterias list empty - nothing to export")

        names = [c.name for c in self.criterias]
        n = len(names)

        if len(self.judgment_matrix) != n or any(len(row) != n for row in self.judgment_matrix):
            raise ValueError("Dimension of the judgment matrix and number of criterias doesn't match")
        rows = []

        header = [""] + names  # Cabeçalho: ["", C1, C2, C3, ...]
        rows.append(header)

        for i, name in enumerate(names):
            row = [name] + [self.judgment_matrix[i][j] for j in range(n)]
            rows.append(row)

        with open(filepath, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"Judgment matrix successfuly exported to '{filepath}'")

class Criteria:
    def __init__(self, name, type="+"):
        if type not in {"+", "-"}:
            raise ValueError("Criteria type must be '+' (benefit) or '-' (cost).")
        self.name = name
        self.weight = 0.0
        self.type = type  # "+" for benefit, "-" for cost
    
    def set_weight(self, weight):
        self.weight = weight
    
    def __repr__(self):
        return f"Criteria(name={self.name}, weight={self.weight}, type={self.type})"
    
    def set_weight(self, weight):
        if not isinstance(weight, (int, float)):
            raise TypeError("Weight must be numeric.")
        if weight < 0:
            raise ValueError("Weight must be non-negative.")
        self.weight = float(weight)
    
    def is_benefit(self):
        return self.type == "+"
    
    def is_cost(self):
        return self.type == "-"

class Alternative:
    def __init__(self, name):
        self.name = name
        self.scores = {}  # key: criteria name, value: score
    
    def __repr__(self):
        return f"Alternative(name={self.name}, scores={self.scores})"
    
    def set_score(self, criteria_name, score):
        self.scores[criteria_name] = score
    
    def get_score(self, criteria_name):
        if criteria_name not in self.scores:
            raise ValueError(f"The criteria '{criteria_name}' has no score defined for '{self.name}'.")
        return self.scores[criteria_name]

    def get_all_scores(self):
        return self.scores
    
    def has_score(self, criteria_name):
        return criteria_name in self.scores
    
    

    
    