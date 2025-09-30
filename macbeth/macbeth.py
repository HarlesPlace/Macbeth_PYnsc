class Macbeth:
    def __init__(self):
        self.criterias = []
        self.alternatives = []
    
    def add_criteria(self, name, type):
        self.criterias.append(Criteria(name, type))

    def add_alternative(self, name):
        self.alternatives.append(Alternative(name))

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

    def move_up(self, name):
        """Move o critério `name` uma posição para cima (se possível)."""
        for i, c in enumerate(self.criterias):
            if c.name == name:
                if i == 0:
                    return  # já está no topo
                self.criterias[i], self.criterias[i-1] = self.criterias[i-1], self.criterias[i]
                return
        raise ValueError(f"Criteria '{name}' not found.")

    def move_down(self, name):
        """Move o critério `name` uma posição para baixo (se possível)."""
        for i, c in enumerate(self.criterias):
            if c.name == name:
                if i == len(self.criterias) - 1:
                    return  # já está no fim
                self.criterias[i], self.criterias[i+1] = self.criterias[i+1], self.criterias[i]
                return
        raise ValueError(f"Criteria '{name}' not found.")
    
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
    
    

    
    