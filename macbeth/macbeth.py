import csv, pulp

class Macbeth:
    def __init__(self):
        self.criterias = []
        self.alternatives = []
        self.judgment_matrix = []
        self.classes = {"Very Weak": 1, "Weak": 2, "Moderate": 3, "Strong": 4, "Very Strong": 5, "Extreme": 6}
        self.classesBoundaries = []
        self.consistence_checked = False
        self.c_min = None
        self.consistent_judgment = False
    
    def add_criteria(self, name, type="+"):
        self.consistence_checked = False
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
        for i, c in enumerate(self.criterias, start=1):
            if detailed:
                print(f"{i}. {c.name} (weight={c.weight}, type={c.type})")
            else:
                print(f"{i}. {c.name}")

    def swap_criteria(self, name1, name2):
        """Troca a posição de dois critérios."""
        self.consistence_checked = False
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
        self.consistence_checked = False
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
        self.consistence_checked = False
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
        self.consistence_checked = False
        with open(filepath, newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            data = list(reader) 

        n = len(data)
        if n != len(self.criterias):        # precisa ter mesmo número de critérios
            raise ValueError("Número de critérios não bate com a matriz.")

        self.judgment_matrix = [[float(x) for x in row] for row in data]
    
    def import_criterias_and_judments_from_csv(self, filepath):
        """Importa matriz de julgamentos de um CSV e recria critérios e matriz."""
        self.consistence_checked = False
        with open(filepath, newline='', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=';')
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
    
    def interactive_judgment_input(self):
        """Constrói interativamente a matriz de julgamentos pedindo preferências ao usuário."""
        if not self.criterias:
            print("No criteria defined. Please add criteria first.")
            return

        names = [c.name for c in self.criterias]
        n = len(names)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]

        print("\n=== MACBETH INTERACTIVE JUDGMENT MODE ===")
        print("Difference levels (0 = Indifference, 6 = Extreme):")
        print("0-Equal | 1-Very Weak | 2-Weak | 3-Moderate | 4-Strong | 5-Very Strong | 6-Extreme")
        print("Type 'q' at any time to quit.\n")

        for i in range(n):
            for j in range(i+1, n):
                while True:
                    print(f"\nComparing: {names[i]}  ×  {names[j]}")
                    val = input("How much is the first criterion preferred over the second? (0-6): ").strip()
                    if val.lower() == "q":
                        print("Operation canceled.")
                        return
                    try:
                        v = int(val)
                        if 0 <= v <= 6:
                            matrix[i][j] = v
                            matrix[j][i] = -v  # valor recíproco
                            break
                        else:
                            print("Invalid value. Please enter a number between 0 and 6.")
                    except ValueError:
                        print("Invalid input. Please enter a number between 0 and 6.")

                # Exibe a matriz parcial
                print("\nCurrent judgment matrix:")
                self._print_matrix_preview(names, matrix)

        print("\nFinal proposed matrix:")
        self._print_matrix_preview(names, matrix)
        confirm = input("\nConfirm and replace the current judgment matrix? (Y/N): ").strip().lower()
        if confirm == "s":
            self.judgment_matrix = matrix
            print("Judgment matrix successfully updated!")
        else:
            print("Matrix discarded.")

    def _print_matrix_preview(self, names, matrix):
        """Imprime a matriz de julgamentos em formato tabular."""
        header = ["{:>12}".format("")] + [f"{n:>12}" for n in names]
        print("".join(header))
        for i, name in enumerate(names):
            row = [f"{name:>12}"] + [f"{matrix[i][j]:>12}" for j in range(len(names))]
            print("".join(row))
    
    def check_consistency(self):
        """Verifica a consistência dos julgamentos da matriz de julgamentos usando o programa _MC1."""
        if self.consistence_checked:
            print("\n Consistency already checked.")
            if self.consistent_judgment:
                print("\n The judgment matrix is consistent.")
            else:
                print("\n The judgment matrix is NOT consistent.")
            return self.consistent_judgment
        self.c_min = self._MC1()
        self.consistence_checked = True
        self.consistent_judgment = self.c_min == 0
        if self.consistent_judgment:
            print("The judgment matrix is consistent.")
        else:
            print("The judgment matrix is NOT consistent.")
        return self.consistent_judgment
    
    def _MC1(self):
        """Programa MC1 de MACBETH para determinar o valor de incoerência c_min."""
        prob = pulp.LpProblem("Minimizar_c", pulp.LpMinimize)
        theta = 0.001
        c = pulp.LpVariable("c", lowBound=0)

        p = {}
        for i in range(1,len(self.criterias)+1):
            p[i] = pulp.LpVariable(f"p{i}", lowBound=0) 

        s = {}
        for i in range(0, len(self.classes)):
            s[i] = pulp.LpVariable(f"s{i}", lowBound=0)
        
        # 1
        prob += s[0] == 0, "R_s0_fixo"
        prob += s[1] == 1, "R_s1_fixo"

        # 2
        for i in range(2, len(self.classes)):
            prob += s[i] - s[i-1] >= 1, f"R_s{i}_ordem_minima"

        # 3
        for i in range(2, len(self.criterias)+1):
            for j in range(1, i):
                # p_i - p_j >= theta
                prob += p[j] - p[i] >= theta, f"R_{j}_{i}_ordem_minima"
        
        # 4
        prob += p[len(self.criterias)] == 0, "R_pMAX_fixo"

        # 5-6
        for k in range(1,len(self.classes)+1):
            for i in range(len(self.criterias)):
                for j in range(i+1,len(self.criterias)):
                    pi = i + 1
                    pj = j + 1
                    pi_index = len(self.classes) - pi
                    pj_index = len(self.classes) - pj
                    if self.judgment_matrix[i][j] == k:
                        if k == len(self.classes): 
                            prob += p[pi] - p[pj] >= theta + s[k-1] - c, f"R_{pi_index}_{pj_index}_classe_{k}_L"
                        else:
                            prob += p[pi] - p[pj] >= theta + s[k-1] - c, f"R_{pi_index}_{pj_index}_classe_{k}_L"
                            prob += p[pi] - p[pj] <= s[k] + c, f"R_{pi_index}_{pj_index}_classe_{k}_U"
        
        prob += c, "Funcao_Objetivo"
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        print(f"Status: {pulp.LpStatus[prob.status]}")
        if prob.status == pulp.LpStatusOptimal:
            c_min_result = pulp.value(prob.objective)
            print(f"\n Valor Mínimo de c (c_min): {c_min_result:.6f}")
            
        return c_min_result
    
    def _MC2(self):
        """Programa MC2 Do MACBETH"""
        prob = pulp.LpProblem("IntervalosDeClasse", pulp.LpMinimize)
        theta = 0.001
        c = self.c_min

        p = {}
        for i in range(1,len(self.criterias)+1):
            p[i] = pulp.LpVariable(f"p{len(self.criterias)+1-i}", lowBound=0) 

        s = {}
        for i in range(0, len(self.classes)):
            s[i] = pulp.LpVariable(f"s{i}", lowBound=0)
        
        # 1
        prob += s[0] == 0, "s0_fixo"
        prob += s[1] == 1, "s1_fixo"

        # 2
        for i in range(2, len(self.classes)):
            prob += s[i] - s[i-1] >= 1, f"s{i}_ordem_minima"

        # 3
        for i in range(2, len(self.criterias)+1):
            for j in range(1, i):
                # p_i - p_j >= theta
                prob += p[j] - p[i] >= theta, f"Rinit_{j}_{i}_ordem_minima"
        
        # 4
        prob += p[len(self.criterias)] == 1, "pmax_fixo"

        # 5' - 9
        beta = {}
        gamma = {}
        alpha = {}
        delta = {}
        epsilon = {}
        eta = {}
        objetivo_eps_eta = []
        objetivo_alpha = []
   
        for i in range(len(self.criterias)):
            for j in range(i+1,len(self.criterias)):
                k = self.judgment_matrix[i][j]  
                if k == 0:
                    continue # ignora indiferenças
                pi = i + 1
                pj = j + 1
                pi_index = len(self.classes) - pi
                pj_index = len(self.classes) - pj
                if k == len(self.classes): 
                    # 6'
                    prob += p[pi] - p[pj] >= theta + s[k-1] - c, f"R_{pi_index}_{pj_index}_classe_{k}_L"
                    
                else:
                    # 5'
                    prob += p[pi] - p[pj] >= theta + s[k-1] - c, f"R_{pi_index}_{pj_index}_classe_{k}_L"
                    prob += p[pi] - p[pj] <= s[k] + c, f"R_{pi_index}_{pj_index}_classe_{k}_U"
                    # 7
                    beta[(pi, pj)] = pulp.LpVariable(f"b_{pi_index}_{pj_index}", lowBound=0)
                    gamma[(pi, pj)] = pulp.LpVariable(f"g_{pi_index}_{pj_index}", lowBound=0)
                    prob += (p[pi] - p[pj] == s[k] + beta[(pi, pj)] - gamma[(pi, pj)]), f"BETAGAM_Eq_{pi_index}_{pj_index}_k{k}"
                    # 9      
                    epsilon[(pi, pj)] = pulp.LpVariable(f"e_{pi_index}_{pj_index}", lowBound=0)
                    eta[(pi, pj)] = pulp.LpVariable(f"n_{pi_index}_{pj_index}", lowBound=0)
                    prob += (p[pi] - p[pj] == epsilon[(pi, pj)] - eta[(pi, pj)] +(s[k] + s[k-1] + theta) / 2), f"EPETA_Eq_{pi_index}_{pj_index}_k{k}" # Centro do intervalo = (s[k] + s[k-1] + theta) / 2
                    # Adiciona na função objetivo
                    objetivo_eps_eta.append(epsilon[(pi, pj)])
                    objetivo_eps_eta.append(eta[(pi, pj)])

                if k != 1:
                    # 8
                    alpha[(pi, pj)] = pulp.LpVariable(f"a_{pi_index}_{pj_index}", lowBound=0)
                    delta[(pi, pj)] = pulp.LpVariable(f"d_{pi_index}_{pj_index}", lowBound=0)
                    prob += (p[pi] - p[pj] == s[k-1] + delta[(pi, pj)] - alpha[(pi, pj)] + theta), f"ALPHADelta_Eq_{pi_index}_{pj_index}_k{k}"
                    if k == len(self.classes):
                        # Adiciona na função objetivo
                        objetivo_alpha.append(alpha[(pi, pj)])
        
        prob += pulp.lpSum(objetivo_eps_eta + objetivo_alpha), "Funcao_Objetivo_MC2_Min_Desvio"
        prob.writeLP("mc2_debug.lp")

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        print(f"Status: {pulp.LpStatus[prob.status]}")
        if prob.status == pulp.LpStatusOptimal:
            for v in prob.variables():
                print(v.name, "=", v.value())
            
            p_dict = {i: p[i].value() for i in p}

            resultados = {
                "status": pulp.LpStatusOptimal,
                "objective": pulp.value(prob.objective),
                "p": {i: p[i].value() for i in p},
                "s": {i: s[i].value() for i in s},
                "alpha": {(i, j): alpha[(i, j)].value() for (i, j) in alpha},
                "beta": {(i, j): beta[(i, j)].value() for (i, j) in beta},
                "gamma": {(i, j): gamma[(i, j)].value() for (i, j) in gamma},
                "delta": {(i, j): delta[(i, j)].value() for (i, j) in delta},
                "epsilon": {(i, j): epsilon[(i, j)].value() for (i, j) in epsilon},
                "eta": {(i, j): eta[(i, j)].value() for (i, j) in eta}
            }
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(p_dict)
            return p_dict, resultados
    
    def _MC3(self):
        """Programa MC3 Do MACBETH"""
        prob = pulp.LpProblem("IntervalosDeClasse", pulp.LpMinimize)
        theta = 0.001
        c = self.c_min

        p = {}
        for i in range(1,len(self.criterias)+1):
            p[i] = pulp.LpVariable(f"p{len(self.criterias)+1-i}", lowBound=0) 

        s = {}
        for i in range(0, len(self.classes)):
            s[i] = pulp.LpVariable(f"s{i}", lowBound=0)
        
        # 1
        prob += s[0] == 0, "s0_fixo"
        prob += s[1] == 1, "s1_fixo"

        # 2
        for i in range(2, len(self.classes)):
            prob += s[i] - s[i-1] >= 1, f"s{i}_ordem_minima"

        # 3
        for i in range(2, len(self.criterias)+1):
            for j in range(1, i):
                # p_i - p_j >= theta
                prob += p[j] - p[i] >= theta, f"Rinit_{j}_{i}_ordem_minima"
        
        # 4
        prob += p[len(self.criterias)] == 1, "pmax_fixo"

        # 5' - 8
        beta = {}
        gamma = {}
        alpha = {}
        delta = {}
        objetivo_alpha_beta = []
        for i in range(len(self.criterias)):
            for j in range(i+1,len(self.criterias)):
                k = self.judgment_matrix[i][j]  
                if k == 0:
                    continue # ignora indiferenças
                pi = i + 1
                pj = j + 1
                pi_index = len(self.classes) - pi
                pj_index = len(self.classes) - pj
                if k == len(self.classes): 
                    # 6'
                    prob += p[pi] - p[pj] >= theta + s[k-1] - c, f"R_{pi_index}_{pj_index}_classe_{k}_L"
                    
                else:
                    # 5'
                    prob += p[pi] - p[pj] >= theta + s[k-1] - c, f"R_{pi_index}_{pj_index}_classe_{k}_L"
                    prob += p[pi] - p[pj] <= s[k] + c, f"R_{pi_index}_{pj_index}_classe_{k}_U"
                    # 7
                    beta[(pi, pj)] = pulp.LpVariable(f"b_{pi_index}_{pj_index}", lowBound=0)
                    gamma[(pi, pj)] = pulp.LpVariable(f"g_{pi_index}_{pj_index}", lowBound=0)
                    prob += (p[pi] - p[pj] == s[k] + beta[(pi, pj)] - gamma[(pi, pj)]), f"BETAGAM_Eq_{pi_index}_{pj_index}_k{k}"
                    objetivo_alpha_beta.append(beta[(pi, pj)])

                if k != 1:
                    # 8
                    alpha[(pi, pj)] = pulp.LpVariable(f"a_{pi_index}_{pj_index}", lowBound=0)
                    delta[(pi, pj)] = pulp.LpVariable(f"d_{pi_index}_{pj_index}", lowBound=0)
                    prob += (p[pi] - p[pj] == s[k-1] + delta[(pi, pj)] - alpha[(pi, pj)] + theta), f"ALPHADelta_Eq_{pi_index}_{pj_index}_k{k}"
                    objetivo_alpha_beta.append(alpha[(pi, pj)])
        
        prob += pulp.lpSum(objetivo_alpha_beta), "Funcao_Objetivo_MC3"
        prob.writeLP("mc3_debug.lp")

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        print(f"Status: {pulp.LpStatus[prob.status]}")
        if prob.status == pulp.LpStatusOptimal:
            for v in prob.variables():
                print(v.name, "=", v.value())
            
            alpha_dict = {i: alpha[i].value() for i in alpha}
            beta_dict = {i: beta[i].value() for i in beta}

            resultados = {
                "status": pulp.LpStatusOptimal,
                "objective": pulp.value(prob.objective),
                "p": {i: p[i].value() for i in p},
                "s": {i: s[i].value() for i in s},
                "alpha": {(i, j): alpha[(i, j)].value() for (i, j) in alpha},
                "beta": {(i, j): beta[(i, j)].value() for (i, j) in beta},
                "gamma": {(i, j): gamma[(i, j)].value() for (i, j) in gamma},
                "delta": {(i, j): delta[(i, j)].value() for (i, j) in delta},
            }
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Alpha")
            print(alpha_dict)
            print("Beta")
            print(beta_dict)
            return alpha_dict, beta_dict, resultados

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
    
    

    
    