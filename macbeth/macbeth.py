import csv, pulp
from colorama import Fore, Style

class Macbeth:
    def __init__(self):
        self.criterias = []
        self.alternatives = []
        self.judgment_matrix = []
        self.classes = {"Very Weak": 1.0, "Weak": 2.0, "Moderate": 3.0, "Strong": 4.0, "Very Strong": 5.0, "Extreme": 6.0}
        self.classesBoundaries = []
        self.consistence_checked = False
        self.consistent_judgment = False
        self.weights_evaluated = False
        self.c_min = None
        self.minimun_rank_value = 1
        self.sensibility_value = 0.001 #sugerido 0.001 ou 0.0001

    def set_minimun_rank_value(self, value):
        """Define um valor de base para o critério de menor importância."""
        self.weights_evaluated = False
        self.minimun_rank_value = value
    
    def set_sensibility_value(self, value):
        """
        Define o valor de sensibilidade (theta) para os programas MC.
        Sugerido 0.001 ou 0.0001.
        """
        self.consistence_checked = False
        self.consistent_judgment = False
        self.weights_evaluated = False
        self.sensibility_value = value
    
    def add_criteria(self, name, type="+"):
        self.criterias.append(Criteria(name, type))
        self._expand_matrix()
        self.consistence_checked = False
        self.consistent_judgment = False
        self.weights_evaluated = False

    def add_alternative(self, name):
        self.alternatives.append(Alternative(name))

    def _expand_matrix(self):
        n = len(self.criterias)
        for row in self.judgment_matrix:
            row.append(0.0) # o correto é usar none para checar se foi preenchido, alterar depois
        self.judgment_matrix.append([0.0] * n)

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
                print(f"{i}. {c.name} (weight={c.weight:.3f}, type={c.type})")
            else:
                print(f"{i}. {c.name}")
    
    def show_judgment_matrix(self):
        """Exibe a matriz de julgamentos atual."""
        names = [c.name for c in self.criterias]
        self._print_matrix_preview(names, self.judgment_matrix, "Current Judgment Matrix")

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
        self.consistent_judgment = False
        self.weights_evaluated = False
        with open(filepath, newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            data = list(reader) 

        n = len(data)
        if n != len(self.criterias):        # precisa ter mesmo número de critérios
            raise ValueError("Número de critérios não bate com a matriz.")

        #self.judgment_matrix = [[float(x) for x in row] for row in data]
        valid_values = set(self.classes.values())

        for i, row in enumerate(data):
            converted_row = []
            for j, x in enumerate(row):
                value = float(x)
                if value not in valid_values:
                    raise ValueError(f"Invalid judgment value at position ({i+1},{j+1}): {value}")
                converted_row.append(value)
            self.judgment_matrix.append(converted_row)
    
    def import_criterias_and_judments_from_csv(self, filepath):
        """Importa matriz de julgamentos de um CSV e recria critérios e matriz."""
        self.consistence_checked = False
        self.consistent_judgment = False
        self.weights_evaluated = False
        with open(filepath, newline='', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=';')
            data = list(reader)

        if not data or len(data) < 2:
            raise ValueError("Arquivo CSV vazio ou inválido.")

        col_names = [h.strip() for h in data[0][1:]]
        n = len(col_names)

        self.criterias = [Criteria(name) for name in col_names] # recria a lista com base nos nomes do CSV
        valid_values = set(self.classes.values())    
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]

        for i, row in enumerate(data[1:]):  # ignora o cabeçalho
            row_name = row[0].strip()
            if row_name != col_names[i]:
                print(f"Aviso: critério da linha {i+1} ({row_name}) difere do cabeçalho ({col_names[i]})")
            for j, val in enumerate(row[1:]):
                try:
                    value = float(val)
                except ValueError:
                    raise ValueError(f"Valor inválido na posição ({i},{j}): '{val}'")
                if (value not in valid_values) and value != 0.0:
                    raise ValueError(f"Judgment value at position ({i},{j}): {value} does not match valid classes.")
                matrix[i][j] = value
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
    
    def judge(self, criteria1, criteria2, value):
        """Adiciona um julgamento entre dois critérios na matriz de julgamentos."""
        if criteria1 == criteria2:
            raise ValueError("Cannot set judgment between the same criteria.")
        if value not in self.classes.values():
            raise ValueError("Judgment value does not match valid classes.")
        for i, c in enumerate(self.criterias, start=1):
            if c.name == criteria1:
                idx1 = i - 1
            if c.name == criteria2:
                idx2 = i - 1
        if idx2<idx1:
            idx1, idx2 = idx2, idx1  # supoe-se que usuário errou a ordem
        self.judgment_matrix[idx1][idx2] = float(value)
        self.consistence_checked = False
        self.consistent_judgment = False
        self.weights_evaluated = False

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
                        v = float(val)
                        if 0 <= v <= 6:
                            matrix[i][j] = v
                            break
                        else:
                            print("Invalid value. Please enter a number between 0 and 6.")
                    except ValueError:
                        print("Invalid input. Please enter a number between 0 and 6.")

                # Exibe a matriz parcial
                self._print_matrix_preview(names, matrix, "Judgment Matrix Preview")

        self._print_matrix_preview(names, matrix, "Final proposed matrix")
        confirm = input("\nConfirm and replace the current judgment matrix? (Y/N): ").strip().lower()
        if confirm == "y":
            self.judgment_matrix = matrix
            self.consistence_checked = False
            self.c_min = None
            self.consistent_judgment = False
            self.weights_evaluated = False
            print("Judgment matrix successfully updated!")
        else:
            print("Matrix discarded.")

    def _print_matrix_preview(self, names, matrix, title=""):
        """
        Imprime a matriz de julgamentos em formato tabular, com alinhamento dinâmico.
        Exibe uma estrutura semelhante à matriz colorida, mas sem setas ou cores.
        """
        print(f"\n{title}\n")

        n = len(matrix)
        # Calcula a largura ideal de cada coluna com base no maior nome
        col_width = max(len(str(name)) for name in names) + 8

        # Cabeçalho formatado
        header = " " * col_width + "".join(f"{name:^{col_width}}" for name in names)
        print(header)
        print(" " * (col_width - 2) + "-" * (col_width * n + 2))

        # Corpo da matriz
        for i in range(n):
            linha_str = f"{names[i]:<{col_width-2}}| "
            for j in range(n):
                # Usa ponto (·) na diagonal inferior, valor normal acima da diagonal
                if j <= i:
                    cell = "·".center(col_width)
                else:
                    val = matrix[i][j]
                    texto = f"{val:^5}"
                    cell = texto.center(col_width)
                linha_str += cell
            print(linha_str)

    def _print_colored_matrix(self, alpha, beta, title = ""):
        """
        Função auxiliar para imprimir a matriz de julgamentos com destaque para inconsistências.
        Recebe dicionários alpha e beta (pares (i,j): valor), e um título descritivo.
        """
        print(f"\n{title}\n")
        n = len(self.judgment_matrix)
        criterios = [c.name for c in self.criterias]
        # Largura dinâmica de coluna
        col_width = max(len(c) for c in criterios) + 8
        # Cabeçalho
        header = " " * (col_width) + "".join(f"{nome:^{col_width}}" for nome in criterios)
        print(header)
        print(" " * (col_width - 2) + "-" * (col_width * n + 2))
        # Corpo da matriz
        for i in range(n):
            linha_str = f"{criterios[i]:<{col_width-2}}| "
            for j in range(n):
                if j <= i:
                    cell = "·".center(col_width)
                else:
                    val = self.judgment_matrix[i][j]
                    simbolo, cor = "", ""
                    if (i+1, j+1) in alpha:  # destacar α (reduzir)
                        simbolo = "↓"
                        cor = Fore.RED
                    elif (i+1, j+1) in beta:  # destacar β (aumentar)
                        simbolo = "↑"
                        cor = Fore.GREEN
                    texto = f"{val:^5}{simbolo:^2}"
                    cell = f"{cor}{texto.center(col_width)}{Style.RESET_ALL}"
                linha_str += cell
            print(linha_str)
        # Legenda
        print("\nLegenda:")
        print(f"{Fore.RED}↓{Style.RESET_ALL} → reduzir a classe (alpha)")
        print(f"{Fore.GREEN}↑{Style.RESET_ALL} → aumentar a classe (beta)\n")
    
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
        print(f"\n Valor Mínimo de c (c_min): {self.c_min:.6f}")
        self.consistence_checked = True
        self.consistent_judgment = self.c_min <= self.sensibility_value*10 #verificar esse limiar
        if self.consistent_judgment:
            print("The judgment matrix is consistent.")
        else:
            print("The judgment matrix is NOT consistent.")
            print("\n Run hilight_inconsistencies() to check which judgments are inconsistent.")
        return self.consistent_judgment
    
    def highlight_inconsistencies(self, analize=False):
        """
        Hilight the inconsistencys in the judgment matrix based on MC3.
        This function indicates which judgments contribute to inconsistency, maybe one
        or more judgments need to be revised.        
        """
        if self.consistence_checked:
            if self.consistent_judgment:
                print("\n The judgment matrix is consistent. No inconsistencies to highlight.")
                return
            else:
                alpha, beta = self._MC3()
                filtered_alpha = {k: v for k, v in alpha.items() if v >= 1e-6}
                filtered_beta  = {k: v for k, v in beta.items() if v >= 1e-6}
                self._print_colored_matrix(filtered_alpha, filtered_beta,"Judgment matrix with inconsistancies:")
                if analize:
                    print("\n Analysis of inconsistencies: \n")
                    print("\n Alpha (↓) - Inferior violation of classes:")
                    print(alpha)
                    print("\n Beta (↑) - Superior violation of classes:")
                    print(beta) 
                    print("\n")
                print("\n Run sugest_corrections() to calculate best solution")
        else:
            print("Consistency not checked. Please run check_consistency() first.")
            return
    
    def sugest_corrections(self, analize=False):
        """Sugere correções para os julgamentos inconsistentes com base no programa MC4."""
        if self.consistence_checked:
            if self.consistent_judgment:
                print("\n The judgment matrix is consistent. No corrections needed.")
                return
            else:
                # Exemplo
                alpha, beta = self._MC4()
                max_alpha = max(alpha.values()) if alpha else 0
                max_beta = max(beta.values()) if beta else 0
                limite = 0.3 * max(max_alpha, max_beta)  # 30% do valor máximo
                filtered_alpha = {k: v for k, v in alpha.items() if v >= limite}
                filtered_beta  = {k: v for k, v in beta.items() if v >= limite}
                self._print_colored_matrix(filtered_alpha, filtered_beta,"Suggested corrections for inconsistent judgments:")
                if analize:
                    print("\n Analysis of suggested corrections:")
                    print("\n Alpha (↓) - Inferior violation of classes:")
                    print(alpha)
                    print("\n Beta (↑) - Superior violation of classes:")
                    print(beta) 
                    print("\n") 
        else:
            print("Consistency not checked. Please run check_consistency() first.")
            return
        
    def _MC1(self):
        """Programa MC1 de MACBETH para determinar o valor de incoerência c_min."""
        prob = pulp.LpProblem("Minimizar_c", pulp.LpMinimize)
        theta = self.sensibility_value
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
        prob += p[len(self.criterias)] == self.minimun_rank_value, "R_pMAX_fixo"

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
        prob.writeLP("mc1_debug.lp")

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[prob.status]

        if prob.status == pulp.LpStatusOptimal:
            c_min_result = pulp.value(prob.objective)
            return c_min_result
        else:
            raise RuntimeError(
            f"The optimization problem was not successfully solved.\n"
            f"Status: {status}\n"
            "Please check if the constraints are consistent and if the parameters are correct."
            )
    
    def _MC2(self):
        """Programa MC2 Do MACBETH"""
        prob = pulp.LpProblem("IntervalosDeClasse", pulp.LpMinimize)
        theta = self.sensibility_value
        c = self.c_min

        p = {}
        for i in range(1,len(self.criterias)+1):
            p[i] = pulp.LpVariable(f"p{i}", lowBound=0)

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
                prob += p[j] - p[i] >= theta, f"Rinit_{i}_{j}_ordem_minima"
        
        # 4
        prob += p[len(self.criterias)] == self.minimun_rank_value, "pmax_fixo"

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
                pi_index = len(self.classes) - pi+1
                pj_index = len(self.classes) - pj+1
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
        status = pulp.LpStatus[prob.status]

        if prob.status == pulp.LpStatusOptimal:
            p_dict = {i: p[i].value() for i in p}
            return p_dict
        else:
            raise RuntimeError(
            f"The optimization problem was not successfully solved.\n"
            f"Status: {status}\n"
            "Please check if the constraints are consistent and if the parameters are correct."
            )
            
    def _MC3(self):
        """Programa MC3 Do MACBETH"""
        prob = pulp.LpProblem("IntervalosDeClasse", pulp.LpMinimize)
        theta = self.sensibility_value
        c = self.c_min

        p = {}
        for i in range(1,len(self.criterias)+1):
            p[i] = pulp.LpVariable(f"p{i}", lowBound=0)

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
        prob += p[len(self.criterias)] == self.minimun_rank_value, "pmax_fixo"

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
        status = pulp.LpStatus[prob.status]

        if prob.status == pulp.LpStatusOptimal:
            alpha_dict = {i: alpha[i].value() for i in alpha}
            beta_dict = {i: beta[i].value() for i in beta}
            return alpha_dict, beta_dict
        else:
            raise RuntimeError(
            f"The optimization problem was not successfully solved.\n"
            f"Status: {status}\n"
            "Please check if the constraints are consistent and if the parameters are correct."
            )

    def _MC4(self):
        """Programa MC4 Do MACBETH"""
        prob = pulp.LpProblem("IntervalosDeClasse", pulp.LpMinimize)
        theta = self.sensibility_value

        p = {}
        for i in range(1,len(self.criterias)+1):
            p[i] = pulp.LpVariable(f"p{i}", lowBound=0) 

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
                prob += p[i] - p[j] >= theta, f"Rinit_{i}_{j}_ordem_minima"
        
        # 4
        prob += p[1] == self.minimun_rank_value, "pmin_fixo"

        # 7 - 8
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
                pi_index = len(self.classes) - pi+1
                pj_index = len(self.classes) - pj+1
                if k != len(self.classes): 
                    # 7
                    beta[(pi, pj)] = pulp.LpVariable(f"b_{pi_index}_{pj_index}", lowBound=0)
                    gamma[(pi, pj)] = pulp.LpVariable(f"g_{pi_index}_{pj_index}", lowBound=0)
                    prob += (p[pj] - p[pi] == s[k] + beta[(pi, pj)] - gamma[(pi, pj)]), f"BETAGAM_Eq_{pi_index}_{pj_index}_k{k}"
                    objetivo_alpha_beta.append(beta[(pi, pj)])

                if k != 1:
                    # 8
                    alpha[(pi, pj)] = pulp.LpVariable(f"a_{pi_index}_{pj_index}", lowBound=0)
                    delta[(pi, pj)] = pulp.LpVariable(f"d_{pi_index}_{pj_index}", lowBound=0)
                    prob += (p[pj] - p[pi] == s[k-1] + delta[(pi, pj)] - alpha[(pi, pj)] + theta), f"ALPHADelta_Eq_{pi_index}_{pj_index}_k{k}"
                    objetivo_alpha_beta.append(alpha[(pi, pj)])
        
        prob += pulp.lpSum(objetivo_alpha_beta), "Funcao_Objetivo_MC4"
        prob.writeLP("mc4_debug.lp")

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[prob.status]

        if prob.status == pulp.LpStatusOptimal:            
            alpha_dict = {i: alpha[i].value() for i in alpha}
            beta_dict = {i: beta[i].value() for i in beta}
            return alpha_dict, beta_dict
        else:
            raise RuntimeError(
            f"The optimization problem was not successfully solved.\n"
            f"Status: {status}\n"
            "Please check if the constraints are consistent and if the parameters are correct."
            )
    
    def evaluate_weights(self, force = False):
        """Avalia os pesos dos critérios usando o programa MC2."""
        if not self.consistence_checked:
            print("Consistency not checked. Please run check_consistency() first.")
            return
        if force and self.consistent_judgment:
            if self.weights_evaluated:
                print("Judgement matrix is consistent and weights are already evaluated. No re-evaluation...")
                return
            else:
                print("Judgement matrix is consistent, no need to force re-evaluation...")
        elif not self.consistent_judgment and not force:
            print("The judgment matrix is NOT consistent. Please revise judgments or run sugest_corrections() first.")
            return
        p_dict = self._MC2()
        print(p_dict)
        # Normaliza os pesos
        total = sum(p_dict.values())
        for i, c in enumerate(self.criterias, start=1):
            weight = p_dict[i] / total if total > 0 else 0
            c.set_weight(weight)
        self.weights_evaluated = True
        print("\n Criteria weights successfully evaluated and updated.")
        self.show_criteria(detailed=True)
        
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
    
    

    
    