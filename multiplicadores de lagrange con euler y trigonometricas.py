"""El algoritmo se ejecuta de tal manera que se le pide al usuario ingresar una funcion objetivo y restricciones, 
luego se aplica el método de multiplicadores de Lagrange para encontrar los puntos críticos y 
se evalúan en la función objetivo. Se filtran las soluciones complejas y se muestran los puntos 
críticos reales junto con sus valores en la función objetivo."""
"""En este algoritmo se permite la entrada de funciones trigonométricas y de Euler,
además de la entrada de funciones algebraicas, teniendo en cuenta lo dicho en el anterior documento,
se requiere poner correctamente la multiplicación y la potencia, además tener encuenta las siguientes restricciones:

-Para utilizar la entrada euler se debe ingresar como 'e' y no 'E'., si este exponente se compone de mas elementos se debe ingresar como 'exp'.
-Para utilizar la entrada de funciones trigonométricas se debe ingresar como 'sin', 'cos', 'tan' y no 'SEN', 'COS', 'TAN'.
-Para utilizar la entrada de funciones algebraicas se debe ingresar como 'x', 'y', 'z' y no 'X', 'Y', 'Z'.
-Para ingresar logarimos se debe ingresar como 'log' y no 'LOG'.
-Para ingresar potencias se debe ingresar como '**' y no como '^'.
-Para optener maximos y minimos (puntos criticos) de una sola funcion sin restricciones se debe ubicar
en restricciones el valor 0, es decir no se debe ingresar ninguna restriccion.

ESTUDIANTE: CAMILO ANDRÉS VELANDIA MENDOZA
CÓDIGO:2232728
GRUPO: B6 
"""

import sympy as sp
import re

def format_expression(expression):
    """Corrige expresiones como '4x+y' a '4*x + y'"""
    expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
    expression = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expression)
    return expression

def lagrange_multipliers(f_expr, g_exprs, vars):
    """Aplica multiplicadores de Lagrange"""
    lambdas = [sp.Symbol(f'lambda_{i}') for i in range(len(g_exprs))]
    grad_f = [sp.diff(f_expr, var) for var in vars]
    grad_g = [[sp.diff(g, var) for g in g_exprs] for var in vars]

    equations = []
    for i in range(len(vars)):
        eq = grad_f[i] - sum(lambdas[j] * grad_g[i][j] for j in range(len(g_exprs)))
        equations.append(eq)

    equations.extend(g_exprs)

    solutions = sp.solve(equations, vars + lambdas, dict=True)
    return solutions

def partial_derivatives(f_expr, vars):
    """Encuentra puntos críticos sin restricciones"""
    grad_f = [sp.diff(f_expr, var) for var in vars]
    solutions = sp.solve(grad_f, vars, dict=True)
    return solutions

def is_real_solution(solution):
    """Filtra soluciones complejas"""
    for value in solution.values():
        if value.as_real_imag()[1] != 0:
            return False
    return True

def evaluate_points(f_expr, solutions):
    """Evalúa f en los puntos encontrados"""
    real_solutions = [sol for sol in solutions if is_real_solution(sol)]
    values = [(sol, f_expr.subs(sol).evalf()) for sol in real_solutions]

    if values:
        max_point = max(values, key=lambda x: x[1])
        min_point = min(values, key=lambda x: x[1])
        return values, max_point, min_point
    else:
        return values, None, None

def main():
    # Soporte de funciones especiales
    funciones_especiales = {
        'e': sp.E,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'exp': sp.exp,
        'log': sp.log
    }

    # Variables
    vars_str = input("Ingrese las variables separadas por comas (ej: x,y,z): ").split(',')
    vars = [sp.Symbol(var.strip()) for var in vars_str]

    # Función objetivo
    f_input = input("Ingrese la función a optimizar f(x, y, ...): ")
    f_input = format_expression(f_input)
    f_expr = sp.sympify(f_input, locals=funciones_especiales)

    # Número de restricciones
    while True:
        try:
            num_restrictions = int(input("Ingrese el número de restricciones: "))
            if num_restrictions < 0:
                print("⚠️ El número de restricciones no puede ser negativo.")
                continue
            break
        except ValueError:
            print("❌ Error: Debe ingresar un número entero.")

    if num_restrictions == 0:
        solutions = partial_derivatives(f_expr, vars)
    else:
        g_exprs = []
        for i in range(num_restrictions):
            while True:
                try:
                    g_input = input(f"Ingrese la restricción {i+1} g(x, y, ...) = 0: ")
                    g_input = format_expression(g_input)
                    g_expr = sp.sympify(g_input, locals=funciones_especiales)
                    g_exprs.append(g_expr)
                    break
                except sp.SympifyError:
                    print("❌ Restricción inválida. Intente de nuevo.")

        solutions = lagrange_multipliers(f_expr, g_exprs, vars)

    if not solutions:
        print("⚠️ No se encontraron soluciones.")
    else:
        print("\n🔹 Puntos críticos encontrados (solo reales):")
        real_solutions = [sol for sol in solutions if is_real_solution(sol)]
        for sol in real_solutions:
            print(sol)

        values, max_point, min_point = evaluate_points(f_expr, solutions)

        print("\n🔍 Evaluación en la función objetivo:")
        for sol, val in values:
            print(f"{sol} -> f = {val}")

        if max_point and min_point:
            print("\n🔴 Máximo global:")
            print(f"{max_point[0]} -> f = {max_point[1]}")

            print("\n✅ Mínimo global:")
            print(f"{min_point[0]} -> f = {min_point[1]}")

if __name__ == "__main__":
    main()
