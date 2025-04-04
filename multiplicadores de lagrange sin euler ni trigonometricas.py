"""El algoritmo se ejecuta de tal manera que se le pide al usuario ingresar una funcion objetivo y restricciones, 
luego se aplica el método de multiplicadores de Lagrange para encontrar los puntos críticos y se evalúan en la función objetivo. 
Se filtran las soluciones complejas y se muestran los puntos críticos reales junto con sus valores en la función objetivo."""
"""El algoritmo no permite entrada de euler ni trigonometricas, este algoritmo fue el inicialmente presentado,
la unica falla era que al multiplicar 4 por x no estabamos implementando el * entonces
el algoritmo no lo reconocia como una multiplicacion, por lo que se le implemento una 
funcion que corrige el formato de la entrada del usuario, 
para que el algoritmo pueda reconocerlo como una multiplicacion (lo optimo sería ubicar el *).

EL ALGORITMO SE PERMITE EJECUTAR CORRECTAMENTE SI SE MULTIPLICA ES DECIR XY SE ESCRIBIRIA X*Y PERO EN MINUSCULAS, 
ADEMÁS PARA ELEVAR AL CUADRADO SE ESCRIBIRIA X**2, O SEA QUE NO SE PERMITE LA ENTRADA DE FUNCIONES TRIGONOMETRICAS NI DE EULER 
EN EL CASO DE ESTE DOCUMENTO (las mayusculas eran necesarias para optimizar el algoritmo y no tener errores de entrada)
-Para optener maximos y minimos (puntos criticos) de una sola funcion sin restricciones se debe ubicar
en restricciones el valor 0, es decir no se debe ingresar ninguna restriccion.

ESTUDIANTE: CAMILO ANDRÉS VELANDIA MENDOZA
CÓDIGO:2232728
GRUPO: B6 
"""

import sympy as sp
import re

def format_expression(expression):
    """ Corrige expresiones como '-4x+y**2' a '-4*x + y**2' """
    expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)  # Añadir * entre número y variable
    expression = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expression)  # Añadir * entre variable y número
    return expression

def lagrange_multipliers(f_expr, g_exprs, vars):
    """ Aplica multiplicadores de Lagrange y resuelve el sistema de ecuaciones """
    lambdas = [sp.Symbol(f'lambda_{i}') for i in range(len(g_exprs))]
    grad_f = [sp.diff(f_expr, var) for var in vars]
    grad_g = [[sp.diff(g, var) for g in g_exprs] for var in vars]

    equations = []
    for i in range(len(vars)):
        eq = grad_f[i] - sum(lambdas[j] * grad_g[i][j] for j in range(len(g_exprs)))
        equations.append(eq)

    equations.extend(g_exprs)  # Agregar restricciones

    solutions = sp.solve(equations, vars + lambdas, dict=True)  # Resolver el sistema
    return solutions  # Devolver todas las soluciones posibles

def partial_derivatives(f_expr, vars):
    """ Encuentra los puntos críticos de la función objetivo usando derivadas parciales """
    grad_f = [sp.diff(f_expr, var) for var in vars]
    solutions = sp.solve(grad_f, vars, dict=True)  # Resolver el sistema de ecuaciones gradiente
    return solutions

def is_real_solution(solution):
    """ Verifica si todas las variables en la solución son reales """
    for value in solution.values():
        if value.as_real_imag()[1] != 0:  # Si tiene parte imaginaria, descartar
            return False
    return True

def evaluate_points(f_expr, solutions):
    """ Evalúa cada punto crítico en la función objetivo, ignorando valores complejos """
    real_solutions = [sol for sol in solutions if is_real_solution(sol)]  # Filtrar solo soluciones reales

    values = [(sol, f_expr.subs(sol).evalf()) for sol in real_solutions]
    
    if values:
        max_point = max(values, key=lambda x: x[1])  # Punto con el valor máximo
        min_point = min(values, key=lambda x: x[1])  # Punto con el valor mínimo
        return values, max_point, min_point
    else:
        return values, None, None

def main():
    # Ingresar variables
    vars_str = input("Ingrese las variables separadas por comas (ej: x,y,z): ").split(',')
    vars = [sp.Symbol(var.strip()) for var in vars_str]

    # Ingresar función a optimizar
    f_input = input("Ingrese la función a optimizar f(x, y, ...): ")
    f_input = format_expression(f_input)  # Corregir formato
    f_expr = sp.sympify(f_input)

    # Ingresar número de restricciones
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
        # No hay restricciones, usar derivadas parciales
        solutions = partial_derivatives(f_expr, vars)
    else:
        # Ingresar restricciones
        g_exprs = []
        for i in range(num_restrictions):
            while True:
                try:
                    g_input = input(f"Ingrese la restricción {i+1} g(x, y, ...) = 0: ")
                    g_input = format_expression(g_input)  # Corregir formato
                    g_exprs.append(sp.sympify(g_input))
                    break
                except sp.SympifyError:
                    print("❌ Error: La restricción ingresada no es válida. Inténtelo de nuevo.")

        # Hallar puntos críticos usando multiplicadores de Lagrange
        solutions = lagrange_multipliers(f_expr, g_exprs, vars)

    if not solutions:
        print("⚠️ No se encontraron soluciones factibles.")
    else:
        print("\n🔹 Puntos críticos encontrados (Solo reales):")
        real_solutions = [sol for sol in solutions if is_real_solution(sol)]
        for sol in real_solutions:
            print(sol)

        # Evaluar puntos en la función objetivo
        values, max_point, min_point = evaluate_points(f_expr, solutions)

        print("\n🔍 Evaluación en la función objetivo:")
        for sol, val in values:
            print(f"{sol} -> f = {val}")

        # Mostrar el máximo y mínimo global
        if max_point and min_point:
            print("\n🔴 Máximo global:")
            print(f"{max_point[0]} -> f = {max_point[1]}")

            print("\n✅ Mínimo global:")
            print(f"{min_point[0]} -> f = {min_point[1]}")

if __name__ == "__main__":
    main()

