import sympy
import numpy as np
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy import symbols, diff, integrate, lambdify, sympify, sin, cos, tan, exp, log, sqrt, pi, oo, E, Abs
from sympy.calculus.util import continuous_domain
from sympy.sets import Interval, Reals

x = symbols('x', real=True)

TRANSFORMATIONS = (standard_transformations + (implicit_multiplication_application,))

PRESET_FUNCTIONS = {
    "x*sin(x)": "x*sin(x)",
    "e^(-x^2)": "exp(-x**2)",
    "ln(x)/x": "log(x)/x",
    "x^2*sin(x)": "x**2*sin(x)",
    "1/(x - 2)": "1/(x-2)",
    "sin(x)/x": "sin(x)/x",
    "x^3 - 3*x + 2": "x**3 - 3*x + 2",
    "sqrt(x)": "sqrt(x)",
    "tan(x)": "tan(x)",
}


class CalculusEngine:
    def __init__(self):
        self.expr = None
        self.derivative_expr = None
        self.indefinite_integral = None
        self.lambdified_expr = None
        self.lambdified_derivative = None
        self.domain = None
        self.asymptotes = []

    def parse_expression(self, expr_str: str):
        """Parse user input string into SymPy expression."""
        expr_str = expr_str.replace('^', '**')
        expr_str = expr_str.replace('e^', 'exp')
        expr_str = expr_str.replace('ln', 'log')
        expr_str = expr_str.replace('π', 'pi')
        expr_str = expr_str.replace('pi', 'pi')
        expr_str = expr_str.replace('√x', 'x**0.5')
        expr_str = expr_str.replace('arcsin', 'asin')
        expr_str = expr_str.replace('arccos', 'acos')
        expr_str = expr_str.replace('arctan', 'atan')
        try:
            self.expr = parse_expr(expr_str, transformations=TRANSFORMATIONS, local_dict={'x': x})
            self._compute_derivatives_and_integrals()
            self._lambdify()
            self._detect_asymptotes()
            return True, None
        except Exception as e:
            self.expr = None
            return False, str(e)

    def _compute_derivatives_and_integrals(self):
        if self.expr is None:
            return
        self.derivative_expr = diff(self.expr, x)
        try:
            self.indefinite_integral = integrate(self.expr, x)
        except:
            self.indefinite_integral = None

    def _lambdify(self):
        if self.expr is not None:
            self.lambdified_expr = lambdify(x, self.expr, 'numpy')
            self.lambdified_derivative = lambdify(x, self.derivative_expr, 'numpy')

    def _detect_asymptotes(self):
        self.asymptotes = []
        if self.expr is None:
            return
        
        try:
            from sympy import fraction, denom, solve
            num, den = fraction(self.expr)
            if den != 1:
                denom_roots = solve(den, x)
                for root in denom_roots:
                    if root.is_real:
                        val = float(root)
                        if not np.isnan(val) and not np.isinf(val):
                            self.asymptotes.append(val)
        except:
            pass

    def evaluate(self, x_vals: np.ndarray) -> np.ndarray:
        if self.lambdified_expr is None:
            return np.full_like(x_vals, np.nan)
        try:
            with np.errstate(all='ignore'):
                result = self.lambdified_expr(x_vals)
                return np.asarray(result, dtype=float)
        except:
            return np.full_like(x_vals, np.nan)

    def evaluate_derivative(self, x_vals: np.ndarray) -> np.ndarray:
        if self.lambdified_derivative is None:
            return np.full_like(x_vals, np.nan)
        try:
            with np.errstate(all='ignore'):
                result = self.lambdified_derivative(x_vals)
                return np.asarray(result, dtype=float)
        except:
            return np.full_like(x_vals, np.nan)

    def evaluate_at_point(self, x_val: float) -> float:
        if self.lambdified_expr is None:
            return np.nan
        try:
            result = self.lambdified_expr(np.array([x_val]))
            return float(result[0])
        except:
            return np.nan

    def derivative_at_point(self, x_val: float) -> float:
        if self.lambdified_derivative is None:
            return np.nan
        try:
            result = self.lambdified_derivative(np.array([x_val]))
            return float(result[0])
        except:
            return np.nan

    def definite_integral(self, a: float, b: float):
        if self.expr is None:
            return None, None
        try:
            result = integrate(self.expr, (x, a, b))
            numeric = float(result.evalf())
            return result, numeric
        except Exception as e:
            return None, str(e)

    def get_latex(self, expr):
        if expr is None:
            return ""
        return sympy.latex(expr)

    def get_function_latex(self):
        return self.get_latex(self.expr)

    def get_derivative_latex(self):
        return self.get_latex(self.derivative_expr)

    def get_integral_latex(self):
        return self.get_latex(self.indefinite_integral)

    def validate_syntax(self, expr_str: str):
        """Quick syntax validation: returns (is_valid, error_message)"""
        expr_str = expr_str.replace('^', '**')
        open_parens = expr_str.count('(')
        close_parens = expr_str.count(')')
        if open_parens != close_parens:
            return False, "Unclosed parenthesis"
        open_brackets = expr_str.count('[')
        close_brackets = expr_str.count(']')
        if open_brackets != close_brackets:
            return False, "Unclosed bracket"
        try:
            parse_expr(expr_str, transformations=TRANSFORMATIONS, local_dict={'x': x})
            return True, None
        except Exception as e:
            return False, str(e)