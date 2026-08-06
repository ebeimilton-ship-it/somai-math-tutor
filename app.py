"""
SomAi Math Tutor — Gradio-Lite + Browser-Based Python (Pyodide)
Runs 100% in the browser using WebAssembly — no server needed!

Deploy as "Static" SDK on HuggingFace Spaces.
"""

import gradio as gr
from sympy import *
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from fractions import Fraction
import re
import math

# ============================================================================
# MATH KNOWLEDGE BASE — Rule-based explanations for math topics
# ============================================================================

MATH_TOPICS = {
    "calculus": {
        "derivatives": "The derivative measures the rate of change of a function at a point. For a function f(x), the derivative f'(x) = lim(h→0) [f(x+h) - f(x)] / h. Power rule: d/dx[x^n] = n·x^(n-1). Product rule: (fg)' = f'g + fg'. Chain rule: (f∘g)' = f'(g) · g'.",
        "integrals": "Integration is the reverse of differentiation. The indefinite integral ∫f(x)dx finds all functions whose derivative is f(x). Power rule: ∫x^n dx = x^(n+1)/(n+1) + C (n ≠ -1). Definite integrals compute area under curves: ∫[a to b] f(x)dx.",
        "limits": "A limit describes the value that a function approaches as the input approaches some value. lim(x→a) f(x) = L means f(x) gets arbitrarily close to L as x approaches a. Key: L'Hôpital's rule for indeterminate forms (0/0 or ∞/∞): lim f/g = lim f'/g'.",
        "continuity": "A function is continuous at x=a if: (1) f(a) is defined, (2) lim(x→a) f(x) exists, and (3) lim(x→a) f(x) = f(a). Continuous functions have no jumps or holes.",
    },
    "algebra": {
        "quadratic": "A quadratic equation ax² + bx + c = 0 can be solved using: Quadratic formula: x = [-b ± √(b² - 4ac)] / (2a). Discriminant Δ = b² - 4ac tells you: Δ > 0 (two real solutions), Δ = 0 (one solution), Δ < 0 (complex solutions).",
        "factoring": "Factoring breaks down polynomials into simpler factors. Common patterns: a² - b² = (a-b)(a+b) (difference of squares), a² + 2ab + b² = (a+b)² (perfect square trinomial), ax² + bx + c = a(x-r₁)(x-r₂) where r₁, r₂ are roots.",
        "exponents": "Exponent rules: x^a · x^b = x^(a+b), x^a / x^b = x^(a-b), (x^a)^b = x^(ab), x^(-a) = 1/x^a, x^(1/n) = ⁿ√x, x^0 = 1 (for x≠0).",
        "logs": "Logarithms are the inverse of exponentials. log_b(x) = y means b^y = x. Change of base: log_b(x) = ln(x)/ln(b). Properties: log(ab) = log(a) + log(b), log(a/b) = log(a) - log(b), log(a^b) = b·log(a).",
    },
    "linear algebra": {
        "vectors": "A vector is an ordered list of numbers (components). Vector operations: addition (add component-wise), scalar multiplication (multiply each component), dot product u·v = u₁v₁ + u₂v₂ + ... + uₙvₙ, magnitude ||v|| = √(v₁² + v₂² + ... + vₙ²).",
        "matrices": "A matrix is a rectangular array of numbers. Matrix multiplication: (AB)ᵢⱼ = row i of A · column j of B. Determinant (for 2×2): det([a,b; c,d]) = ad - bc. Inverse A⁻¹ satisfies A·A⁻¹ = I (identity).",
        "eigenvalues": "For square matrix A, eigenvalue λ and eigenvector v satisfy: Av = λv. Find eigenvalues by solving det(A - λI) = 0. Eigenvectors point in directions where A just scales by λ.",
    },
    "probability": {
        "basic": "Probability P(event) = (favorable outcomes) / (total outcomes), ranges 0 to 1. Sum rule: P(A or B) = P(A) + P(B) - P(A and B). Product rule: P(A and B) = P(A) · P(B|A) (conditional).",
        "distributions": "Binomial: P(X=k) = C(n,k) · p^k · (1-p)^(n-k) for n trials, k successes, probability p. Normal (Gaussian): bell curve, defined by mean μ and standard deviation σ. Poisson: for rare events occurring at rate λ in time t.",
        "bayes": "Bayes' theorem: P(A|B) = P(B|A) · P(A) / P(B). Relates conditional probabilities. Used for: disease testing (likelihood given positive test), spam filtering, machine learning classification.",
    },
}

# ============================================================================
# HELPER: Parse and solve math expressions
# ============================================================================

def try_solve_expression(expr_str: str) -> str:
    """
    Attempt to parse and simplify/solve a math expression.
    Returns LaTeX-formatted result or error message.
    """
    try:
        expr_str = expr_str.strip()
        transformations = (standard_transformations + (implicit_multiplication_application,))
        expr = parse_expr(expr_str, transformations=transformations)
        
        # Try to simplify
        simplified = simplify(expr)
        
        # Try to factor if it's a polynomial
        try:
            factored = factor(expr)
            if factored != expr:
                return f"Simplified: \\({latex(simplified)}\\)\n\nFactored: \\({latex(factored)}\\)"
        except:
            pass
        
        return f"Result: \\({latex(simplified)}\\)"
    except Exception as e:
        return f"Could not parse expression: {str(e)[:100]}"

def try_solve_equation(eq_str: str) -> str:
    """
    Attempt to solve an equation (format: "expression = 0" or "left = right").
    Returns LaTeX-formatted solutions.
    """
    try:
        eq_str = eq_str.strip()
        transformations = (standard_transformations + (implicit_multiplication_application,))
        
        # Handle "lhs = rhs" format
        if "=" in eq_str:
            lhs_str, rhs_str = eq_str.split("=", 1)
            lhs = parse_expr(lhs_str, transformations=transformations)
            rhs = parse_expr(rhs_str, transformations=transformations)
            equation = Eq(lhs, rhs)
        else:
            expr = parse_expr(eq_str, transformations=transformations)
            equation = Eq(expr, 0)
        
        # Detect variable (usually x, or the first free symbol)
        var = list(equation.free_symbols)[0] if equation.free_symbols else Symbol('x')
        
        # Solve
        solutions = solve(equation, var)
        
        if not solutions:
            return f"No solutions found for: \\({latex(equation)}\\)"
        
        result = f"Solutions:\n"
        for sol in solutions:
            result += f"• \\({latex(var)}\\) = \\({latex(sol)}\\)\n"
        
        return result
    except Exception as e:
        return f"Could not solve equation: {str(e)[:100]}"

def try_differentiate(expr_str: str, var_str: str = "x") -> str:
    """
    Differentiate an expression with respect to a variable.
    """
    try:
        expr_str = expr_str.strip()
        transformations = (standard_transformations + (implicit_multiplication_application,))
        expr = parse_expr(expr_str, transformations=transformations)
        var = Symbol(var_str)
        
        derivative = diff(expr, var)
        simplified = simplify(derivative)
        
        return f"d/d{var_str}[\\({latex(expr)}\\)] = \\({latex(simplified)}\\)"
    except Exception as e:
        return f"Could not differentiate: {str(e)[:100]}"

def try_integrate(expr_str: str, var_str: str = "x") -> str:
    """
    Integrate an expression with respect to a variable.
    """
    try:
        expr_str = expr_str.strip()
        transformations = (standard_transformations + (implicit_multiplication_application,))
        expr = parse_expr(expr_str, transformations=transformations)
        var = Symbol(var_str)
        
        integral = integrate(expr, var)
        
        return f"∫\\({latex(expr)}\\)d{var_str} = \\({latex(integral)}\\) + C"
    except Exception as e:
        return f"Could not integrate: {str(e)[:100]}"

# ============================================================================
# MAIN: Identify topic and generate response
# ============================================================================

def detect_topic(user_input: str) -> tuple[str, str]:
    """
    Detect the math topic and category from user input.
    Returns (topic, category) or ("unknown", "general").
    """
    user_lower = user_input.lower()
    
    for topic, categories in MATH_TOPICS.items():
        for category in categories.keys():
            if category in user_lower or topic in user_lower:
                return topic, category
    
    return "unknown", "general"

def math_tutor(message: str, history: list) -> str:
    """
    Main chat function for the math tutor.
    Detects the topic and provides an explanation, or solves an expression.
    """
    message = message.strip()
    
    if not message:
        return "Please ask me a math question or provide an expression to work with."
    
    topic, category = detect_topic(message)
    
    # ========================================================================
    # SECTION 1: Math expression solving (if input looks like math)
    # ========================================================================
    
    has_math_symbols = any(c in message for c in "x^()[]{}+-*/∫∑d√")
    
    if has_math_symbols:
        # Try to detect what to do
        if any(word in message.lower() for word in ["simplify", "solve", "expand"]):
            # Solve expression
            expr_match = re.search(r"(?:simplify|solve|expand)\s+(.+?)(?:\s+for\s+|$)", message, re.IGNORECASE)
            if expr_match:
                expr_str = expr_match.group(1).strip()
                if "=" in expr_str:
                    result = try_solve_equation(expr_str)
                else:
                    result = try_solve_expression(expr_str)
                
                return f"**Math Solver**\n\n{result}\n\n---\n\n**Explanation:** I used SymPy to parse and solve your expression. Feel free to ask for more details!"
        
        elif any(word in message.lower() for word in ["derivative", "differentiate", "d/dx"]):
            expr_match = re.search(r"(?:derivative|differentiate|d/dx)\s+(?:of\s+)?(.+?)(?:\s+with respect to|$)", message, re.IGNORECASE)
            if expr_match:
                expr_str = expr_match.group(1).strip()
                result = try_differentiate(expr_str)
                return f"**Derivative Solver**\n\n{result}\n\n---\n\n**Quick Rule:** Power rule for \\(x^n\\): derivative is \\(n \\cdot x^{{n-1}}\\). Chain rule: \\((f \\circ g)' = f'(g) \\cdot g'\\)"
        
        elif any(word in message.lower() for word in ["integral", "integrate", "∫"]):
            expr_match = re.search(r"(?:integral|integrate|∫)\s+(?:of\s+)?(.+?)(?:\s+with respect to|$)", message, re.IGNORECASE)
            if expr_match:
                expr_str = expr_match.group(1).strip()
                result = try_integrate(expr_str)
                return f"**Integration Solver**\n\n{result}\n\n---\n\n**Quick Rule:** Power rule for \\(x^n\\): integral is \\(\\frac{{x^{{n+1}}}}{{n+1}} + C\\) (for \\(n \\neq -1\\))."
        
        else:
            # Generic expression simplification
            result = try_solve_expression(message)
            return f"**Expression Simplified**\n\n{result}"
    
    # ========================================================================
    # SECTION 2: Topic-based explanations (from knowledge base)
    # ========================================================================
    
    if topic in MATH_TOPICS and category in MATH_TOPICS[topic]:
        explanation = MATH_TOPICS[topic][category]
        return f"**{topic.title()} — {category.title()}**\n\n{explanation}\n\n---\n\n**Try asking:** Can you solve this? Give me an example? Explain [specific concept]?"
    
    # ========================================================================
    # SECTION 3: Fallback for unrecognized topics
    # ========================================================================
    
    return f"""**I'm SomAi, your Maseno University Math Tutor!**

I can help with:
✓ **Calculus:** derivatives, integrals, limits, continuity  
✓ **Algebra:** quadratic equations, factoring, exponents, logarithms  
✓ **Linear Algebra:** vectors, matrices, eigenvalues  
✓ **Probability:** distributions, Bayes' theorem, combinations  

**I can also:**
• Solve equations (e.g., "solve x² + 3x - 4 = 0")
• Simplify expressions (e.g., "simplify (x² + 2x + 1)")
• Compute derivatives (e.g., "differentiate x³ + 2x")
• Integrate functions (e.g., "integrate sin(x)")

Your question: "{message}"

Try rephrasing with a math topic or provide an expression to solve!"""


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(
    title="SomAi Math Tutor",
    theme=gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="slate"
    ),
    css="""
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        max-width: 700px;
    }
    h1 { color: #B8860B; font-weight: 700; }
    .description { color: #666; font-size: 14px; line-height: 1.6; }
    """
) as demo:
    
    gr.Markdown("""
    # 📚 SomAi Math Tutor
    
    Ask me questions about Maseno University mathematics — or give me an expression to solve!
    
    **Powered by:** Pure Python (SymPy) + WebAssembly (Gradio-Lite)  
    **Cost:** $0 — runs 100% in your browser!
    """)
    
    # Chat interface
    chatbot = gr.ChatInterface(
        math_tutor,
        type="messages",
        examples=[
            "What is a derivative?",
            "Solve x² + 3x - 4 = 0",
            "Differentiate x³ + 2x",
            "Explain Bayes' theorem",
            "Simplify (x² + 2x + 1) / (x + 1)",
            "Integrate sin(x)",
            "What's the difference between permutations and combinations?",
        ],
        title="Chat with SomAi",
    )
    
    gr.Markdown("""
    ---
    
    **What I can do:**
    - Explain math topics (calculus, algebra, probability, linear algebra)
    - Solve equations: "solve x² + 3x = 0"
    - Simplify expressions: "simplify (2x² + 4x) / 2x"
    - Find derivatives: "differentiate sin(x) * x²"
    - Integrate functions: "integrate x³"
    
    **Note:** Responses render with LaTeX for beautiful math formatting. Try hovering over formulas!
    """)


if __name__ == "__main__":
    demo.launch(show_api=False)
