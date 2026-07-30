import ast
import math
import operator
from typing import Any


class CalcError(ValueError):
    pass


MAX_EXPRESSION_LENGTH = 500
MAX_EXPONENT = 100
MAX_RESULT_BITS = 10_000
MAX_DECIMAL_PLACES = 5


BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "degrees": math.degrees,
    "radians": math.radians,
    "floor": math.floor,
    "ceil": math.ceil,
}


CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _evaluate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalcError("Only numbers are allowed")

        return node.value

    if isinstance(node, ast.Name):
        if node.id not in CONSTANTS:
            raise CalcError(f"Unknown constant: {node.id}")

        return CONSTANTS[node.id]

    if isinstance(node, ast.UnaryOp):
        operation = UNARY_OPERATORS.get(type(node.op))

        if operation is None:
            raise CalcError("Unary operator is not allowed")

        return operation(_evaluate_node(node.operand))

    if isinstance(node, ast.BinOp):
        operation = BINARY_OPERATORS.get(type(node.op))

        if operation is None:
            raise CalcError("Operator is not allowed")

        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalcError(
                f"Exponent cannot exceed {MAX_EXPONENT}"
            )

        try:
            result = operation(left, right)
        except ZeroDivisionError:
            raise CalcError("Division by zero") from None
        except OverflowError:
            raise CalcError("Result is too large") from None

        if isinstance(result, complex):
            raise CalcError("Complex numbers are not supported")

        # The exponent cap applies to a single operation, so nesting escapes it:
        # (((10**100)**100)**100)**100 is 28 characters and each step is legal.
        # Bounding the result instead also covers repeated multiplication.
        if isinstance(result, int) and result.bit_length() > MAX_RESULT_BITS:
            raise CalcError("Result is too large")

        return result

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalcError("Only direct function calls are allowed")

        function = FUNCTIONS.get(node.func.id)

        if function is None:
            raise CalcError(f"Unknown function: {node.func.id}")

        if node.keywords:
            raise CalcError("Named arguments are not supported")

        arguments = [_evaluate_node(argument) for argument in node.args]

        try:
            result = function(*arguments)
        except (TypeError, ValueError, OverflowError) as error:
            raise CalcError(f"{node.func.id}: {error}") from None

        if isinstance(result, bool) or not isinstance(result, (int, float)):
            raise CalcError("Calculation did not return a number")

        return result

    raise CalcError(
        f"Unsupported expression: {type(node).__name__}"
    )


def calculate(expression: str) -> int | float:
    expression = (expression or "").strip()

    if not expression:
        raise CalcError("Expression is empty")

    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalcError("Expression is too long")

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise CalcError("Invalid mathematical expression") from None

    return _evaluate_node(parsed.body)


def format_result(value: int | float) -> int | float:
    """Trim floating point noise for display. 0.1 + 0.2 becomes 0.3."""

    if not isinstance(value, float):
        return value

    if math.isnan(value) or math.isinf(value):
        return value

    rounded = round(value, MAX_DECIMAL_PLACES)

    # Small values would round away to zero, which is wrong rather than
    # imprecise, so fall back to significant figures for those.
    if rounded == 0 and value != 0:
        rounded = float(f"{value:.{MAX_DECIMAL_PLACES}g}")

    if rounded.is_integer():
        return int(rounded)

    return rounded


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression. Use this when an accurate "
            "numerical calculation is required. Supports arithmetic, powers, "
            "percentages written as division by 100, square roots, logarithms, "
            "trigonometry, rounding, minimum and maximum values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "maxLength": MAX_EXPRESSION_LENGTH,
                    "description": (
                        "A Python-style mathematical expression, such as "
                        "'1450 * 20 / 100', 'sqrt(144)', or "
                        "'round(1250 * 1.075, 2)'."
                    ),
                }
            },
            "required": ["expression"],
        },
    },
}


async def execute_calculator_tool(args: dict[str, Any]) -> dict[str, Any]:
    expression = str(args.get("expression") or "").strip()

    try:
        result = format_result(calculate(expression))

        return {
            "ok": True,
            "expression": expression,
            "result": result,
        }

    except CalcError as error:
        return {
            "ok": False,
            "expression": expression,
            "error": str(error),
        }

    except Exception as error:
        return {
            "ok": False,
            "expression": expression,
            "error": f"Calculation failed: {type(error).__name__}",
        }