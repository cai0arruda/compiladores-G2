"""UpsideLang-to-Python translator using PLY (Lex/Yacc).

This script translates a custom Stranger Things inspired language (.up files)
into equivalent Python source code. It uses the PLY library to perform
lexical analysis and parsing, producing a small AST that is then rendered
back to Python.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional

import ply.lex as lex
import ply.yacc as yacc

# ---------------------------------------------------------------------------
# Abstract Syntax Tree definitions
# ---------------------------------------------------------------------------


@dataclass
class Node:
    def to_python(self, indent: int = 0) -> str:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class Program(Node):
    statements: List[Node]

    def to_python(self, indent: int = 0) -> str:
        return "\n".join(stmt.to_python(indent) for stmt in self.statements)


@dataclass
class FunctionDef(Node):
    name: str
    params: List[str]
    body: List[Node]

    def to_python(self, indent: int = 0) -> str:
        indent_str = "    " * indent
        params = ", ".join(self.params)
        header = f"{indent_str}def {self.name}({params}):"
        if not self.body:
            return f"{header}\n{indent_str}    pass"
        body_lines = [stmt.to_python(indent + 1) for stmt in self.body]
        return "\n".join([header] + body_lines)


@dataclass
class While(Node):
    test: Node
    body: List[Node]

    def to_python(self, indent: int = 0) -> str:
        indent_str = "    " * indent
        header = f"{indent_str}while {self.test.to_python()}:"
        body_lines = [stmt.to_python(indent + 1) for stmt in self.body]
        return "\n".join([header] + body_lines)


@dataclass
class If(Node):
    test: Node
    body: List[Node]
    orelse: Optional[List[Node]] = None

    def to_python(self, indent: int = 0) -> str:
        indent_str = "    " * indent
        header = f"{indent_str}if {self.test.to_python()}:"
        body_lines = [stmt.to_python(indent + 1) for stmt in self.body]
        lines = [header] + body_lines
        if self.orelse is not None:
            else_lines = [stmt.to_python(indent + 1) for stmt in self.orelse]
            lines.append(f"{indent_str}else:")
            lines.extend(else_lines)
        return "\n".join(lines)


@dataclass
class Return(Node):
    value: Node

    def to_python(self, indent: int = 0) -> str:
        indent_str = "    " * indent
        return f"{indent_str}return {self.value.to_python()}"


@dataclass
class Assign(Node):
    target: str
    value: Node

    def to_python(self, indent: int = 0) -> str:
        indent_str = "    " * indent
        return f"{indent_str}{self.target} = {self.value.to_python()}"


@dataclass
class ExprStmt(Node):
    value: Node

    def to_python(self, indent: int = 0) -> str:
        indent_str = "    " * indent
        return f"{indent_str}{self.value.to_python()}"


@dataclass
class BinaryOp(Node):
    op: str
    left: Node
    right: Node

    def to_python(self, indent: int = 0) -> str:
        return f"{self.left.to_python()} {self.op} {self.right.to_python()}"


@dataclass
class Compare(Node):
    op: str
    left: Node
    right: Node

    def to_python(self, indent: int = 0) -> str:
        return f"{self.left.to_python()} {self.op} {self.right.to_python()}"


@dataclass
class Call(Node):
    func: str
    args: List[Node]

    def to_python(self, indent: int = 0) -> str:
        mapped = {
            "walkieTalkie": "print",
            "walkieListen": "input",
        }.get(self.func, self.func)
        args = ", ".join(arg.to_python() for arg in self.args)
        return f"{mapped}({args})"


@dataclass
class Identifier(Node):
    name: str

    def to_python(self, indent: int = 0) -> str:
        return self.name


@dataclass
class Number(Node):
    value: int

    def to_python(self, indent: int = 0) -> str:
        return str(self.value)


@dataclass
class String(Node):
    raw: str

    def to_python(self, indent: int = 0) -> str:
        return self.raw


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

reserved = {
    "openGate": "OPEN_GATE",
    "closeGate": "CLOSE_GATE",
    "labsExperiments": "LABS_EXPERIMENTS",
    "vecnaAwake": "VECNA_AWAKE",
    "elevenHelps": "ELEVEN_HELPS",
    "walkieTalkie": "WALKIE_TALKIE",
    "walkieListen": "WALKIE_LISTEN",
}

tokens = [
    "IDENTIFIER",
    "NUMBER",
    "STRING",
    "EQUAL",
    "PLUS",
    "MINUS",
    "TIMES",
    "DIVIDE",
    "LPAREN",
    "RPAREN",
    "COLON",
    "COMMA",
    "LT",
    "GT",
    "LE",
    "GE",
    "EQEQ",
    "NEWLINE",
    "INDENT",
    "DEDENT",
] + list(reserved.values())

# Tokens for simple punctuation/operators

t_ignore = " \t"


def t_COMMENT(t):
    r"\#.*"
    pass


t_PLUS = r"\+"
t_MINUS = r"-"
t_TIMES = r"\*"
t_DIVIDE = r"/"
t_LPAREN = r"\("
t_RPAREN = r"\)"
t_COLON = r":"
t_COMMA = r"," 

t_LE = r"<="
t_GE = r">="
t_EQEQ = r"=="
t_EQUAL = r"="
t_LT = r"<"
t_GT = r">"

def t_NUMBER(t):
    r"\d+"
    t.value = Number(int(t.value))
    return t


def t_STRING(t):
    r"f?\"([^\\\n]|\\.)*?\""
    t.value = String(t.value)
    return t


def t_IDENTIFIER(t):
    r"[A-Za-z_][A-Za-z0-9_]*"
    t.type = reserved.get(t.value, "IDENTIFIER")
    if t.type == "IDENTIFIER":
        t.value = Identifier(t.value)
    return t


def t_NEWLINE(t):
    r"\n+"
    t.lexer.lineno += len(t.value)
    return t


def t_error(t):
    raise SyntaxError(f"Illegal character '{t.value[0]}' at line {t.lexer.lineno}")


base_lexer = lex.lex()


def generate_tokens(data: str):
    """Yield tokens while inserting INDENT/DEDENT markers based on leading spaces."""

    indent_stack = [0]
    lineno = 1
    for raw_line in data.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            lineno += 1
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent > indent_stack[-1]:
            tok = lex.LexToken()
            tok.type = "INDENT"
            tok.value = None
            tok.lineno = lineno
            tok.lexpos = 0
            indent_stack.append(indent)
            yield tok
        while indent < indent_stack[-1]:
            indent_stack.pop()
            tok = lex.LexToken()
            tok.type = "DEDENT"
            tok.value = None
            tok.lineno = lineno
            tok.lexpos = 0
            yield tok

        base_lexer.input(raw_line.lstrip())
        base_lexer.lineno = lineno
        while True:
            tok = base_lexer.token()
            if not tok:
                break
            yield tok
        # Statement separator
        tok = lex.LexToken()
        tok.type = "NEWLINE"
        tok.value = None
        tok.lineno = lineno
        tok.lexpos = 0
        yield tok
        lineno += 1

    while len(indent_stack) > 1:
        indent_stack.pop()
        tok = lex.LexToken()
        tok.type = "DEDENT"
        tok.value = None
        tok.lineno = lineno
        tok.lexpos = 0
        yield tok


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

precedence = (
    ("left", "LT", "LE", "GT", "GE", "EQEQ"),
    ("left", "PLUS", "MINUS"),
    ("left", "TIMES", "DIVIDE"),
)


class Parser:
    def __init__(self):
        self.tokens = tokens
        self.parser = yacc.yacc(module=self)

    def p_program(self, p):
        "program : statements"
        p[0] = Program(p[1])

    def p_statements(self, p):
        """
        statements : statement
                   | statements statement
        """
        if len(p) == 2:
            p[0] = [p[1]] if p[1] is not None else []
        else:
            p[0] = p[1]
            if p[2] is not None:
                p[0].append(p[2])

    def p_statement(self, p):
        """
        statement : func_def
                  | while_stmt
                  | if_stmt
                  | simple_stmt NEWLINE
                  | NEWLINE
        """
        if len(p) == 3:
            p[0] = p[1]
        elif p.slice[1].type == "NEWLINE":
            p[0] = None
        else:
            p[0] = p[1]

    def p_func_def(self, p):
        "func_def : OPEN_GATE IDENTIFIER LPAREN parameters RPAREN COLON NEWLINE INDENT statements DEDENT"
        p[0] = FunctionDef(p[2].name, p[4], p[9])

    def p_parameters(self, p):
        """
        parameters : parameter_list
                   | empty
        """
        p[0] = p[1] if p[1] is not None else []

    def p_parameter_list(self, p):
        """
        parameter_list : IDENTIFIER
                       | parameter_list COMMA IDENTIFIER
        """
        if len(p) == 2:
            p[0] = [p[1].name]
        else:
            p[0] = p[1] + [p[3].name]

    def p_while(self, p):
        "while_stmt : LABS_EXPERIMENTS expression COLON NEWLINE INDENT statements DEDENT"
        p[0] = While(p[2], p[6])

    def p_if(self, p):
        """
        if_stmt : VECNA_AWAKE expression COLON NEWLINE INDENT statements DEDENT
                | VECNA_AWAKE expression COLON NEWLINE INDENT statements DEDENT ELEVEN_HELPS COLON NEWLINE INDENT statements DEDENT
        """
        if len(p) == 8:
            p[0] = If(p[2], p[6])
        else:
            p[0] = If(p[2], p[6], p[12])

    def p_simple_stmt(self, p):
        """
        simple_stmt : return_stmt
                    | assignment
                    | expr_stmt
        """
        p[0] = p[1]

    def p_return(self, p):
        "return_stmt : CLOSE_GATE expression"
        p[0] = Return(p[2])

    def p_assignment(self, p):
        "assignment : IDENTIFIER EQUAL expression"
        p[0] = Assign(p[1].name, p[3])

    def p_expr_stmt(self, p):
        "expr_stmt : expression"
        p[0] = ExprStmt(p[1])

    def p_expression_binop(self, p):
        """
        expression : expression PLUS expression
                   | expression MINUS expression
                   | expression TIMES expression
                   | expression DIVIDE expression
        """
        p[0] = BinaryOp(p[2], p[1], p[3])

    def p_expression_compare(self, p):
        """
        expression : expression LT expression
                   | expression LE expression
                   | expression GT expression
                   | expression GE expression
                   | expression EQEQ expression
        """
        p[0] = Compare(p[2], p[1], p[3])

    def p_expression_group(self, p):
        "expression : LPAREN expression RPAREN"
        p[0] = p[2]

    def p_expression_number(self, p):
        "expression : NUMBER"
        p[0] = p[1]

    def p_expression_string(self, p):
        "expression : STRING"
        p[0] = p[1]

    def p_expression_identifier(self, p):
        "expression : IDENTIFIER"
        p[0] = p[1]

    def p_expression_call(self, p):
        """
        expression : WALKIE_TALKIE LPAREN arguments RPAREN
                   | WALKIE_LISTEN LPAREN arguments RPAREN
                   | IDENTIFIER LPAREN arguments RPAREN
        """
        func_name = p[1] if isinstance(p[1], str) else p[1].name
        p[0] = Call(func_name, p[3])

    def p_arguments(self, p):
        """
        arguments : argument_list
                  | empty
        """
        p[0] = p[1] if p[1] is not None else []

    def p_argument_list(self, p):
        """
        argument_list : expression
                      | argument_list COMMA expression
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_empty(self, p):
        "empty :"
        p[0] = None

    def p_error(self, p):
        if p is None:
            raise SyntaxError("Unexpected end of input")
        raise SyntaxError(f"Unexpected token {p.type} at line {p.lineno}")

    def parse(self, data: str) -> Program:
        token_stream = list(generate_tokens(data))
        # PLY expects a function to supply tokens; wrap iterator
        def token_func():
            return token_stream.pop(0) if token_stream else None

        result = self.parser.parse(lexer=None, tokenfunc=token_func)
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def translate_source(source: str) -> str:
    parser = Parser()
    ast = parser.parse(source)
    return ast.to_python()


def main(argv: List[str]) -> None:
    if len(argv) != 3:
        raise SystemExit("Usage: python translator.py <input.up> <output.py>")

    input_path, output_path = argv[1], argv[2]
    with open(input_path, "r", encoding="utf-8") as f:
        source = f.read()

    translated = translate_source(source)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated + "\n")


if __name__ == "__main__":
    main(sys.argv)
