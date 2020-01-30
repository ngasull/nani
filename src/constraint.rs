use crate::ascii::{Ascii, AsciiChar};
use crate::ast;
use std::collections::HashMap;

pub struct Constraint<'a>(&'a dyn ast::Positioned, Term<'a>);

enum Term<'a> {
    Constant(&'a ast::ConstantIdentifier<'a>),
    Expression(&'a ast::Expression<'a>),
    Function(&'a Vec<ast::Expression<'a>>, &'a ast::Expression<'a>),
    FunctionStatement(&'a Vec<ast::Expression<'a>>),
    Numeric,
    Variable(&'a ast::VariableIdentifier<'a>),
    Vartype(ast::Vartype),
}

type Substitution<'a> = HashMap<&'a dyn ast::Positioned, Term<'a>>;

fn program_constraints(program_ast: ast::Program) {
    let mut constraints = Vec::new();

    for constant in &program_ast.constants {
        // Identifier constraint
        constraints.push(Constraint(
            &constant.expression,
            Term::Constant(&constant.identifier),
        ));
        // Recursive descent in value's expression
        constraints.append(&mut expression_constraints(&constant.expression));
    }

    for props in &program_ast.props {
        // Identifier constraint
        constraints.push(Constraint(
            &props.initial_value,
            Term::Variable(&props.identifier),
        ));
        // Recursive descent in value's expression
        constraints.append(&mut expression_constraints(&props.initial_value));
    }

    for function in &program_ast.functions {
        // Function interface constraint
        constraints.push(Constraint(
            &function.identifier,
            Term::Vartype(ast::Vartype::Function(
                function
                    .args
                    .iter()
                    .map(|a| a.annotation.vartype.clone())
                    .collect(),
                Box::new(function.return_type.as_ref().map(|a| a.vartype.clone())),
            )),
        ));
        // Argument type constraints
        constraints.append(
            &mut function
                .args
                .iter()
                .map(|a| Constraint(a, Term::Vartype(a.annotation.vartype.clone())))
                .collect(),
        );
        // Function body constraints
        constraints.append(
            &mut function
                .body
                .iter()
                .flat_map(statement_constraints)
                .collect(),
        );
    }
}

fn statement_constraints<'a>(statement: &'a ast::Statement<'a>) -> Vec<Constraint<'a>> {
    match statement {
        ast::Statement::VariableAssignment {
            identifier,
            expression,
            ..
        } => {
            let mut constraints = expression_constraints(expression);
            constraints.push(Constraint(identifier, Term::Expression(expression)));
            constraints
        }
        ast::Statement::FunctionCall {
            identifier, args, ..
        } => {
            let mut constraints: Vec<Constraint<'a>> =
                args.iter().flat_map(expression_constraints).collect();
            constraints.push(Constraint(identifier, Term::FunctionStatement(&args)));
            constraints
        }
        ast::Statement::If {
            condition, body, ..
        }
        | ast::Statement::While {
            condition, body, ..
        } => {
            let mut constraints: Vec<Constraint<'a>> =
                body.iter().flat_map(statement_constraints).collect();
            constraints.append(&mut expression_constraints(condition));
            constraints.push(Constraint(condition, Term::Vartype(ast::Vartype::Bool)));
            constraints
        }
    }
}

fn expression_constraints<'a>(expression: &'a ast::Expression<'a>) -> Vec<Constraint<'a>> {
    match expression {
        ast::Expression::FunctionCall { identifier, args } => {
            let mut constraints: Vec<Constraint> =
                args.iter().flat_map(expression_constraints).collect();
            constraints.push(Constraint(identifier, Term::Function(&args, expression)));
            constraints
        }
        ast::Expression::BinaryOperation {
            operator,
            left,
            right,
            ..
        } => {
            let mut constraints = expression_constraints(left);
            constraints.append(&mut expression_constraints(right));

            match operator {
                ast::BinaryOperator::And | ast::BinaryOperator::Or => {
                    constraints.push(Constraint(expression, Term::Vartype(ast::Vartype::Bool)));
                    constraints.push(Constraint(left.as_ref(), Term::Vartype(ast::Vartype::Bool)));
                    constraints.push(Constraint(
                        right.as_ref(),
                        Term::Vartype(ast::Vartype::Bool),
                    ));
                }
                ast::BinaryOperator::Eq | ast::BinaryOperator::Ne => {
                    constraints.push(Constraint(expression, Term::Vartype(ast::Vartype::Bool)));
                    constraints.push(Constraint(left.as_ref(), Term::Numeric));
                    constraints.push(Constraint(right.as_ref(), Term::Numeric));
                }
                ast::BinaryOperator::Ge
                | ast::BinaryOperator::Gt
                | ast::BinaryOperator::Le
                | ast::BinaryOperator::Lt => {
                    constraints.push(Constraint(expression, Term::Vartype(ast::Vartype::Bool)));
                    constraints.push(Constraint(left.as_ref(), Term::Numeric));
                    constraints.push(Constraint(right.as_ref(), Term::Numeric));
                }
                ast::BinaryOperator::Plus
                | ast::BinaryOperator::Minus
                | ast::BinaryOperator::Multiply
                | ast::BinaryOperator::Divide => {
                    constraints.push(Constraint(expression, Term::Numeric));
                    constraints.push(Constraint(expression, Term::Expression(left.as_ref())));
                    constraints.push(Constraint(left.as_ref(), Term::Numeric));
                    constraints.push(Constraint(right.as_ref(), Term::Numeric));
                }
                ast::BinaryOperator::Modulo
                | ast::BinaryOperator::BitShiftLeft
                | ast::BinaryOperator::BitShiftRight => {
                    constraints.push(Constraint(expression, Term::Numeric));
                    constraints.push(Constraint(expression, Term::Expression(left.as_ref())));
                    constraints.push(Constraint(left.as_ref(), Term::Numeric));
                    constraints.push(Constraint(
                        right.as_ref(),
                        Term::Vartype(ast::Vartype::Integer),
                    ));
                }
                ast::BinaryOperator::BitAnd | ast::BinaryOperator::BitOr => {
                    constraints.push(Constraint(expression, Term::Vartype(ast::Vartype::Integer)));
                    constraints.push(Constraint(
                        left.as_ref(),
                        Term::Vartype(ast::Vartype::Integer),
                    ));
                    constraints.push(Constraint(
                        right.as_ref(),
                        Term::Vartype(ast::Vartype::Integer),
                    ));
                }
            };
            constraints
        }
        ast::Expression::Constant(identifier) => {
            vec![Constraint(expression, Term::Constant(identifier))]
        }
        ast::Expression::Variable(identifier) => {
            vec![Constraint(expression, Term::Variable(identifier))]
        }
        ast::Expression::Literal { value, .. } => {
            vec![Constraint(expression, Term::Vartype(value.into()))]
        }
    }
}
