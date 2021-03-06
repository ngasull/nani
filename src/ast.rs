use crate::{
    ascii::{is_ascii, is_lower, is_upper, Ascii, AsciiChar},
    span::{position, AstSpan, Span},
};
use nom::{
    branch::alt,
    bytes::complete::{escaped, is_not, tag, take, take_while1},
    character::complete::{digit1, space0, space1},
    combinator::{cut as required, opt, recognize},
    error::{ErrorKind, ParseError},
    multi::{many0, separated_list, separated_nonempty_list},
    sequence::{delimited, preceded, tuple},
    AsBytes, InputIter, InputLength, Slice,
};
use std::convert::TryFrom;

pub trait Positioned {
    fn get_pos(&self) -> &AstSpan;
}

#[derive(Debug, PartialEq)]
pub struct Program<'a> {
    pub constants: Vec<Constant<'a>>,
    pub props: Vec<Property<'a>>,
    pub functions: Vec<FunctionDefinition<'a>>,
}

#[derive(Debug, PartialEq)]
pub struct Constant<'a> {
    pub identifier: ConstantIdentifier<'a>,
    pub expression: Expression<'a>,
}

#[derive(Debug, PartialEq)]
pub struct Property<'a> {
    pub identifier: VariableIdentifier<'a>,
    pub initial_value: Expression<'a>,
}

#[derive(Debug, PartialEq)]
pub struct FunctionDefinition<'a> {
    pub identifier: FunctionIdentifier<'a>,
    pub args: Vec<FunctionArgument<'a>>,
    pub return_type: Option<TypeAnnotation>,
    pub body: Vec<Statement<'a>>,
}

#[derive(Debug, PartialEq)]
pub struct FunctionArgument<'a> {
    pub pos: AstSpan,
    pub name: Ascii<'a>,
    pub annotation: TypeAnnotation,
}

impl<'a> Positioned for FunctionArgument<'a> {
    fn get_pos(&self) -> &AstSpan {
        &self.pos
    }
}

#[derive(Debug, PartialEq)]
pub struct ConstantIdentifier<'a> {
    pub name: Ascii<'a>,
    pub pos: AstSpan,
}

impl<'a> Positioned for ConstantIdentifier<'a> {
    fn get_pos(&self) -> &AstSpan {
        &self.pos
    }
}

#[derive(Debug, PartialEq)]
pub struct VariableIdentifier<'a> {
    pub name: Ascii<'a>,
    pub pos: AstSpan,
}

impl<'a> Positioned for VariableIdentifier<'a> {
    fn get_pos(&self) -> &AstSpan {
        &self.pos
    }
}

#[derive(Debug, PartialEq)]
pub struct FunctionIdentifier<'a> {
    pub pos: AstSpan,
    pub name: Ascii<'a>,
    pub scope: Option<FunctionScope>,
}

impl<'a> Positioned for FunctionIdentifier<'a> {
    fn get_pos(&self) -> &AstSpan {
        &self.pos
    }
}

#[derive(Debug, PartialEq)]
pub enum Statement<'a> {
    VariableAssignment {
        identifier: VariableIdentifier<'a>,
        expression: Expression<'a>,
    },
    FunctionCall {
        identifier: FunctionIdentifier<'a>,
        args: Vec<Expression<'a>>,
    },
    If {
        pos: AstSpan,
        condition: Expression<'a>,
        body: Vec<Statement<'a>>,
    },
    While {
        pos: AstSpan,
        condition: Expression<'a>,
        body: Vec<Statement<'a>>,
    },
}

#[derive(Debug, PartialEq)]
pub enum Expression<'a> {
    FunctionCall {
        identifier: FunctionIdentifier<'a>,
        args: Vec<Expression<'a>>,
    },
    BinaryOperation {
        pos: AstSpan,
        operator: BinaryOperator,
        left: Box<Expression<'a>>,
        right: Box<Expression<'a>>,
    },
    Constant(ConstantIdentifier<'a>),
    Variable(VariableIdentifier<'a>),
    Literal {
        pos: AstSpan,
        value: Value<'a>,
    },
}

impl<'a> Positioned for Expression<'a> {
    fn get_pos(&self) -> &AstSpan {
        match self {
            Expression::FunctionCall { identifier, .. } => &identifier.pos,
            Expression::Constant(identifier) => &identifier.pos,
            Expression::Variable(identifier) => &identifier.pos,
            Expression::Literal { pos, .. } => &pos,
            Expression::BinaryOperation { pos, .. } => &pos,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct FunctionScope {
    pub pos: AstSpan,
    pub token: AsciiChar,
}

#[derive(Debug, PartialEq)]
pub enum Value<'a> {
    Bool(bool),
    Integer(i32),
    Decimal(f32),
    String(Ascii<'a>),
}

#[derive(Debug, PartialEq)]
pub struct TypeAnnotation {
    pub pos: AstSpan,
    pub vartype: Vartype,
}

impl Positioned for TypeAnnotation {
    fn get_pos(&self) -> &AstSpan {
        &self.pos
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Vartype {
    Bool,
    Integer,
    Decimal,
    Function(Vec<Vartype>, Box<Option<Vartype>>),
    String,
}

impl<'a> From<&'a Value<'a>> for Vartype {
    fn from(v: &'a Value) -> Vartype {
        match v {
            Value::Bool(_) => Vartype::Bool,
            Value::Integer(_) => Vartype::Integer,
            Value::Decimal(_) => Vartype::Decimal,
            Value::String(_) => Vartype::String,
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum BinaryOperator {
    And,
    Or,
    Eq,
    Ne,
    Ge,
    Gt,
    Le,
    Lt,
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    BitShiftLeft,
    BitShiftRight,
    BitAnd,
    BitOr,
}

impl From<BinaryOperator> for &[u8] {
    fn from(operator: BinaryOperator) -> &'static [u8] {
        match operator {
            BinaryOperator::And => TOKEN_AND,
            BinaryOperator::Or => TOKEN_OR,
            BinaryOperator::Eq => TOKEN_EQ,
            BinaryOperator::Ne => TOKEN_NE,
            BinaryOperator::Ge => TOKEN_GE,
            BinaryOperator::Gt => TOKEN_GT,
            BinaryOperator::Le => TOKEN_LE,
            BinaryOperator::Lt => TOKEN_LT,
            BinaryOperator::Plus => TOKEN_PLUS,
            BinaryOperator::Minus => TOKEN_MINUS,
            BinaryOperator::Multiply => TOKEN_MULTIPLY,
            BinaryOperator::Divide => TOKEN_DIVIDE,
            BinaryOperator::Modulo => TOKEN_MODULO,
            BinaryOperator::BitShiftLeft => TOKEN_BIT_SHIFT_LEFT,
            BinaryOperator::BitShiftRight => TOKEN_BIT_SHIFT_RIGHT,
            BinaryOperator::BitAnd => TOKEN_BIT_AND,
            BinaryOperator::BitOr => TOKEN_BIT_OR,
        }
    }
}

const TOKEN_AND: &[u8] = b"&&";
const TOKEN_OR: &[u8] = b"||";
const TOKEN_EQ: &[u8] = b"==";
const TOKEN_NE: &[u8] = b"!=";
const TOKEN_GE: &[u8] = b">=";
const TOKEN_GT: &[u8] = b">";
const TOKEN_LE: &[u8] = b"<=";
const TOKEN_LT: &[u8] = b"<";
const TOKEN_PLUS: &[u8] = b"+";
const TOKEN_MINUS: &[u8] = b"-";
const TOKEN_MULTIPLY: &[u8] = b"*";
const TOKEN_DIVIDE: &[u8] = b"/";
const TOKEN_MODULO: &[u8] = b"%";
const TOKEN_BIT_SHIFT_LEFT: &[u8] = b"<<";
const TOKEN_BIT_SHIFT_RIGHT: &[u8] = b">>";
const TOKEN_BIT_AND: &[u8] = b"&";
const TOKEN_BIT_OR: &[u8] = b"|";

const BIN_OPS_WITH_BY_PRECEDENCE: &[BinaryOperator] = &[
    BinaryOperator::Or,
    BinaryOperator::And,
    BinaryOperator::Eq,
    BinaryOperator::Ne,
    BinaryOperator::Ge,
    BinaryOperator::Gt,
    BinaryOperator::Le,
    BinaryOperator::Lt,
    BinaryOperator::BitOr,
    BinaryOperator::BitAnd,
    BinaryOperator::BitShiftLeft,
    BinaryOperator::BitShiftRight,
    BinaryOperator::Plus,
    BinaryOperator::Minus,
    BinaryOperator::Divide,
    BinaryOperator::Multiply,
    BinaryOperator::Modulo,
];

const KEYWORD_IF: &[u8] = b"if";
const KEYWORD_FOR: &[u8] = b"for";
const KEYWORD_WHILE: &[u8] = b"while";
const KEYWORD_TRUE: &[u8] = b"true";
const KEYWORD_FALSE: &[u8] = b"false";
const KEYWORD_RETURN: &[u8] = b"return";
const KEYWORD_BREAK: &[u8] = b"break";
const KEYWORD_CONTINUE: &[u8] = b"continue";
const KEYWORD_PROP: &[u8] = b"prop";
const KEYWORD_VAR: &[u8] = b"var";

const RESERVED_KEYWORDS: &[&[u8]] = &[
    KEYWORD_IF,
    KEYWORD_FOR,
    KEYWORD_WHILE,
    KEYWORD_TRUE,
    KEYWORD_FALSE,
    KEYWORD_RETURN,
    KEYWORD_BREAK,
    KEYWORD_CONTINUE,
    KEYWORD_PROP,
    KEYWORD_VAR,
];

fn constant_assignment(s: Span) -> IResult<Constant> {
    let (s, pos) = position(s)?;
    let (s, name) = not_reserved(snakecase_upper)(s)?;
    let (s, _) = required(delimited(space0, byte(b'='), space0))(s)?;
    let (s, expr) = required(expression)(s)?;
    let (s, end_pos) = position(s)?;
    return Ok((
        s,
        Constant {
            identifier: ConstantIdentifier {
                pos: pos.to(end_pos),
                name: Ascii::from(name),
            },
            expression: expr,
        },
    ));
}

fn property_assignment(s: Span) -> IResult<Property> {
    let (s, pos) = position(s)?;
    let (s, _) = tag(KEYWORD_PROP)(s)?;
    let (s, _) = space1(s)?;
    let (s, name) = required(not_reserved(snakecase_lower))(s)?;
    let (s, _) = required(delimited(space0, byte(b'='), space0))(s)?;
    let (s, initial_value) = required(expression)(s)?;
    let (s, end_pos) = position(s)?;
    return Ok((
        s,
        Property {
            identifier: VariableIdentifier {
                pos: pos.to(end_pos),
                name: Ascii::from(name),
            },
            initial_value,
        },
    ));
}

fn function_definition(s: Span) -> IResult<FunctionDefinition> {
    let (s, pos) = position(s)?;
    let (s, name) = not_reserved(snakecase_lower)(s)?;
    let (s, scope) = opt(function_scope)(s)?;
    let (s, args) = trim(delimited(
        byte(b'('),
        required(separated_list(trim(byte(b',')), function_argument)),
        byte(b')'),
    ))(s)?;
    let (s, body) = delimited(
        required(byte(b'{')),
        many0(on_a_line(statement)),
        finishes_multiline(required(byte(b'}'))),
    )(s)?;
    let (s, end_pos) = position(s)?;
    Ok((
        s,
        FunctionDefinition {
            identifier: FunctionIdentifier {
                pos: pos.to(end_pos),
                name: Ascii::from(name),
                scope,
            },
            args,
            return_type: None,
            body,
        },
    ))
}

fn function_argument(s: Span) -> IResult<FunctionArgument> {
    let (s, pos) = position(s)?;
    let (s, name) = not_reserved(snakecase_lower)(s)?;
    let (s, annotation) = required(type_annotation)(s)?;
    let (s, end_pos) = position(s)?;
    Ok((
        s,
        FunctionArgument {
            pos: pos.to(end_pos),
            name: Ascii::from(name),
            annotation,
        },
    ))
}

fn type_annotation(s: Span) -> IResult<TypeAnnotation> {
    let (s, pos) = position(s)?;
    let (s, _) = delimited(space0, byte(b':'), space0)(s)?;
    let (s, vartype) = required(vartype)(s)?;
    let (s, end_pos) = position(s)?;
    Ok((
        s,
        TypeAnnotation {
            pos: pos.to(end_pos),
            vartype,
        },
    ))
}

fn vartype(s: Span) -> IResult<Vartype> {
    alt((
        vartype_bool,
        vartype_integer,
        vartype_decimal,
        vartype_string,
    ))(s)
}

#[inline]
fn vartype_bool(s: Span) -> IResult<Vartype> {
    let (s, _) = tag(&b"bool"[..])(s)?;
    Ok((s, Vartype::Bool))
}

#[inline]
fn vartype_integer(s: Span) -> IResult<Vartype> {
    let (s, _) = tag(&b"integer"[..])(s)?;
    Ok((s, Vartype::Integer))
}

#[inline]
fn vartype_decimal(s: Span) -> IResult<Vartype> {
    let (s, _) = tag(&b"decimal"[..])(s)?;
    Ok((s, Vartype::Decimal))
}

#[inline]
fn vartype_string(s: Span) -> IResult<Vartype> {
    let (s, _) = tag(&b"string"[..])(s)?;
    Ok((s, Vartype::String))
}

#[inline]
fn statement(s: Span) -> IResult<Statement> {
    alt((
        variable_assignment,
        if_statement,
        while_statement,
        function_call_statement,
    ))(s)
}

fn function_call_statement(s: Span) -> IResult<Statement> {
    let (s, fncall) = function_call(s)?;
    if let Expression::FunctionCall { identifier, args } = fncall {
        Ok((s, Statement::FunctionCall { identifier, args }))
    } else {
        panic!()
    }
}

fn variable_assignment(s: Span) -> IResult<Statement> {
    let (s, pos) = position(s)?;
    let (s, name) = not_reserved(snakecase_lower)(s)?;
    let (s, _) = delimited(space0, byte(b'='), space0)(s)?;
    let (s, expr) = required(expression)(s)?;
    let (s, end_pos) = position(s)?;
    return Ok((
        s,
        Statement::VariableAssignment {
            identifier: VariableIdentifier {
                pos: pos.to(end_pos),
                name: Ascii::from(name),
            },
            expression: expr,
        },
    ));
}

fn if_statement(s: Span) -> IResult<Statement> {
    let (s, pos) = position(s)?;
    let (s, _) = tag(KEYWORD_IF)(s)?;
    let (s, expr) = delimited(space1, required(expression), space0)(s)?;
    let (s, body) = delimited(
        required(byte(b'{')),
        many0(on_a_line(statement)),
        finishes_multiline(required(byte(b'}'))),
    )(s)?;
    let (s, end_pos) = position(s)?;
    return Ok((
        s,
        Statement::If {
            pos: pos.to(end_pos),
            condition: expr,
            body,
        },
    ));
}

fn while_statement(s: Span) -> IResult<Statement> {
    let (s, pos) = position(s)?;
    let (s, _) = tag(KEYWORD_WHILE)(s)?;
    let (s, expr) = delimited(space1, required(expression), space0)(s)?;
    let (s, body) = delimited(
        required(byte(b'{')),
        many0(on_a_line(statement)),
        finishes_multiline(required(byte(b'}'))),
    )(s)?;
    let (s, end_pos) = position(s)?;
    return Ok((
        s,
        Statement::While {
            pos: pos.to(end_pos),
            condition: expr,
            body,
        },
    ));
}

fn expression(s: Span) -> IResult<Expression> {
    let (s, expr) = expression_raw(s)?;
    let (s, ops) = many0(binary_operation_right)(s)?;
    Ok((s, binary_operations_aggregate(expr, ops)))
}

fn binary_operations_aggregate<'a>(
    left: Expression<'a>,
    mut ops: Vec<(BinaryOperator, Expression<'a>)>,
) -> Expression<'a> {
    if ops.len() == 0 {
        left
    } else if ops.len() == 1 {
        let (operator, right) = ops.pop().unwrap();
        Expression::BinaryOperation {
            pos: left.get_pos().to(&right.get_pos()),
            operator,
            left: Box::new(left),
            right: Box::new(right),
        }
    } else {
        let i = BIN_OPS_WITH_BY_PRECEDENCE
            .iter()
            .find_map(|&candidate_operator| {
                ops.iter().enumerate().find_map(|(i, &(operator, _))| {
                    if operator == candidate_operator {
                        Some(i)
                    } else {
                        None
                    }
                })
            })
            .expect("An operator has been parsed but then not recognized");

        let right_ops = ops.split_off(i + 1);
        let (operator, right) = ops.pop().unwrap();
        let left = binary_operations_aggregate(left, ops);
        let right = binary_operations_aggregate(right, right_ops);
        Expression::BinaryOperation {
            pos: left.get_pos().to(&right.get_pos()),
            operator,
            left: Box::new(left),
            right: Box::new(right),
        }
    }
}

#[inline]
fn expression_raw(s: Span) -> IResult<Expression> {
    // Order matters here, for instance:
    // literal > function_call > variable
    // Also matters for performance: some matchers are more expensive than others
    alt((parens, constant, literal, function_call, variable))(s)
}

#[inline]
fn binary_operation_right(s: Span) -> IResult<(BinaryOperator, Expression)> {
    tuple((
        delimited(space0, binary_operator, space0),
        required(expression_raw),
    ))(s)
}

fn binary_operator(s: Span) -> IResult<BinaryOperator> {
    BIN_OPS_WITH_BY_PRECEDENCE
        .iter()
        .find_map(|&op| {
            tag::<&[u8], Span, (Span, ErrorKind)>(op.into())(s)
                .ok()
                .map(|(s, _)| Ok((s, op)))
        })
        .unwrap_or_else(|| Err(nom::Err::Error(AstError::from(s, None))))
}

fn function_call(s: Span) -> IResult<Expression> {
    let (s, pos) = position(s)?;
    let (s, name) = snakecase_lower(s)?;
    let (s, scope) = opt(function_scope)(s)?;
    let (s, args) = delimited(
        byte(b'('),
        required(separated_list(trim(byte(b',')), expression)),
        required(byte(b')')),
    )(s)?;
    let (s, end_pos) = position(s)?;
    Ok((
        s,
        Expression::FunctionCall {
            identifier: FunctionIdentifier {
                pos: pos.to(end_pos),
                name: Ascii::from(name),
                scope,
            },
            args,
        },
    ))
}

fn function_scope(s: Span) -> IResult<FunctionScope> {
    let (s, pos) = position(s)?;
    match alt((byte(b'!'), byte(b'?')))(s) {
        Ok((s, token)) => {
            let (s, end_pos) = position(s)?;
            Ok((
                s,
                FunctionScope {
                    pos: pos.to(end_pos),
                    token: AsciiChar::try_from(token).unwrap(),
                },
            ))
        }
        // If not a scope, and if we are still calling/defining a function,
        // the scope is invalid.
        Err(e) => match tuple((take(1usize), byte(b'(')))(s) {
            Ok((_, (c, _))) if c.fragment[0] != b'(' => Err(failure_from(
                s,
                match AsciiChar::try_from(c.fragment[0]) {
                    Ok(c) => AstErrorKind::InvalidScope(c),
                    _ => AstErrorKind::NonAscii(c.fragment[0]),
                },
            )),
            _ => Err(e),
        },
    }
}

fn parens(s: Span) -> IResult<Expression> {
    delimited(byte(b'('), expression, byte(b')'))(s)
}

fn literal(s: Span) -> IResult<Expression> {
    let (s, pos) = position(s)?;
    let (s, value) = value(s)?;
    let (s, end_pos) = position(s)?;
    Ok((
        s,
        Expression::Literal {
            pos: pos.to(end_pos),
            value,
        },
    ))
}

fn constant(s: Span) -> IResult<Expression> {
    let (s, pos) = position(s)?;
    let (s, name) = not_reserved(snakecase_upper)(s)?;
    let (s, end_pos) = position(s)?;
    Ok((
        s,
        Expression::Constant(ConstantIdentifier {
            pos: pos.to(end_pos),
            name: Ascii::from(name),
        }),
    ))
}

fn variable(s: Span) -> IResult<Expression> {
    let (s, pos) = position(s)?;
    let (s, name) = not_reserved(snakecase_lower)(s)?;
    let (s, end_pos) = position(s)?;
    Ok((
        s,
        Expression::Variable(VariableIdentifier {
            pos: pos.to(end_pos),
            name: Ascii::from(name),
        }),
    ))
}

#[inline]
fn value(s: Span) -> IResult<Value> {
    alt((value_bool, value_decimal, value_integer, value_string))(s)
}

fn value_bool(s: Span) -> IResult<Value> {
    let (s, value) = if let (s, Some(_)) = opt(tag(KEYWORD_TRUE))(s)? {
        (s, true)
    } else {
        let (s, _) = tag(KEYWORD_FALSE)(s)?;
        (s, false)
    };
    Ok((s, Value::Bool(value)))
}

fn value_integer(s: Span) -> IResult<Value> {
    let (s, i) = digit1(s)?;
    let i = Ascii::from(i.fragment)
        .to_string()
        .parse()
        .map_err(|_| nom::Err::Failure(AstError::from(s, None)))?;
    Ok((s, Value::Integer(i)))
}

fn value_decimal(s: Span) -> IResult<Value> {
    let (s, n) = digit1(s)?;
    let (s, _) = byte(b'.')(s)?;
    let (s, after_decimal) = position(s)?;
    let (s, d) = digit1::<Span, AstError>(s)
        .map_err(|_| failure_from(after_decimal, AstErrorKind::Decimal))?;
    let i = format!("{}.{}", Ascii::from(n.fragment), Ascii::from(d.fragment))
        .parse()
        .map_err(|_| failure_from(s, AstErrorKind::Decimal))?;
    Ok((s, Value::Decimal(i)))
}

fn value_string(s: Span) -> IResult<Value> {
    let (s, str_literal) = delimited(
        byte(b'"'),
        required(escaped(
            is_not("\"\\"),
            '\\',
            alt((byte(b'"'), byte(b'\\'))),
        )),
        byte(b'"'),
    )(s)?;
    if let Some(&c) = str_literal.fragment.iter().find(|&&c| !is_ascii(c)) {
        Err(failure_from(s, AstErrorKind::NonAscii(c)))
    } else {
        Ok((s, Value::String(Ascii::from(str_literal.fragment))))
    }
}

fn snakecase_upper(s: Span) -> IResult<&[u8]> {
    let (s, n) = recognize(separated_nonempty_list(byte(b'_'), take_while1(is_upper)))(s)?;
    Ok((s, n.fragment))
}

fn snakecase_lower(s: Span) -> IResult<&[u8]> {
    let (s, n) = recognize(separated_nonempty_list(byte(b'_'), take_while1(is_lower)))(s)?;
    Ok((s, n.fragment))
}

fn byte(b: u8) -> impl Fn(Span) -> IResult<u8> {
    move |input: Span| match (input).iter_elements().next().map(|t: u8| (b, t)) {
        Some((b, t)) => {
            if b == t {
                Ok((input.slice(1..), b))
            } else if let Ok(got) = AsciiChar::try_from(t) {
                Err(error_from(
                    input,
                    AstErrorKind::ExpectedCharInstead(AsciiChar::try_from(b).unwrap(), got),
                ))
            } else {
                Err(error_from(input, AstErrorKind::NonAscii(t)))
            }
        }
        None => Err(error_from(
            input,
            AstErrorKind::ExpectedChar(AsciiChar::try_from(b).unwrap()),
        )),
    }
}

fn trim<'a, O, F>(f: F) -> impl Fn(Span<'a>) -> IResult<O>
where
    F: Fn(Span<'a>) -> IResult<O>,
{
    move |input: Span| {
        let (input, _) = many0(space)(input)?;
        let (input, o) = f(input)?;
        let (input, _) = many0(space)(input)?;
        Ok((input, o))
    }
}

fn on_a_line<'a, O, F>(f: F) -> impl Fn(Span<'a>) -> IResult<O>
where
    F: Fn(Span<'a>) -> IResult<O>,
{
    move |input: Span| {
        let (input, _) = many0(alt((space, newline)))(input)?;
        let (input, o) = f(input)?;
        let (input, _) = many0(space)(input)?;
        let (input, _) = alt((newline, eof))(input)
            .map_err(|_| failure_from(input, AstErrorKind::ExpectedNewLine))?;
        Ok((input, o))
    }
}

fn finishes_multiline<'a, O, F>(f: F) -> impl Fn(Span<'a>) -> IResult<O>
where
    F: Fn(Span<'a>) -> IResult<O>,
{
    move |input: Span| {
        let (input, _) = many0(alt((space, newline)))(input)?;
        let (input, o) = f(input)?;
        Ok((input, o))
    }
}

#[inline]
fn space<'a>(input: Span<'a>) -> IResult<u8> {
    byte(b' ')(input)
}

#[inline]
fn newline<'a>(input: Span<'a>) -> IResult<u8> {
    byte(b'\n')(input)
}

#[inline]
fn eof<'a>(input: Span<'a>) -> IResult<u8> {
    if input.input_len() == 0 {
        Ok((input, b'\0'))
    } else {
        Err(error_from(input, AstErrorKind::ExpectedEof))
    }
}

fn not_reserved<'a, O, F>(f: F) -> impl Fn(Span<'a>) -> IResult<O>
where
    F: Fn(Span<'a>) -> IResult<O>,
    O: AsBytes,
{
    move |input: Span| {
        let (new_input, res) = f(input)?;
        if let Some(&k) = RESERVED_KEYWORDS.iter().find(|&&k| k == res.as_bytes()) {
            Err(failure_from(input, AstErrorKind::ReservedKeyword(k.into())))
        } else {
            Ok((new_input, res))
        }
    }
}

#[inline]
fn error_from<'a>(s: Span<'a>, kind: AstErrorKind<'a>) -> nom::Err<AstError<'a>> {
    nom::Err::Error(AstError::from(s, Some(kind)))
}

#[inline]
fn failure_from<'a>(s: Span<'a>, kind: AstErrorKind<'a>) -> nom::Err<AstError<'a>> {
    nom::Err::Failure(AstError::from(s, Some(kind)))
}

fn nani_structure<'a>(input: &'a [u8]) -> IResult<Program<'a>> {
    let s = Span::new(input);
    let (s, constants) = many0(on_a_line(constant_assignment))(s)?;
    let (s, props) = many0(on_a_line(property_assignment))(s)?;
    let (s, functions) = many0(on_a_line(function_definition))(s)?;
    let (s, _) = many0(alt((space, newline)))(s)?;
    let (s, _) = required(eof)(s)?;
    Ok((
        s,
        Program {
            constants,
            props,
            functions,
        },
    ))
}

pub fn parse_nani<'a>(input: &'a [u8]) -> Result<Program<'a>, AstError> {
    match nani_structure(input) {
        Ok((_, program)) => Ok(program),
        Err(nom::Err::Error(err)) | Err(nom::Err::Failure(err)) => Err(err),
        _ => panic!(),
    }
}

type IResult<'a, O> = Result<(Span<'a>, O), nom::Err<AstError<'a>>>;

pub struct AstError<'a> {
    input: Span<'a>,
    explicit_kind: Option<AstErrorKind<'a>>,
    pub previous: Option<Box<AstError<'a>>>,
}

impl<'a> AstError<'a> {
    fn from(input: Span<'a>, kind: Option<AstErrorKind<'a>>) -> Self {
        Self {
            input,
            explicit_kind: kind,
            previous: None,
        }
    }

    fn kind(&self) -> AstErrorKind {
        if let Some(kind) = self.explicit_kind {
            kind
        } else if self.input.fragment.len() > 0 {
            match AsciiChar::try_from(self.input.fragment[0]) {
                Ok(token) => AstErrorKind::UnexpectedToken(token),
                Err(_) => AstErrorKind::NonAscii(self.input.fragment[0]),
            }
        } else {
            AstErrorKind::EndOfInput
        }
    }
}

impl<'a> ParseError<Span<'a>> for AstError<'a> {
    fn from_error_kind(input: Span<'a>, nom_kind: ErrorKind) -> Self {
        let explicit_kind = match nom_kind {
            _ => None,
        };
        Self {
            input,
            explicit_kind,
            previous: None,
        }
    }

    fn append(input: Span<'a>, nom_kind: ErrorKind, other: Self) -> Self {
        Self {
            previous: Some(Box::new(Self::from_error_kind(input, nom_kind))),
            ..other
        }
    }
}

impl<'a> std::fmt::Display for AstError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[L{} C{}] ", self.input.line, self.input.col)?;
        match self.kind() {
            AstErrorKind::Decimal => write!(f, "Expected number after decimal's '.'"),
            AstErrorKind::EndOfInput => write!(f, "End of input reached"),
            AstErrorKind::ExpectedChar(c) => write!(f, "Expected character {}", c),
            AstErrorKind::ExpectedCharInstead(c, instead) => {
                write!(f, "Expected character {} but got {} instead", c, instead)
            }
            AstErrorKind::ExpectedEof => write!(f, "Expected end of file"),
            AstErrorKind::ExpectedNewLine => write!(f, "Expected new line"),
            AstErrorKind::InvalidScope(c) => write!(f, "Invalid scope token {}", c),
            AstErrorKind::NonAscii(byte) => {
                write!(f, "Can't use non-ASCII character {}", char::from(byte))
            }
            AstErrorKind::ReservedKeyword(keyword) => {
                write!(f, "Can't use reserved keyword {}", keyword)
            }
            AstErrorKind::UnexpectedToken(token) => write!(f, "Unexpected token {}", token),
        }
    }
}

impl<'a> std::fmt::Debug for AstError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum AstErrorKind<'a> {
    Decimal,
    EndOfInput,
    ExpectedChar(AsciiChar),
    ExpectedCharInstead(AsciiChar, AsciiChar),
    ExpectedEof,
    ExpectedNewLine,
    InvalidScope(AsciiChar),
    NonAscii(u8),
    ReservedKeyword(Ascii<'a>),
    UnexpectedToken(AsciiChar),
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn expect_inline(offset: usize, line: usize, col: usize, length: usize) -> AstSpan {
        AstSpan {
            offset,
            line,
            col,
            end_offset: offset + length,
            end_line: line,
            end_col: col + length - 1,
        }
    }

    #[test]
    fn it_parses_a_while_statement<'a>() {
        let (_, s) =
            while_statement(Span::new(&b"while x < 42 {\n  draw_particle!(x)\n}"[..])).unwrap();
        assert_eq!(
            s,
            Statement::While {
                pos: AstSpan {
                    end_line: 3,
                    end_col: 1,
                    ..expect_inline(0, 1, 1, 36)
                },
                condition: Expression::BinaryOperation {
                    pos: expect_inline(6, 1, 7, 6),
                    operator: BinaryOperator::Lt,
                    left: Box::new(Expression::Variable(VariableIdentifier {
                        pos: expect_inline(6, 1, 7, 1),
                        name: Ascii::from(&b"x"[..]),
                    })),
                    right: Box::new(Expression::Literal {
                        pos: expect_inline(10, 1, 11, 2),
                        value: Value::Integer(42),
                    }),
                },
                body: vec![Statement::FunctionCall {
                    identifier: FunctionIdentifier {
                        pos: expect_inline(17, 2, 3, 17),
                        name: Ascii::from(&b"draw_particle"[..]),
                        scope: Some(FunctionScope {
                            token: AsciiChar::try_from(b'!').unwrap(),
                            pos: expect_inline(30, 2, 16, 1),
                        }),
                    },
                    args: vec![Expression::Variable(VariableIdentifier {
                        pos: expect_inline(32, 2, 18, 1),
                        name: Ascii::from(&b"x"[..]),
                    })],
                }],
            }
        );
    }

    #[test]
    fn it_detects_invalid_function_scopes<'a>() {
        if let nom::Err::Failure(err) = expression(Span::new(&b"foo#(x)"[..])).unwrap_err() {
            assert_eq!(
                err.explicit_kind,
                Some(AstErrorKind::InvalidScope(
                    AsciiChar::try_from(b'#').unwrap()
                ))
            )
        } else {
            panic!()
        }
    }

    #[test]
    fn it_treats_precedence_correctly<'a>() {
        let (_, c) = expression(Span::new(&b"a && b || c"[..])).unwrap();
        assert_eq!(
            c,
            Expression::BinaryOperation {
                pos: expect_inline(0, 1, 1, 11),
                operator: BinaryOperator::Or,
                left: Box::new(Expression::BinaryOperation {
                    pos: expect_inline(0, 1, 1, 6),
                    operator: BinaryOperator::And,
                    left: Box::new(Expression::Variable(VariableIdentifier {
                        pos: expect_inline(0, 1, 1, 1),
                        name: Ascii::from(&b"a"[..]),
                    })),
                    right: Box::new(Expression::Variable(VariableIdentifier {
                        pos: expect_inline(5, 1, 6, 1),
                        name: Ascii::from(&b"b"[..]),
                    })),
                }),
                right: Box::new(Expression::Variable(VariableIdentifier {
                    pos: expect_inline(10, 1, 11, 1),
                    name: Ascii::from(&b"c"[..]),
                })),
            },
        );
    }

    #[test]
    fn it_parses_a_constant_definition<'a>() {
        let (_, c) = constant_assignment(Span::new(&b"THIS_IS_CONSTANT = 42"[..])).unwrap();
        assert_eq!(
            c,
            Constant {
                identifier: ConstantIdentifier {
                    pos: expect_inline(0, 1, 1, 21),
                    name: Ascii::from(&b"THIS_IS_CONSTANT"[..]),
                },
                expression: Expression::Literal {
                    value: Value::Integer(42),
                    pos: expect_inline(19, 1, 20, 2)
                },
            }
        );
    }

    #[test]
    fn it_parses_a_sample_short_program() {
        let program = parse_nani(
            &br#"
THIS_IS_CONSTANT = 42
THIS_ALSO_IS = 666

initialize() {
  x = 20
  d = 2.5
  b = true
  s = "abc"
  foo!()
  r = bar(x, 42)
}
        "#[..],
        )
        .unwrap();
        assert_eq!(
            program,
            Program {
                constants: vec![
                    Constant {
                        identifier: ConstantIdentifier {
                            pos: expect_inline(1, 2, 1, 21),
                            name: Ascii::from(&b"THIS_IS_CONSTANT"[..]),
                        },
                        expression: Expression::Literal {
                            value: Value::Integer(42),
                            pos: expect_inline(20, 2, 20, 2)
                        },
                    },
                    Constant {
                        identifier: ConstantIdentifier {
                            pos: expect_inline(23, 3, 1, 18),
                            name: Ascii::from(&b"THIS_ALSO_IS"[..]),
                        },
                        expression: Expression::Literal {
                            value: Value::Integer(666),
                            pos: expect_inline(38, 3, 16, 3)
                        },
                    }
                ],
                props: vec![],
                functions: vec![FunctionDefinition {
                    identifier: FunctionIdentifier {
                        pos: AstSpan {
                            offset: 43,
                            line: 5,
                            col: 1,
                            end_offset: 127,
                            end_line: 12,
                            end_col: 1,
                        },
                        name: Ascii::from(&b"initialize"[..]),
                        scope: None,
                    },
                    args: vec![],
                    return_type: None,
                    body: vec![
                        Statement::VariableAssignment {
                            identifier: VariableIdentifier {
                                pos: expect_inline(60, 6, 3, 6),
                                name: Ascii::from(&b"x"[..]),
                            },
                            expression: Expression::Literal {
                                pos: expect_inline(64, 6, 7, 2),
                                value: Value::Integer(20),
                            },
                        },
                        Statement::VariableAssignment {
                            identifier: VariableIdentifier {
                                pos: expect_inline(69, 7, 3, 7),
                                name: Ascii::from(&b"d"[..]),
                            },
                            expression: Expression::Literal {
                                pos: expect_inline(73, 7, 7, 3),
                                value: Value::Decimal(2.5),
                            },
                        },
                        Statement::VariableAssignment {
                            identifier: VariableIdentifier {
                                pos: expect_inline(79, 8, 3, 8),
                                name: Ascii::from(&b"b"[..]),
                            },
                            expression: Expression::Literal {
                                pos: expect_inline(83, 8, 7, 4),
                                value: Value::Bool(true),
                            },
                        },
                        Statement::VariableAssignment {
                            identifier: VariableIdentifier {
                                pos: expect_inline(90, 9, 3, 9),
                                name: Ascii::from(&b"s"[..]),
                            },
                            expression: Expression::Literal {
                                pos: expect_inline(94, 9, 7, 5),
                                value: Value::String(Ascii::from(&b"abc"[..])),
                            },
                        },
                        Statement::FunctionCall {
                            identifier: FunctionIdentifier {
                                pos: expect_inline(102, 10, 3, 6),
                                name: Ascii::from(&b"foo"[..]),
                                scope: Some(FunctionScope {
                                    token: AsciiChar::try_from(b'!').unwrap(),
                                    pos: expect_inline(105, 10, 6, 1),
                                }),
                            },
                            args: vec![],
                        },
                        Statement::VariableAssignment {
                            identifier: VariableIdentifier {
                                pos: expect_inline(111, 11, 3, 14),
                                name: Ascii::from(&b"r"[..]),
                            },
                            expression: Expression::FunctionCall {
                                identifier: FunctionIdentifier {
                                    pos: expect_inline(115, 11, 7, 10),
                                    name: Ascii::from(&b"bar"[..]),
                                    scope: None,
                                },
                                args: vec![
                                    Expression::Variable(VariableIdentifier {
                                        pos: expect_inline(119, 11, 11, 1),
                                        name: Ascii::from(&b"x"[..]),
                                    }),
                                    Expression::Literal {
                                        pos: expect_inline(122, 11, 14, 2),
                                        value: Value::Integer(42),
                                    }
                                ],
                            },
                        },
                    ],
                }],
            }
        );
    }
}
