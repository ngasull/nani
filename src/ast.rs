use crate::{
    ascii::{is_ascii, is_lower, is_upper, Ascii, AsciiChar},
    span::{position, AstSpan, Span},
};
use nom::{
    branch::alt,
    bytes::complete::{escaped, is_not, tag, take_while1},
    character::complete::{digit1, space0, space1},
    combinator::{opt, recognize},
    error::{ErrorKind, ParseError},
    lib::std::ops::RangeFrom,
    multi::{many0, separated_list, separated_nonempty_list},
    sequence::delimited,
    AsBytes, IResult, InputIter, InputLength, Slice,
};
use std::convert::TryFrom;

#[derive(Debug, PartialEq)]
pub struct Program<'a> {
    constants: Vec<ConstantAssignment<'a>>,
    functions: Vec<FunctionDefinition<'a>>,
}

#[derive(Debug, PartialEq)]
struct ConstantAssignment<'a> {
    name: Ascii<'a>,
    expression: Expression<'a>,
    pos: AstSpan,
}

#[derive(Debug, PartialEq)]
struct FunctionDefinition<'a> {
    pos: AstSpan,
    name: Ascii<'a>,
    scope: Option<FunctionScope>,
    args: Vec<FunctionArgument<'a>>,
    body: Vec<Statement<'a>>,
}

#[derive(Debug, PartialEq)]
struct FunctionArgument<'a> {
    pos: AstSpan,
    name: Ascii<'a>,
    vartype: Vartype,
}

#[derive(Debug, PartialEq)]
enum Statement<'a> {
    VariableAssignment {
        name: Ascii<'a>,
        expression: Expression<'a>,
        pos: AstSpan,
    },
    FunctionCall {
        pos: AstSpan,
        name: Ascii<'a>,
        scope: Option<FunctionScope>,
        args: Box<Vec<Expression<'a>>>,
    },
    If {
        pos: AstSpan,
        condition: Expression<'a>,
        body: Box<Vec<Statement<'a>>>,
    },
}

#[derive(Debug, PartialEq)]
enum Expression<'a> {
    Parens {
        expression: Box<Expression<'a>>,
        pos: AstSpan,
    },
    FunctionCall {
        pos: AstSpan,
        name: Ascii<'a>,
        scope: Option<FunctionScope>,
        args: Box<Vec<Expression<'a>>>,
    },
    Constant {
        pos: AstSpan,
        name: Ascii<'a>,
    },
    Variable {
        pos: AstSpan,
        name: Ascii<'a>,
    },
    Literal {
        pos: AstSpan,
        value: Value<'a>,
    },
}

#[derive(Debug, PartialEq)]
struct FunctionScope {
    pos: AstSpan,
    token: AsciiChar,
}

#[derive(Debug, PartialEq)]
enum Value<'a> {
    Bool(bool),
    Integer(i32),
    Decimal(f32),
    String(Ascii<'a>),
}

#[derive(Debug, PartialEq)]
enum Vartype {
    Inferred,
}

static KEYWORD_IF: &[u8] = b"if";
static KEYWORD_FOR: &[u8] = b"for";
static KEYWORD_WHILE: &[u8] = b"while";
static KEYWORD_TRUE: &[u8] = b"true";
static KEYWORD_FALSE: &[u8] = b"false";
static KEYWORD_RETURN: &[u8] = b"return";
static KEYWORD_BREAK: &[u8] = b"break";
static KEYWORD_CONTINUE: &[u8] = b"continue";

static RESERVED_KEYWORDS: &[&[u8]] = &[
    KEYWORD_IF,
    KEYWORD_FOR,
    KEYWORD_WHILE,
    KEYWORD_TRUE,
    KEYWORD_FALSE,
    KEYWORD_RETURN,
    KEYWORD_BREAK,
    KEYWORD_CONTINUE,
];

fn constant_assignment(s: Span) -> IResult<Span, ConstantAssignment> {
    let (s, pos) = position(s)?;
    let (s, name) = not_reserved(snakecase_upper)(s)?;
    let (s, _) = required(delimited(space0, byte(b'='), space0))(s)?;
    let (s, expr) = required(expression)(s)?;
    let (s, end_pos) = position(s)?;
    return Ok((
        s,
        ConstantAssignment {
            name: Ascii::from(name),
            expression: expr,
            pos: pos.to(end_pos),
        },
    ));
}

fn function_definition(s: Span) -> IResult<Span, FunctionDefinition> {
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
            pos: pos.to(end_pos),
            name: Ascii::from(name),
            scope,
            args,
            body,
        },
    ))
}

fn function_argument(s: Span) -> IResult<Span, FunctionArgument> {
    let (s, pos) = position(s)?;
    let (s, name) = not_reserved(snakecase_lower)(s)?;
    let (s, end_pos) = position(s)?;
    Ok((
        s,
        FunctionArgument {
            pos: pos.to(end_pos),
            name: Ascii::from(name),
            vartype: Vartype::Inferred,
        },
    ))
}

#[inline]
fn statement(s: Span) -> IResult<Span, Statement> {
    alt((variable_assignment, if_statement, function_call_statement))(s)
}

fn function_call_statement(s: Span) -> IResult<Span, Statement> {
    let (s, fncall) = function_call(s)?;
    if let Expression::FunctionCall {
        pos,
        name,
        scope,
        args,
    } = fncall
    {
        Ok((
            s,
            Statement::FunctionCall {
                pos,
                name,
                scope,
                args,
            },
        ))
    } else {
        panic!()
    }
}

fn variable_assignment(s: Span) -> IResult<Span, Statement> {
    let (s, pos) = position(s)?;
    let (s, name) = not_reserved(snakecase_lower)(s)?;
    let (s, _) = delimited(space0, byte(b'='), space0)(s)?;
    let (s, expr) = required(expression)(s)?;
    let (s, end_pos) = position(s)?;
    return Ok((
        s,
        Statement::VariableAssignment {
            name: Ascii::from(name),
            expression: expr,
            pos: pos.to(end_pos),
        },
    ));
}

fn if_statement(s: Span) -> IResult<Span, Statement> {
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
            body: Box::new(body),
        },
    ));
}

#[inline]
fn expression(s: Span) -> IResult<Span, Expression> {
    // Order matters here, for instance function call must be checked before variable
    alt((function_call, literal, constant, variable, parens))(s)
}

fn function_call(s: Span) -> IResult<Span, Expression> {
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
            pos: pos.to(end_pos),
            name: Ascii::from(name),
            scope,
            args: Box::new(args),
        },
    ))
}

fn function_scope(s: Span) -> IResult<Span, FunctionScope> {
    let (s, pos) = position(s)?;
    let (s, token) = alt((byte(b'!'), byte(b'?')))(s)?;
    let token = AsciiChar::try_from(token)
        .or_else(|_| Err(nom::Err::Failure(error_position!(s, ErrorKind::ParseTo))))?;
    let (s, end_pos) = position(s)?;
    Ok((
        s,
        FunctionScope {
            pos: pos.to(end_pos),
            token,
        },
    ))
}

fn parens(s: Span) -> IResult<Span, Expression> {
    let (s, pos) = position(s)?;
    let (s, expr) = delimited(byte(b'('), expression, byte(b')'))(s)?;
    let (s, end_pos) = position(s)?;
    Ok((
        s,
        Expression::Parens {
            expression: Box::from(expr),
            pos: pos.to(end_pos),
        },
    ))
}

fn literal(s: Span) -> IResult<Span, Expression> {
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

fn constant(s: Span) -> IResult<Span, Expression> {
    let (s, pos) = position(s)?;
    let (s, name) = not_reserved(snakecase_upper)(s)?;
    let (s, end_pos) = position(s)?;
    Ok((
        s,
        Expression::Constant {
            pos: pos.to(end_pos),
            name: Ascii::from(name),
        },
    ))
}

fn variable(s: Span) -> IResult<Span, Expression> {
    let (s, pos) = position(s)?;
    let (s, name) = not_reserved(snakecase_lower)(s)?;
    let (s, end_pos) = position(s)?;
    Ok((
        s,
        Expression::Variable {
            pos: pos.to(end_pos),
            name: Ascii::from(name),
        },
    ))
}

#[inline]
fn value(s: Span) -> IResult<Span, Value> {
    alt((value_bool, value_decimal, value_integer, value_string))(s)
}

fn value_bool(s: Span) -> IResult<Span, Value> {
    let (s, value) = if let (s, Some(_)) = opt(tag(KEYWORD_TRUE))(s)? {
        (s, true)
    } else {
        let (s, _) = tag(KEYWORD_FALSE)(s)?;
        (s, false)
    };
    Ok((s, Value::Bool(value)))
}

fn value_integer(s: Span) -> IResult<Span, Value> {
    let (s, i) = digit1(s)?;
    let i = Ascii::from(i.fragment)
        .to_string()
        .parse()
        .or_else(|_| Err(nom::Err::Failure(error_position!(s, ErrorKind::Digit))))?;
    Ok((s, Value::Integer(i)))
}

fn value_decimal(s: Span) -> IResult<Span, Value> {
    let (s, n) = digit1(s)?;
    let (s, _) = byte(b'.')(s)?;
    let (s, d) = digit1(s)?;
    let i = format!("{}.{}", Ascii::from(n.fragment), Ascii::from(d.fragment))
        .parse()
        .or_else(|_| Err(nom::Err::Failure(error_position!(s, ErrorKind::Float))))?;
    Ok((s, Value::Decimal(i)))
}

fn value_string(s: Span) -> IResult<Span, Value> {
    let (s, str_literal) = delimited(
        byte(b'"'),
        required(escaped(
            is_not("\"\\"),
            '\\',
            alt((byte(b'"'), byte(b'\\'))),
        )),
        byte(b'"'),
    )(s)?;
    if str_literal.fragment.iter().all(|&c| is_ascii(c)) {
        Ok((s, Value::String(Ascii::from(str_literal.fragment))))
    } else {
        Err(nom::Err::Failure(error_position!(s, ErrorKind::Verify)))
    }
}

fn snakecase_upper(s: Span) -> IResult<Span, &[u8]> {
    let (s, n) = recognize(separated_nonempty_list(byte(b'_'), take_while1(is_upper)))(s)?;
    Ok((s, n.fragment))
}

fn snakecase_lower(s: Span) -> IResult<Span, &[u8]> {
    let (s, n) = recognize(separated_nonempty_list(byte(b'_'), take_while1(is_lower)))(s)?;
    Ok((s, n.fragment))
}

fn byte<I, Error: ParseError<I>>(b: u8) -> impl Fn(I) -> IResult<I, u8, Error>
where
    I: Slice<RangeFrom<usize>> + InputIter<Item = u8>,
{
    move |i: I| match (i).iter_elements().next().map(|t: u8| (b, t == b)) {
        Some((b, true)) => Ok((i.slice(1..), b)),
        _ => Err(nom::Err::Error(Error::from_char(i, char::from(b)))),
    }
}

fn trim<'a, O, F>(f: F) -> impl Fn(Span<'a>) -> IResult<Span<'a>, O>
where
    F: Fn(Span<'a>) -> IResult<Span<'a>, O>,
{
    move |input: Span| {
        let (input, _) = many0(space)(input)?;
        let (input, o) = f(input)?;
        let (input, _) = many0(space)(input)?;
        Ok((input, o))
    }
}

fn on_a_line<'a, O, F>(f: F) -> impl Fn(Span<'a>) -> IResult<Span<'a>, O>
where
    F: Fn(Span<'a>) -> IResult<Span<'a>, O>,
{
    move |input: Span| {
        let (input, _) = many0(alt((space, newline)))(input)?;
        let (input, o) = f(input)?;
        let (input, _) = many0(space)(input)?;
        let (input, _) = alt((newline, eof))(input)?;
        Ok((input, o))
    }
}

fn finishes_multiline<'a, O, F>(f: F) -> impl Fn(Span<'a>) -> IResult<Span<'a>, O>
where
    F: Fn(Span<'a>) -> IResult<Span<'a>, O>,
{
    move |input: Span| {
        let (input, _) = many0(alt((space, newline)))(input)?;
        let (input, o) = f(input)?;
        Ok((input, o))
    }
}

#[inline]
fn space<'a>(input: Span<'a>) -> IResult<Span<'a>, u8> {
    byte(b' ')(input)
}

#[inline]
fn newline<'a>(input: Span<'a>) -> IResult<Span<'a>, u8> {
    byte(b'\n')(input)
}

#[inline]
fn eof<'a, Error: ParseError<Span<'a>>>(input: Span<'a>) -> IResult<Span<'a>, u8, Error> {
    if input.input_len() == 0 {
        Ok((input, b'\0'))
    } else {
        Err(nom::Err::Error(Error::from_char(
            input,
            char::from(input.fragment[0]),
        )))
    }
}

fn required<'a, O, F>(f: F) -> impl Fn(Span<'a>) -> IResult<Span<'a>, O>
where
    F: Fn(Span<'a>) -> IResult<Span<'a>, O>,
{
    move |input: Span| match f(input) {
        Ok(io) => Ok(io),
        Err(nom::Err::Error(e)) | Err(nom::Err::Failure(e)) => Err(nom::Err::Failure(e)),
        Err(nom::Err::Incomplete(n)) => Err(nom::Err::Incomplete(n)),
    }
}

fn not_reserved<'a, O, F>(f: F) -> impl Fn(Span<'a>) -> IResult<Span<'a>, O>
where
    F: Fn(Span<'a>) -> IResult<Span<'a>, O>,
    O: AsBytes,
{
    move |input: Span| {
        let (new_input, res) = f(input)?;
        if RESERVED_KEYWORDS.iter().any(|&k| k == res.as_bytes()) {
            Err(nom::Err::Failure(error_position!(input, ErrorKind::Tag)))
        } else {
            Ok((new_input, res))
        }
    }
}

fn nani_structure<'a>(input: &'a [u8]) -> IResult<Span<'a>, Program<'a>> {
    let s = Span::new(input);
    let (s, constants) = many0(on_a_line(constant_assignment))(s)?;
    let (s, functions) = many0(on_a_line(function_definition))(s)?;
    let (s, _) = many0(alt((space, newline)))(s)?;
    let (s, _) = required(eof)(s)?;
    Ok((
        s,
        Program {
            constants,
            functions,
        },
    ))
}

pub fn parse_nani<'a>(input: &'a [u8]) -> Result<Program<'a>, String> {
    match nani_structure(input) {
        Ok((_, program)) => Ok(program),
        Err(nom::Err::Failure((s, _))) if s.fragment.len() > 0 => {
            match AsciiChar::try_from(s.fragment[0]) {
                Ok(token) => Err(format!(
                    "Unexpected token [L{} C{}]: {}",
                    s.line, s.col, token
                )),
                Err(_) => Err(format!(
                    "Unexpected non-ASCII byte at [L{} C{}]: {}",
                    s.line, s.col, s.fragment[0]
                )),
            }
        }
        _ => panic!("Unexpected end of parsing"),
    }
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
    fn it_parses_a_constant_definition<'a>() {
        let (_, c) = constant_assignment(Span::new(&b"THIS_IS_CONSTANT = 42"[..])).unwrap();
        assert_eq!(
            c,
            ConstantAssignment {
                name: Ascii::from(&b"THIS_IS_CONSTANT"[..]),
                expression: Expression::Literal {
                    value: Value::Integer(42),
                    pos: expect_inline(19, 1, 20, 2)
                },
                pos: expect_inline(0, 1, 1, 21),
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
                    ConstantAssignment {
                        name: Ascii::from(&b"THIS_IS_CONSTANT"[..]),
                        expression: Expression::Literal {
                            value: Value::Integer(42),
                            pos: expect_inline(20, 2, 20, 2)
                        },
                        pos: expect_inline(1, 2, 1, 21)
                    },
                    ConstantAssignment {
                        name: Ascii::from(&b"THIS_ALSO_IS"[..]),
                        expression: Expression::Literal {
                            value: Value::Integer(666),
                            pos: expect_inline(38, 3, 16, 3)
                        },
                        pos: expect_inline(23, 3, 1, 18)
                    }
                ],
                functions: vec![FunctionDefinition {
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
                    args: vec![],
                    body: vec![
                        Statement::VariableAssignment {
                            name: Ascii::from(&b"x"[..]),
                            expression: Expression::Literal {
                                pos: expect_inline(64, 6, 7, 2),
                                value: Value::Integer(20),
                            },
                            pos: expect_inline(60, 6, 3, 6),
                        },
                        Statement::VariableAssignment {
                            name: Ascii::from(&b"d"[..]),
                            expression: Expression::Literal {
                                pos: expect_inline(73, 7, 7, 3),
                                value: Value::Decimal(2.5),
                            },
                            pos: expect_inline(69, 7, 3, 7),
                        },
                        Statement::VariableAssignment {
                            name: Ascii::from(&b"b"[..]),
                            expression: Expression::Literal {
                                pos: expect_inline(83, 8, 7, 4),
                                value: Value::Bool(true),
                            },
                            pos: expect_inline(79, 8, 3, 8),
                        },
                        Statement::VariableAssignment {
                            name: Ascii::from(&b"s"[..]),
                            expression: Expression::Literal {
                                pos: expect_inline(94, 9, 7, 5),
                                value: Value::String(Ascii::from(&b"abc"[..])),
                            },
                            pos: expect_inline(90, 9, 3, 9),
                        },
                        Statement::FunctionCall {
                            name: Ascii::from(&b"foo"[..]),
                            scope: Some(FunctionScope {
                                token: AsciiChar::try_from(b'!').unwrap(),
                                pos: expect_inline(105, 10, 6, 1),
                            }),
                            args: Box::new(vec![]),
                            pos: expect_inline(102, 10, 3, 6),
                        },
                        Statement::VariableAssignment {
                            name: Ascii::from(&b"r"[..]),
                            expression: Expression::FunctionCall {
                                pos: expect_inline(115, 11, 7, 10),
                                name: Ascii::from(&b"bar"[..]),
                                scope: None,
                                args: Box::new(vec![
                                    Expression::Variable {
                                        pos: expect_inline(119, 11, 11, 1),
                                        name: Ascii::from(&b"x"[..]),
                                    },
                                    Expression::Literal {
                                        pos: expect_inline(122, 11, 14, 2),
                                        value: Value::Integer(42),
                                    }
                                ]),
                            },
                            pos: expect_inline(111, 11, 3, 14),
                        },
                    ],
                }],
            }
        );
    }
}
