#[macro_use]
extern crate nom;

use crate::span::{position, Span};
use nom::{
    branch::{alt, permutation},
    bytes::complete::{tag, take_while1},
    // see the "streaming/complete" paragraph lower for an explanation of these submodules
    character::complete::{digit1, space0},
    combinator::{opt, recognize},
    error::{ErrorKind, ParseError},
    lib::std::ops::RangeFrom,
    multi::{many0, many1, separated_list, separated_nonempty_list},
    sequence::{delimited, terminated},
    IResult,
    InputIter,
    InputLength,
    Slice,
};
use std::convert::TryFrom;

mod span;

#[derive(Clone, Copy, PartialEq)]
pub struct Ascii<'a> {
    bytes: &'a [u8],
}

impl<'a> std::fmt::Display for Ascii<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", unsafe {
            String::from_utf8_unchecked(Vec::from(self.bytes))
        })
    }
}

impl<'a> std::fmt::Debug for Ascii<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "b\"{}\"", self)
    }
}

impl<'a> From<&'a [u8]> for Ascii<'a> {
    fn from(bytes: &'a [u8]) -> Self {
        Ascii { bytes }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct AsciiChar {
    byte: u8,
}

impl std::fmt::Display for AsciiChar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", char::from(self.byte))
    }
}

impl std::fmt::Debug for AsciiChar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "b\'{}\'", self)
    }
}

impl TryFrom<u8> for AsciiChar {
    type Error = &'static str;

    fn try_from(byte: u8) -> Result<Self, Self::Error> {
        if byte < 0xF0 {
            Ok(AsciiChar { byte })
        } else {
            Err("Given byte is not ASCII")
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Program<'a> {
    constants: Vec<ConstantAssignment<'a>>,
    functions: Vec<FunctionDefinition<'a>>,
}

#[derive(Debug, PartialEq)]
struct ConstantAssignment<'a> {
    name: Ascii<'a>,
    expression: Expression<'a>,
    pos: Span<'a>,
}

#[derive(Debug, PartialEq)]
struct FunctionDefinition<'a> {
    pos: Span<'a>,
    name: Ascii<'a>,
    scope: Option<FunctionScope<'a>>,
    args: Vec<FunctionArgument<'a>>,
    body: Vec<Statement<'a>>,
}

#[derive(Debug, PartialEq)]
struct FunctionArgument<'a> {
    pos: Span<'a>,
    name: Ascii<'a>,
    vartype: Vartype,
}

#[derive(Debug, PartialEq)]
enum Statement<'a> {
    VariableAssignment {
        name: Ascii<'a>,
        expression: Expression<'a>,
        pos: Span<'a>,
    },
    FunctionCall {
        pos: Span<'a>,
        name: Ascii<'a>,
        scope: Option<FunctionScope<'a>>,
        args: Box<Vec<Expression<'a>>>,
    },
}

#[derive(Debug, PartialEq)]
enum Expression<'a> {
    Parens {
        expression: Box<Expression<'a>>,
        pos: Span<'a>,
    },
    FunctionCall {
        pos: Span<'a>,
        name: Ascii<'a>,
        scope: Option<FunctionScope<'a>>,
        args: Box<Vec<Expression<'a>>>,
    },
    Literal {
        pos: Span<'a>,
        value: Value<'a>,
    },
}

#[derive(Debug, PartialEq)]
struct FunctionScope<'a> {
    pos: Span<'a>,
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

fn constant_assignment(s: Span) -> IResult<Span, ConstantAssignment> {
    let (s, pos) = position(s)?;
    let (s, name) = snakecase_upper(s)?;
    let (s, _) = delimited(space0, byte(b'='), space0)(s)?;
    let (s, expr) = expression(s)?;
    return Ok((
        s,
        ConstantAssignment {
            name,
            expression: expr,
            pos,
        },
    ));
}

fn function_definition(s: Span) -> IResult<Span, FunctionDefinition> {
    let (s, pos) = position(s)?;
    let (s, name) = snakecase_lower(s)?;
    let (s, scope) = opt(function_scope)(s)?;
    let (s, args) = trim(delimited(
        byte(b'('),
        separated_list(trim(byte(b',')), function_argument),
        byte(b')'),
    ))(s)?;
    let (s, body) = delimited(byte(b'{'), many0(on_a_line(statement)), byte(b'}'))(s)?;
    Ok((
        s,
        FunctionDefinition {
            pos,
            name,
            scope,
            args,
            body,
        },
    ))
}

fn function_argument(s: Span) -> IResult<Span, FunctionArgument> {
    let (s, pos) = position(s)?;
    let (s, name) = snakecase_lower(s)?;
    Ok((
        s,
        FunctionArgument {
            pos,
            name,
            vartype: Vartype::Inferred,
        },
    ))
}

#[inline]
fn statement(s: Span) -> IResult<Span, Statement> {
    alt((variable_assignment, function_call_statement))(s)
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

#[inline]
fn expression(s: Span) -> IResult<Span, Expression> {
    alt((literal, parens))(s)
}

fn variable_assignment(s: Span) -> IResult<Span, Statement> {
    let (s, pos) = position(s)?;
    let (s, name) = snakecase_lower(s)?;
    let (s, _) = delimited(space0, byte(b'='), space0)(s)?;
    let (s, expr) = expression(s)?;
    return Ok((
        s,
        Statement::VariableAssignment {
            name,
            expression: expr,
            pos,
        },
    ));
}

fn function_call(s: Span) -> IResult<Span, Expression> {
    let (s, pos) = position(s)?;
    let (s, name) = snakecase_lower(s)?;
    let (s, scope) = opt(function_scope)(s)?;
    let (s, args) = delimited(
        byte(b'('),
        separated_list(trim(byte(b',')), expression),
        byte(b')'),
    )(s)?;
    Ok((
        s,
        Expression::FunctionCall {
            pos,
            name,
            scope,
            args: Box::new(args),
        },
    ))
}

fn function_scope(s: Span) -> IResult<Span, FunctionScope> {
    let (s, pos) = position(s)?;
    let (s, token) = alt((byte(b'!'), byte(b'?')))(s)?;
    let token = AsciiChar::try_from(token)
        .or_else(|_| Err(nom::Err::Error(error_position!(s, ErrorKind::ParseTo))))?;
    Ok((s, FunctionScope { pos, token }))
}

fn parens(s: Span) -> IResult<Span, Expression> {
    let (s, pos) = position(s)?;
    let (s, expr) = delimited(byte(b'('), expression, byte(b')'))(s)?;
    Ok((
        s,
        Expression::Parens {
            expression: Box::from(expr),
            pos,
        },
    ))
}

fn literal(s: Span) -> IResult<Span, Expression> {
    let (s, pos) = position(s)?;
    let (s, value) = value(s)?;
    Ok((s, Expression::Literal { pos, value }))
}

#[inline]
fn value(s: Span) -> IResult<Span, Value> {
    alt((value_bool, value_decimal, value_integer))(s)
}

fn value_bool(s: Span) -> IResult<Span, Value> {
    let (s, value) = if let (s, Some(_)) = opt(tag(&b"true"[..]))(s)? {
        (s, true)
    } else {
        let (s, _) = tag(&b"false"[..])(s)?;
        (s, false)
    };
    Ok((s, Value::Bool(value)))
}

fn value_integer(s: Span) -> IResult<Span, Value> {
    let (s, i) = digit1(s)?;
    let i = i
        .fragment
        .to_string()
        .parse()
        .or_else(|_| Err(nom::Err::Error(error_position!(s, ErrorKind::Digit))))?;
    Ok((s, Value::Integer(i)))
}

fn value_decimal(s: Span) -> IResult<Span, Value> {
    let (s, n) = digit1(s)?;
    let (s, _) = byte(b'.')(s)?;
    let (s, d) = digit1(s)?;
    let i = format!("{}.{}", n.fragment, d.fragment)
        .parse()
        .or_else(|_| Err(nom::Err::Error(error_position!(s, ErrorKind::Float))))?;
    Ok((s, Value::Decimal(i)))
}

fn snakecase_upper(s: Span) -> IResult<Span, Ascii> {
    let (s, n) = recognize(separated_nonempty_list(byte(b'_'), take_while1(is_upper)))(s)?;
    Ok((s, n.fragment))
}

fn snakecase_lower(s: Span) -> IResult<Span, Ascii> {
    let (s, n) = recognize(separated_nonempty_list(byte(b'_'), take_while1(is_lower)))(s)?;
    Ok((s, n.fragment))
}

#[inline]
fn is_upper(chr: u8) -> bool {
    chr >= 0x41 && chr <= 0x5A
}

#[inline]
fn is_lower(chr: u8) -> bool {
    chr >= 0x61 && chr <= 0x7A
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

#[inline]
fn space<'a>(input: Span<'a>) -> IResult<Span<'a>, u8> {
    byte(b' ')(input)
}

#[inline]
fn newline<'a>(input: Span<'a>) -> IResult<Span<'a>, u8> {
    byte(b'\n')(input)
}

#[inline]
fn eof<'a>(input: Span<'a>) -> IResult<Span<'a>, u8> {
    if input.input_len() == 0 {
        Ok((input, b'\0'))
    } else {
        Err(nom::Err::Error(error_position!(input, ErrorKind::Eof)))
    }
}

//#[inline]
//fn multispace<'a>(input: Span<'a>) -> IResult<Span<'a>, u8> {
//    alt((byte(b' '), byte(b'\n')))(input)
//}

pub fn nani<'a>(input: Ascii<'a>) -> IResult<Span<'a>, Program<'a>> {
    let s = Span::new(input);
    let (s, constants) = many0(on_a_line(constant_assignment))(s)?;
    let (s, functions) = many0(on_a_line(function_definition))(s)?;
    let (s, _) = many0(alt((space, newline)))(s)?;
    let (s, _) = eof(s)?;
    Ok((
        s,
        Program {
            constants,
            functions,
        },
    ))
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn it_parses_a_constant_definition<'a>() {
        let (_, c) =
            constant_assignment(Span::new(Ascii::from(&b"THIS_IS_CONSTANT = 42"[..]))).unwrap();
        let expected = assert_eq!(
            c,
            ConstantAssignment {
                name: Ascii::from(&b"THIS_IS_CONSTANT"[..]),
                expression: Expression::Literal {
                    value: Value::Integer(42),
                    pos: Span {
                        offset: 19,
                        line: 1,
                        col: 20,
                        fragment: Ascii::from(&b""[..]),
                    }
                },
                pos: Span {
                    offset: 0,
                    line: 1,
                    col: 1,
                    fragment: Ascii::from(&b""[..]),
                }
            }
        );
    }

    #[test]
    fn it_parses_constant_definitions() {
        let (_, program) = nani(Ascii::from(
            &br#"
THIS_IS_CONSTANT = 42
THIS_ALSO_IS = 666

initialize() {
  x = 20
  d = 2.5
  b = true
}
        "#[..],
        ))
        .unwrap();
        assert_eq!(
            program,
            Program {
                constants: vec![
                    ConstantAssignment {
                        name: Ascii::from(&b"THIS_IS_CONSTANT"[..]),
                        expression: Expression::Literal {
                            value: Value::Integer(42),
                            pos: Span {
                                offset: 20,
                                line: 2,
                                col: 20,
                                fragment: Ascii::from(&b""[..]),
                            }
                        },
                        pos: Span {
                            offset: 1,
                            line: 2,
                            col: 1,
                            fragment: Ascii::from(&b""[..]),
                        }
                    },
                    ConstantAssignment {
                        name: Ascii::from(&b"THIS_ALSO_IS"[..]),
                        expression: Expression::Literal {
                            value: Value::Integer(666),
                            pos: Span {
                                offset: 38,
                                line: 3,
                                col: 16,
                                fragment: Ascii::from(&b""[..]),
                            }
                        },
                        pos: Span {
                            offset: 23,
                            line: 3,
                            col: 1,
                            fragment: Ascii::from(&b""[..]),
                        }
                    }
                ],
                functions: vec![FunctionDefinition {
                    pos: Span {
                        offset: 43,
                        line: 5,
                        col: 1,
                        fragment: Ascii::from(&b""[..]),
                    },
                    name: Ascii::from(&b"initialize"[..]),
                    scope: None,
                    args: vec![],
                    body: vec![
                        Statement::VariableAssignment {
                            name: Ascii::from(&b"x"[..]),
                            expression: Expression::Literal {
                                pos: Span {
                                    offset: 64,
                                    line: 6,
                                    col: 7,
                                    fragment: Ascii::from(&b""[..]),
                                },
                                value: Value::Integer(20),
                            },
                            pos: Span {
                                offset: 60,
                                line: 6,
                                col: 3,
                                fragment: Ascii::from(&b""[..]),
                            },
                        },
                        Statement::VariableAssignment {
                            name: Ascii::from(&b"d"[..]),
                            expression: Expression::Literal {
                                pos: Span {
                                    offset: 73,
                                    line: 7,
                                    col: 7,
                                    fragment: Ascii::from(&b""[..]),
                                },
                                value: Value::Decimal(2.5),
                            },
                            pos: Span {
                                offset: 69,
                                line: 7,
                                col: 3,
                                fragment: Ascii::from(&b""[..]),
                            },
                        },
                        Statement::VariableAssignment {
                            name: Ascii::from(&b"b"[..]),
                            expression: Expression::Literal {
                                pos: Span {
                                    offset: 83,
                                    line: 8,
                                    col: 7,
                                    fragment: Ascii::from(&b""[..]),
                                },
                                value: Value::Bool(true),
                            },
                            pos: Span {
                                offset: 79,
                                line: 8,
                                col: 3,
                                fragment: Ascii::from(&b""[..]),
                            },
                        },
                    ],
                }],
            }
        );
    }
}
