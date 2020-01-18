#[macro_use]
extern crate nom;

use crate::span::{position, Span};
use nom::{
    branch::{alt, permutation},
    bytes::complete::{is_a, take_while1},
    // see the "streaming/complete" paragraph lower for an explanation of these submodules
    character::complete::{char, digit1, space0},
    combinator::{opt, recognize},
    error::{ErrorKind, ParseError},
    lib::std::ops::RangeFrom,
    multi::{many0, many1},
    sequence::{delimited, terminated},
    IResult,
    InputIter,
    InputLength,
    Slice,
};

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

//impl<'a> ToString for Ascii<'a> {
//    fn to_string(&self) -> String {
//        unsafe { String::from_utf8_unchecked(Vec::from(self.bytes)) }
//    }
//}

impl<'a> From<&'a [u8]> for Ascii<'a> {
    fn from(bytes: &'a [u8]) -> Self {
        Ascii { bytes }
    }
}

#[derive(Debug, PartialEq)]
pub struct Program<'a> {
    constants: Vec<Constant<'a>>,
}

#[derive(Debug, PartialEq)]
struct Constant<'a> {
    name: Ascii<'a>,
    literal: Literal<'a>,
    pos: Span<'a>,
}

#[derive(Debug, PartialEq)]
struct Literal<'a> {
    value: Value<'a>,
    pos: Span<'a>,
}

#[derive(Debug, PartialEq)]
enum Value<'a> {
    Integer(i32),
    Decimal(f32),
    String(Ascii<'a>),
}

fn constant(s: Span) -> IResult<Span, Constant> {
    let (s, pos) = position(s)?;
    let (s, n) = recognize(terminated(
        many0(terminated(take_while1(is_capital), byte(b'_'))),
        take_while1(is_capital),
    ))(s)?;
    let (s, _) = delimited(space0, byte(b'='), space0)(s)?;
    let (s, l) = literal(s)?;
    return Ok((
        s,
        Constant {
            name: n.fragment,
            literal: l,
            pos,
        },
    ));
}

#[inline]
pub fn is_capital(chr: u8) -> bool {
    chr >= 0x41 && chr <= 0x5A
}

#[inline]
pub fn is_lower(chr: u8) -> bool {
    chr >= 0x61 && chr <= 0x7A
}

fn literal(s: Span) -> IResult<Span, Literal> {
    let (s, pos) = position(s)?;
    let (s, value) = value(s)?;
    Ok((s, Literal { pos, value }))
}

#[inline]
fn value(s: Span) -> IResult<Span, Value> {
    alt((value_integer, value_decimal))(s)
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

pub fn byte<I, Error: ParseError<I>>(b: u8) -> impl Fn(I) -> IResult<I, u8, Error>
where
    I: Slice<RangeFrom<usize>> + InputIter<Item = u8>,
{
    move |i: I| match (i).iter_elements().next().map(|t: u8| (b, t == b)) {
        Some((b, true)) => Ok((i.slice(1..), b)),
        _ => Err(nom::Err::Error(Error::from_char(i, char::from(b)))),
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
    let (s, constants) = many0(on_a_line(constant))(s)?;
    Ok((s, Program { constants }))
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn it_parses_a_constant_definition<'a>() {
        let (_, c) = constant(Span::new(Ascii::from(&b"THIS_IS_CONSTANT = 42"[..]))).unwrap();
        let expected = assert_eq!(
            c,
            Constant {
                name: Ascii::from(&b"THIS_IS_CONSTANT"[..]),
                literal: Literal {
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
        assert_eq!(c.name, Ascii::from(&b"THIS_IS_CONSTANT"[..]));
        assert_eq!(c.literal.value, Value::Integer(42));
    }

    #[test]
    fn it_parses_constant_definitions() {
        let (_, program) = nani(Ascii::from(
            &br#"
THIS_IS_CONSTANT = 42
THIS_ALSO_IS = 666

        "#[..],
        ))
        .unwrap();
        assert_eq!(
            program,
            Program {
                constants: vec![
                    Constant {
                        name: Ascii::from(&b"THIS_IS_CONSTANT"[..]),
                        literal: Literal {
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
                    Constant {
                        name: Ascii::from(&b"THIS_ALSO_IS"[..]),
                        literal: Literal {
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
                ]
            }
        );
    }
}
