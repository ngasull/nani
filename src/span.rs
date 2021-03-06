use nom::{
    error::{ErrorKind, ParseError},
    AsBytes, Compare, CompareResult, Err, FindSubstring, IResult, InputIter, InputLength,
    InputTake, InputTakeAtPosition, Offset, ParseTo, Slice,
};
use std::iter::{Enumerate, Map};
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};
use std::slice::Iter;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AstSpan {
    pub offset: usize,
    pub line: usize,
    pub col: usize,
    pub end_offset: usize,
    pub end_line: usize,
    pub end_col: usize,
}

impl AstSpan {
    pub fn to(&self, other: &AstSpan) -> AstSpan {
        let AstSpan {
            offset, line, col, ..
        } = *self;
        let AstSpan {
            end_offset,
            end_line,
            end_col,
            ..
        } = *other;
        AstSpan {
            offset,
            line,
            col,
            end_offset,
            end_line,
            end_col,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Span<'a> {
    pub offset: usize,
    pub line: usize,
    pub col: usize,
    pub fragment: &'a [u8],
}

impl<'a> Span<'a> {
    pub fn new(program: &'a [u8]) -> Span {
        Span {
            line: 1,
            col: 1,
            offset: 0,
            fragment: program,
        }
    }

    pub fn to(self, end: Self) -> AstSpan {
        AstSpan {
            offset: self.offset,
            line: self.line,
            col: self.col,
            end_offset: end.offset,
            end_line: end.line,
            end_col: end.col - 1,
        }
    }
}

impl<'a> PartialEq for Span<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.line == other.line && self.offset == other.offset && self.fragment == other.fragment
    }
}

impl<'a> Eq for Span<'a> {}

impl<'a> AsBytes for Span<'a> {
    fn as_bytes(&self) -> &[u8] {
        self.fragment
    }
}

impl<'a> InputLength for Span<'a> {
    fn input_len(&self) -> usize {
        self.fragment.len()
    }
}

impl<'a> InputTake for Span<'a>
where
    Self: Slice<RangeFrom<usize>> + Slice<RangeTo<usize>>,
{
    fn take(&self, count: usize) -> Self {
        self.slice(..count)
    }

    fn take_split(&self, count: usize) -> (Self, Self) {
        (self.slice(count..), self.slice(..count))
    }
}

impl<'a> InputTakeAtPosition for Span<'a>
where
    Self: Slice<RangeFrom<usize>> + Slice<RangeTo<usize>> + Clone,
{
    type Item = <&'a [u8] as InputIter>::Item;

    fn split_at_position_complete<P, E: ParseError<Self>>(
        &self,
        predicate: P,
    ) -> IResult<Self, Self, E>
    where
        P: Fn(Self::Item) -> bool,
    {
        match self.split_at_position(predicate) {
            Err(Err::Incomplete(_)) => Ok(self.take_split(self.input_len())),
            res => res,
        }
    }

    fn split_at_position<P, E: ParseError<Self>>(&self, predicate: P) -> IResult<Self, Self, E>
    where
        P: Fn(Self::Item) -> bool,
    {
        match self.fragment.position(predicate) {
            Some(n) => Ok(self.take_split(n)),
            None => Err(Err::Incomplete(nom::Needed::Size(1))),
        }
    }

    fn split_at_position1<P, E: ParseError<Self>>(
        &self,
        predicate: P,
        e: ErrorKind,
    ) -> IResult<Self, Self, E>
    where
        P: Fn(Self::Item) -> bool,
    {
        match self.fragment.position(predicate) {
            Some(0) => Err(Err::Error(E::from_error_kind(self.clone(), e))),
            Some(n) => Ok(self.take_split(n)),
            None => Err(Err::Incomplete(nom::Needed::Size(1))),
        }
    }

    fn split_at_position1_complete<P, E: ParseError<Self>>(
        &self,
        predicate: P,
        e: ErrorKind,
    ) -> IResult<Self, Self, E>
    where
        P: Fn(Self::Item) -> bool,
    {
        match self.fragment.position(predicate) {
            Some(0) => Err(Err::Error(E::from_error_kind(self.clone(), e))),
            Some(n) => Ok(self.take_split(n)),
            None => {
                if self.fragment.input_len() == 0 {
                    Err(Err::Error(E::from_error_kind(self.clone(), e)))
                } else {
                    Ok(self.take_split(self.input_len()))
                }
            }
        }
    }
}

impl<'a> InputIter for Span<'a> {
    type Item = u8;
    type Iter = Enumerate<Self::IterElem>;
    type IterElem = Map<Iter<'a, Self::Item>, fn(&u8) -> u8>;
    #[inline]
    fn iter_indices(&self) -> Self::Iter {
        self.fragment.iter_indices()
    }
    #[inline]
    fn iter_elements(&self) -> Self::IterElem {
        self.fragment.iter_elements()
    }
    #[inline]
    fn position<P>(&self, predicate: P) -> Option<usize>
    where
        P: Fn(Self::Item) -> bool,
    {
        self.fragment.position(predicate)
    }
    #[inline]
    fn slice_index(&self, count: usize) -> Option<usize> {
        self.fragment.slice_index(count)
    }
}

#[macro_export]
macro_rules! impl_compare {
    ( $compare_to_type:ty ) => {
        impl<'a, 'b> Compare<$compare_to_type> for Span<'a> {
            #[inline(always)]
            fn compare(&self, t: $compare_to_type) -> CompareResult {
                self.fragment.compare(t)
            }

            #[inline(always)]
            fn compare_no_case(&self, t: $compare_to_type) -> CompareResult {
                self.fragment.compare_no_case(t)
            }
        }
    };
}

impl_compare!(&'a [u8]);

impl<'a, 'b> Compare<Span<'b>> for Span<'a> {
    #[inline(always)]
    fn compare(&self, t: Span<'b>) -> CompareResult {
        self.fragment.compare(t.fragment)
    }

    #[inline(always)]
    fn compare_no_case(&self, t: Span<'b>) -> CompareResult {
        self.fragment.compare_no_case(t.fragment)
    }
}

#[macro_export]
macro_rules! impl_slice_range {
    ( $range_type:ty, $can_return_self:expr ) => {
        impl<'a> Slice<$range_type> for Span<'a> {
            fn slice(&self, range: $range_type) -> Self {
                if $can_return_self(&range) {
                    return self.clone();
                }
                let next_fragment = self.fragment.slice(range);
                let consumed_len = self.fragment.offset(&next_fragment);
                if consumed_len == 0 {
                    return Span {
                        fragment: next_fragment,
                        ..*self
                    };
                }

                let consumed = self.fragment.slice(..consumed_len);
                let (line, col) = consumed
                    .split(|&b| b == b'\n')
                    .fold((self.line - 1, self.col), |(l, _), bs| (l + 1, bs.len()));

                Span {
                    line,
                    col: if line == self.line {
                        self.col + col
                    } else {
                        col + 1
                    },
                    offset: self.offset + consumed_len,
                    fragment: next_fragment,
                }
            }
        }
    };
}

impl_slice_range! {Range<usize>, |_| false }
impl_slice_range! {RangeTo<usize>, |_| false }
impl_slice_range! {RangeFrom<usize>, |range:&RangeFrom<usize>| range.start == 0}
impl_slice_range! {RangeFull, |_| true}

impl<'a> FindSubstring<&'a str> for Span<'a> {
    #[inline]
    fn find_substring(&self, substr: &'a str) -> Option<usize> {
        self.fragment.find_substring(substr)
    }
}

impl<'a, R: FromStr> ParseTo<R> for Span<'a> {
    #[inline]
    fn parse_to(&self) -> Option<R> {
        self.fragment.parse_to()
    }
}

impl<'a> Offset for Span<'a> {
    fn offset(&self, second: &Self) -> usize {
        let fst = self.offset;
        let snd = second.offset;

        snd - fst
    }
}

pub fn position<'a, E>(s: Span<'a>) -> IResult<Span<'a>, Span<'a>, E>
where
    E: ParseError<Span<'a>>,
{
    nom::bytes::complete::take(0usize)(s)
}
