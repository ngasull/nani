use std::convert::TryFrom;

#[derive(Clone, Copy, PartialEq)]
pub struct Ascii<'a> {
    pub bytes: &'a [u8],
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
    pub byte: u8,
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
        if is_ascii(byte) {
            Ok(AsciiChar { byte })
        } else {
            Err("Given byte is not ASCII")
        }
    }
}

#[inline]
pub fn is_lower(chr: u8) -> bool {
    chr >= 0x61 && chr <= 0x7A
}

#[inline]
pub fn is_upper(chr: u8) -> bool {
    chr >= 0x41 && chr <= 0x5A
}

#[inline]
pub fn is_ascii(chr: u8) -> bool {
    chr < 0x80
}
