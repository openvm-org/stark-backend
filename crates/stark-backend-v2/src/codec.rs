use std::{
    array::from_fn,
    io::{self, Cursor, Read, Result, Write},
};

pub use codec_derive::{Decode, Encode};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32};

use crate::baby_bear_poseidon2::{D_EF, EF, F};

/// Hardware and language independent encoding.
/// Uses the Writer pattern for more efficient encoding without intermediate buffers.
// @dev Trait just for implementation sanity
pub trait Encode {
    /// Writes the encoded representation of `self` to the given writer.
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()>;

    /// Convenience method to encode into a `Vec<u8>`
    fn encode_to_vec(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        self.encode(&mut buffer)?;
        Ok(buffer)
    }
}

/// Hardware and language independent decoding.
/// Uses the Reader pattern for efficient decoding.
pub trait Decode: Sized {
    /// Reads and decodes a value from the given reader.
    fn decode<R: Read>(reader: &mut R) -> Result<Self>;
    fn decode_from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = Cursor::new(bytes);
        Self::decode(&mut reader)
    }
}

// ==================== Encode implementations for basic types ====================

impl Encode for bool {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&[*self as u8])?;
        Ok(())
    }
}

impl Encode for u8 {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&[*self])
    }
}

impl Encode for u32 {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.to_le_bytes())
    }
}

impl Encode for usize {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        let x: u32 = (*self).try_into().map_err(io::Error::other)?;
        x.encode(writer)
    }
}

impl Encode for F {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.as_canonical_u32().to_le_bytes())
    }
}

impl Encode for EF {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        let base_slice: &[F] = self.as_basis_coefficients_slice();
        // Fixed length slice, so don't encode length
        for val in base_slice {
            val.encode(writer)?;
        }
        Ok(())
    }
}

// ==================== Encode helpers ====================

/// Encodes length of slice and then each element
pub fn encode_slice<T: Encode, W: Write>(slice: &[T], writer: &mut W) -> Result<()> {
    slice.len().encode(writer)?;
    for elt in slice {
        elt.encode(writer)?;
    }
    Ok(())
}

/// Encodes each element (no length)
pub fn encode_iter<'a, T: Encode + 'a, W: Write>(
    iter: impl Iterator<Item = &'a T>,
    writer: &mut W,
) -> Result<()> {
    for elt in iter {
        elt.encode(writer)?;
    }
    Ok(())
}

impl<T: Encode, const N: usize> Encode for [T; N] {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        for val in self {
            val.encode(writer)?;
        }
        Ok(())
    }
}

impl<T: Encode> Encode for Vec<T> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        encode_slice(self, writer)
    }
}

impl<S: Encode, T: Encode> Encode for (S, T) {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.0.encode(writer)?;
        self.1.encode(writer)
    }
}

impl<T: Encode> Encode for Option<T> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.is_some().encode(writer)?;
        if let Some(val) = self {
            val.encode(writer)?;
        }
        Ok(())
    }
}

// ==================== Decode implementations for basic types ====================

impl Decode for bool {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let mut bytes = [0u8; 1];
        reader.read_exact(&mut bytes)?;
        Ok(bytes[0] != 0)
    }
}

impl Decode for u8 {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let mut bytes = [0u8; 1];
        reader.read_exact(&mut bytes)?;
        Ok(bytes[0])
    }
}

impl Decode for u32 {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }
}

impl Decode for usize {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let val = u32::decode(reader)?;
        Ok(val as usize)
    }
}

impl Decode for F {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        let value = u32::from_le_bytes(bytes);
        if value < F::ORDER_U32 {
            Ok(F::from_u32(value))
        } else {
            Err(io::Error::other(format!(
                "Attempted read of {} into F >= F::ORDER_U32 {}",
                value,
                F::ORDER_U32
            )))
        }
    }
}

impl Decode for EF {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let mut base_slice = [F::ZERO; D_EF];
        for val in &mut base_slice {
            *val = F::decode(reader)?;
        }
        Ok(EF::from_basis_coefficients_fn(|i| base_slice[i]))
    }
}

// ==================== Decode helpers ====================

/// Decodes into a vector given preset length
pub fn decode_into_vec<T: Decode, R: Read>(reader: &mut R, len: usize) -> Result<Vec<T>> {
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(T::decode(reader)?);
    }
    Ok(vec)
}

impl<T: Decode + Default, const N: usize> Decode for [T; N] {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let mut result = from_fn(|_| T::default());
        for val in &mut result {
            *val = T::decode(reader)?;
        }
        Ok(result)
    }
}

impl<T: Decode> Decode for Vec<T> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let len = usize::decode(reader)?;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(T::decode(reader)?);
        }
        Ok(vec)
    }
}

impl<S: Decode, T: Decode> Decode for (S, T) {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        Ok((S::decode(reader)?, T::decode(reader)?))
    }
}

impl<T: Decode> Decode for Option<T> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let is_some = bool::decode(reader)?;
        if is_some {
            Ok(Some(T::decode(reader)?))
        } else {
            Ok(None)
        }
    }
}
