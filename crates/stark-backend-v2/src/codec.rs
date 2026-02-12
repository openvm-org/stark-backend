use std::{
    array::from_fn,
    io::{self, Cursor, Read, Result, Write},
};

pub use codec_derive::{Decode, Encode};
use p3_field::{BasedVectorSpace, PrimeField32};

use crate::StarkProtocolConfig;

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

/// [StarkProtocolConfig] that has encodable associated types.
/// This is a separate trait to avoid Rust's orphan rule.
pub trait EncodableConfig: StarkProtocolConfig {
    fn encode_base_field<W: Write>(val: &Self::F, writer: &mut W) -> Result<()>;

    fn encode_extension_field<W: Write>(val: &Self::EF, writer: &mut W) -> Result<()>;

    fn encode_digest<W: Write>(val: &Self::Digest, writer: &mut W) -> Result<()>;

    /// Encode each element from an iterator (no length prefix).
    fn encode_base_field_iter<'a, W: Write>(
        iter: impl Iterator<Item = &'a Self::F>,
        writer: &mut W,
    ) -> Result<()>
    where
        Self::F: 'a,
    {
        for val in iter {
            Self::encode_base_field(val, writer)?;
        }
        Ok(())
    }

    /// Encode each element from an iterator (no length prefix).
    fn encode_extension_field_iter<'a, W: Write>(
        iter: impl Iterator<Item = &'a Self::EF>,
        writer: &mut W,
    ) -> Result<()>
    where
        Self::EF: 'a,
    {
        for val in iter {
            Self::encode_extension_field(val, writer)?;
        }
        Ok(())
    }

    /// Encode each element from an iterator (no length prefix).
    fn encode_digest_iter<'a, W: Write>(
        iter: impl Iterator<Item = &'a Self::Digest>,
        writer: &mut W,
    ) -> Result<()>
    where
        Self::Digest: 'a,
    {
        for val in iter {
            Self::encode_digest(val, writer)?;
        }
        Ok(())
    }

    /// Encode length-prefixed slice of base field elements.
    fn encode_base_field_slice<W: Write>(vals: &[Self::F], writer: &mut W) -> Result<()> {
        vals.len().encode(writer)?;
        Self::encode_base_field_iter(vals.iter(), writer)
    }

    /// Encode length-prefixed slice of extension field elements.
    fn encode_extension_field_slice<W: Write>(vals: &[Self::EF], writer: &mut W) -> Result<()> {
        vals.len().encode(writer)?;
        Self::encode_extension_field_iter(vals.iter(), writer)
    }

    /// Encode length-prefixed slice of digests.
    fn encode_digest_slice<W: Write>(vals: &[Self::Digest], writer: &mut W) -> Result<()> {
        vals.len().encode(writer)?;
        Self::encode_digest_iter(vals.iter(), writer)
    }
}

/// [StarkProtocolConfig] that has decodable associated types.
/// This is a separate trait to avoid Rust's orphan rule.
pub trait DecodableConfig: StarkProtocolConfig {
    fn decode_base_field<R: Read>(reader: &mut R) -> Result<Self::F>;

    fn decode_extension_field<R: Read>(reader: &mut R) -> Result<Self::EF>;

    fn decode_digest<R: Read>(reader: &mut R) -> Result<Self::Digest>;

    /// Decode `n` base field elements (known length, no length prefix).
    fn decode_base_field_n<R: Read>(reader: &mut R, n: usize) -> Result<Vec<Self::F>> {
        let mut vec = Vec::with_capacity(n);
        for _ in 0..n {
            vec.push(Self::decode_base_field(reader)?);
        }
        Ok(vec)
    }

    /// Decode `n` extension field elements (known length, no length prefix).
    fn decode_extension_field_n<R: Read>(reader: &mut R, n: usize) -> Result<Vec<Self::EF>> {
        let mut vec = Vec::with_capacity(n);
        for _ in 0..n {
            vec.push(Self::decode_extension_field(reader)?);
        }
        Ok(vec)
    }

    /// Decode `n` digests (known length, no length prefix).
    fn decode_digest_n<R: Read>(reader: &mut R, n: usize) -> Result<Vec<Self::Digest>> {
        let mut vec = Vec::with_capacity(n);
        for _ in 0..n {
            vec.push(Self::decode_digest(reader)?);
        }
        Ok(vec)
    }

    /// Decode a length-prefixed vector of base field elements.
    fn decode_base_field_vec<R: Read>(reader: &mut R) -> Result<Vec<Self::F>> {
        let len = usize::decode(reader)?;
        Self::decode_base_field_n(reader, len)
    }

    /// Decode a length-prefixed vector of extension field elements.
    fn decode_extension_field_vec<R: Read>(reader: &mut R) -> Result<Vec<Self::EF>> {
        let len = usize::decode(reader)?;
        Self::decode_extension_field_n(reader, len)
    }

    /// Decode a length-prefixed vector of digests.
    fn decode_digest_vec<R: Read>(reader: &mut R) -> Result<Vec<Self::Digest>> {
        let len = usize::decode(reader)?;
        Self::decode_digest_n(reader, len)
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

// ==================== Generic field codec helpers ====================

/// Encode a `PrimeField32` element as 4 little-endian bytes of its canonical u32 value.
pub fn encode_prime_field32<F: PrimeField32, W: Write>(val: &F, writer: &mut W) -> Result<()> {
    writer.write_all(&val.as_canonical_u32().to_le_bytes())
}

/// Decode a `PrimeField32` element from 4 little-endian bytes.
pub fn decode_prime_field32<F: PrimeField32, R: Read>(reader: &mut R) -> Result<F> {
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

/// Encode an extension field element by encoding each basis coefficient.
pub fn encode_extension_field32<F: PrimeField32, EF: BasedVectorSpace<F>, W: Write>(
    val: &EF,
    writer: &mut W,
) -> Result<()> {
    let base_slice: &[F] = val.as_basis_coefficients_slice();
    for v in base_slice {
        encode_prime_field32(v, writer)?;
    }
    Ok(())
}

/// Decode an extension field element by decoding each basis coefficient.
pub fn decode_extension_field32<F: PrimeField32, EF: BasedVectorSpace<F>, R: Read>(
    reader: &mut R,
) -> Result<EF> {
    let d = <EF as BasedVectorSpace<F>>::DIMENSION;
    let mut base_vec = Vec::with_capacity(d);
    for _ in 0..d {
        base_vec.push(decode_prime_field32(reader)?);
    }
    EF::from_basis_coefficients_slice(&base_vec)
        .ok_or(io::Error::other("from_basis_coefficients_slice failed"))
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
