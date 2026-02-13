use p3_field::Field;
use tracing::instrument;

use crate::air_builders::debug::USE_DEBUG_BUILDER;

mod batch_inverse;
pub use batch_inverse::*;

// Copied from valida-util
/// Calculates and returns the multiplicative inverses of each field element, with zero
/// values remaining unchanged.
#[instrument(name = "batch_multiplicative_inverse", level = "info", skip_all)]
pub fn batch_multiplicative_inverse_allowing_zero<F: Field>(values: Vec<F>) -> Vec<F> {
    // Check if values are zero, and construct a new vector with only nonzero values
    let mut nonzero_values = Vec::with_capacity(values.len());
    let mut indices = Vec::with_capacity(values.len());
    for (i, value) in values.iter().cloned().enumerate() {
        if value.is_zero() {
            continue;
        }
        nonzero_values.push(value);
        indices.push(i);
    }

    // Compute the multiplicative inverse of nonzero values
    let inverse_nonzero_values = p3_field::batch_multiplicative_inverse(&nonzero_values);

    // Reconstruct the original vector
    let mut result = values.clone();
    for (i, index) in indices.into_iter().enumerate() {
        result[index] = inverse_nonzero_values[i];
    }

    result
}

/// This utility function will parallelize an operation that is to be
/// performed over a mutable slice.
///
/// Assumes that slice length is a multiple of `chunk_size` and parallelization preserves the chunks
/// so each slice in a thread is still multiple of `chunk_size`.
///
/// The closure `f` takes `(thread_slice, idx)` where `thread_slice` is a sub-slice starting at
/// `v[idx]`.
// Copied and modified from https://github.com/axiom-crypto/halo2/blob/4e584896b62c981ec7c7dced4a9ca95b82306550/halo2_proofs/src/arithmetic.rs#L157
pub fn parallelize_chunks<T, F>(v: &mut [T], chunk_size: usize, f: F)
where
    T: Send,
    F: Fn(&mut [T], usize) + Send + Sync + Clone,
{
    debug_assert_eq!(v.len() % chunk_size, 0);
    #[cfg(not(feature = "parallel"))]
    {
        f(v, 0)
    }
    // Algorithm rationale:
    //
    // Using the stdlib `chunks_mut` will lead to severe load imbalance.
    // From https://github.com/rust-lang/rust/blob/e94bda3/library/core/src/slice/iter.rs#L1607-L1637
    // if the division is not exact, the last chunk will be the remainder.
    //
    // Dividing 40 items on 12 threads will lead to a chunk size of 40/12 = 3,
    // There will be a 13 chunks of size 3 and 1 of size 1 distributed on 12 threads.
    // This leads to 1 thread working on 6 iterations, 1 on 4 iterations and 10 on 3 iterations,
    // a load imbalance of 2x.
    //
    // Instead we can divide work into chunks of size
    // 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3 = 4*4 + 3*8 = 40
    //
    // This would lead to a 6/4 = 1.5x speedup compared to naive chunks_mut
    //
    // See also OpenMP spec (page 60)
    // http://www.openmp.org/mp-documents/openmp-4.5.pdf
    // "When no chunk_size is specified, the iteration space is divided into chunks
    // that are approximately equal in size, and at most one chunk is distributed to
    // each thread. The size of the chunks is unspecified in this case."
    // This implies chunks are the same size ±1
    #[cfg(feature = "parallel")]
    {
        let f = &f;
        let total_iters = v.len() / chunk_size;
        let num_threads = rayon::current_num_threads();

        let lo_slice_size = (total_iters / num_threads) * chunk_size;
        let hi_slice_size = lo_slice_size + chunk_size;
        let cutoff_thread_idx = total_iters % num_threads;
        let split_pos = cutoff_thread_idx * hi_slice_size;
        let (v_hi, v_lo) = v.split_at_mut(split_pos);

        rayon::scope(|scope| {
            // Skip special-case: number of iterations is cleanly divided by number of threads.
            if cutoff_thread_idx != 0 {
                for (chunk_id, chunk) in v_hi.chunks_exact_mut(hi_slice_size).enumerate() {
                    let offset = chunk_id * hi_slice_size;
                    scope.spawn(move |_| f(chunk, offset));
                }
            }
            // Skip special-case: less iterations than number of threads.
            if lo_slice_size != 0 {
                for (chunk_id, chunk) in v_lo.chunks_exact_mut(lo_slice_size).enumerate() {
                    let offset = split_pos + (chunk_id * lo_slice_size);
                    scope.spawn(move |_| f(chunk, offset));
                }
            }
        });
    }
}

/// Disables the debug builder so there are not debug assert panics.
/// Commonly used in negative tests to prevent panics.
pub fn disable_debug_builder() {
    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
}

#[macro_export]
#[cfg(feature = "parallel")]
macro_rules! parizip {
    ( $first:expr $( , $rest:expr )* $(,)* ) => {
        {
            use rayon::iter::*;
            ( $first $( , $rest)* ).into_par_iter()
        }
    };
}
#[macro_export]
#[cfg(not(feature = "parallel"))]
macro_rules! parizip {
    ( $first:expr $( , $rest:expr )* $(,)* ) => {
        itertools::izip!( $first $( , $rest)* )
    };
}
