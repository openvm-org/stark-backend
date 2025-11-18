# Cross-Stream Allocation/Free Handling Approaches

## Problem
When memory is allocated on stream S1 but freed on stream S2, CUDA's async allocator can't efficiently reuse the memory because it's stuck in S1's per-stream memory pool. This can cause OOM errors even when plenty of memory is available.

## Implemented Approach: Event Synchronization

**Current Implementation (Approach 1)**

```rust
// Free on current stream
cudaFreeAsync(ptr, cudaStreamPerThread)?;

// Record event on current stream after free
let event = CudaEvent::new()?;
event.record_on_this()?;

// Wait for event to complete
default_stream_wait(&event)?;
```

**Behavior:**
- ✅ Ensures free operation completes before returning
- ✅ Maintains async behavior (non-blocking until wait)
- ✅ Proper ordering: free completes before subsequent operations
- ⚠️ Doesn't solve pool fragmentation - memory still stuck in original stream's pool
- ⚠️ Can't free on original stream (we're on different thread)

**When to use:** General case - ensures correctness but doesn't solve pool fragmentation.

---

## Alternative Approach 2: cudaMemPoolTrimTo

**Implementation:**
```rust
// Free on current stream
cudaFreeAsync(ptr, cudaStreamPerThread)?;

// Get default memory pool
let mut device = 0;
cudaGetDevice(&mut device)?;
let mut pool: *mut c_void = std::ptr::null_mut();
cudaDeviceGetDefaultMemPool(&mut pool, device)?;

// Trim pool to release memory (0 = release all possible)
cudaMemPoolTrimTo(pool, 0)?;
```

**Behavior:**
- ✅ Forces CUDA to release memory from pools back to OS
- ✅ Can help with pool fragmentation
- ✅ Non-blocking operation
- ❌ **Doesn't ensure synchronization** - memory might still be in use by original stream
- ❌ **Use-after-free risk** if original stream hasn't finished
- ❌ May release memory that's still needed by other allocations

**When to use:** Only if you're certain the original stream has finished using the memory AND you're experiencing severe pool fragmentation. **Dangerous without proper synchronization.**

---

## Alternative Approach 3: Stream Synchronization

**Implementation:**
```rust
// Free on current stream
cudaFreeAsync(ptr, cudaStreamPerThread)?;

// Block until current stream completes (including free)
current_stream_sync()?;
```

**Behavior:**
- ✅ Ensures free completes before returning
- ✅ Simple and safe
- ✅ Guarantees proper ordering
- ❌ **Blocks host thread** - defeats async benefits
- ❌ Doesn't solve pool fragmentation
- ❌ Performance impact - synchronous operation

**When to use:** When you need maximum safety and don't care about async performance, or during cleanup/shutdown.

---

## Comparison Table

| Approach | Async | Safety | Fragmentation Fix | Performance Impact |
|----------|-------|--------|-------------------|-------------------|
| **1. Event Sync** (Current) | ✅ Yes | ✅ Safe | ❌ No | Low |
| **2. Trim Pool** | ✅ Yes | ⚠️ Risky* | ✅ Yes | Low |
| **3. Stream Sync** | ❌ No | ✅ Safe | ❌ No | High |

*Risky because it doesn't ensure original stream has finished using memory.

---

## Recommendation

**Use Approach 1 (Event Synchronization)** as the default:
- It's safe and maintains async behavior
- It ensures proper ordering
- It detects and logs cross-stream issues for debugging

**Consider Approach 2 (Trim Pool)** only if:
- You're experiencing severe OOM due to pool fragmentation
- You can guarantee the original stream has finished (via other synchronization)
- You're willing to accept the risk

**Use Approach 3 (Stream Sync)** only:
- During cleanup/shutdown
- When async performance doesn't matter
- For debugging purposes

---

## Root Cause Fix

The real solution is to **ensure allocation and free happen on the same stream**. This implementation detects mismatches and handles them gracefully, but the best fix is to prevent them in the first place by:
1. Using explicit `CudaStream` objects instead of `cudaStreamPerThread`
2. Passing the same stream handle for both malloc and free
3. Ensuring the same OS thread handles both operations

