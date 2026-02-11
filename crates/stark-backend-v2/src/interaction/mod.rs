use std::fmt::Debug;

use p3_air::AirBuilder;
use p3_challenger::CanObserve;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_util::log2_ceil_usize;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::air_builders::symbolic::symbolic_expression::SymbolicExpression;

/// Interaction debugging tools
pub mod debug;
mod utils;

// Must be a type smaller than u32 to make BusIndex p - 1 unrepresentable.
pub type BusIndex = u16;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Interaction<Expr> {
    pub message: Vec<Expr>,
    pub count: Expr,
    /// The bus index specifying the bus to send the message over. All valid instantiations of
    /// `BusIndex` are safe.
    pub bus_index: BusIndex,
    /// Determines the contribution of each interaction message to a linear constraint on the trace
    /// heights in the verifier.
    ///
    /// For each bus index and trace, `count_weight` values are summed per interaction on that
    /// bus index and multiplied by the trace height. The total sum over all traces is constrained
    /// by the verifier to not overflow the field characteristic \( p \).
    ///
    /// This is used to impose sufficient conditions for bus constraint soundness and setting a
    /// proper value depends on the bus and the constraint it imposes.
    pub count_weight: u32,
}

pub type SymbolicInteraction<F> = Interaction<SymbolicExpression<F>>;

/// An [AirBuilder] with additional functionality to build special logUp arguments for
/// communication between AIRs across buses. These arguments use randomness to
/// add additional trace columns (in the extension field) and constraints to the AIR.
///
/// An interactive AIR is a AIR that can specify buses for sending and receiving data
/// to other AIRs. The original AIR is augmented by virtual columns determined by
/// the interactions to define a [RAP](crate::rap::Rap).
pub trait InteractionBuilder: AirBuilder<F: Field, Var: Copy> {
    /// Stores a new interaction in the builder.
    ///
    /// See [Interaction] for more details on `count_weight`.
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        bus_index: BusIndex,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
        count_weight: u32,
    );

    /// Returns the current number of interactions.
    fn num_interactions(&self) -> usize;

    /// Returns all interactions stored.
    fn all_interactions(&self) -> &[Interaction<Self::Expr>];

    // This used to be in Plonky3 but is no longer there. We preserve the
    // implementation here for downstream callers.
    fn assert_tern(&mut self, x: impl Into<Self::Expr>) {
        let x = x.into();
        self.assert_zero(x.clone() * (x.clone() - Self::Expr::ONE) * (x - Self::Expr::TWO));
    }
}

/// A `Lookup` bus is used to establish that one multiset of values (the queries) are subset of
/// another multiset of values (the keys).
///
/// Soundness requires that the total number of queries sent over the bus per message is at most the
/// field characteristic.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LookupBus {
    pub index: BusIndex,
}

impl LookupBus {
    pub const fn new(index: BusIndex) -> Self {
        Self { index }
    }

    /// Performs a lookup on the given bus.
    ///
    /// This method asserts that `key` is present in the lookup table. The parameter `enabled`
    /// must be constrained to be boolean, and the lookup constraint is imposed provided `enabled`
    /// is one.
    ///
    /// Caller must constrain that `enabled` is boolean.
    pub fn lookup_key<AB, E>(
        &self,
        builder: &mut AB,
        query: impl IntoIterator<Item = E>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        E: Into<AB::Expr>,
    {
        // We embed the query multiplicity as {0, 1} in the integers and the lookup table key
        // multiplicity to be {0, -1, ..., -p + 1}. Setting `count_weight = 1` will ensure that the
        // total number of lookups is at most p, which is sufficient to establish lookup multiset is
        // a subset of the key multiset. See Corollary 3.6 in
        // [docs/Soundess_of_Interactions_via_LogUp.pdf].
        builder.push_interaction(self.index, query, enabled, 1);
    }

    /// Adds a key to the lookup table.
    ///
    /// The `num_lookups` parameter should equal the number of enabled lookups performed.
    pub fn add_key_with_lookups<AB, E>(
        &self,
        builder: &mut AB,
        key: impl IntoIterator<Item = E>,
        num_lookups: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        E: Into<AB::Expr>,
    {
        // Since we only want a subset constraint, `count_weight` can be zero here. See the comment
        // in `LookupBus::lookup_key`.
        builder.push_interaction(self.index, key, -num_lookups.into(), 0);
    }
}

/// A `PermutationCheckBus` bus is used to establish that two multi-sets of values are equal.
///
/// Soundness requires that both the total number of messages sent and received over the bus per
/// message is at most the field characteristic.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PermutationCheckBus {
    pub index: BusIndex,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PermutationInteractionType {
    Send,
    Receive,
}

impl PermutationCheckBus {
    pub const fn new(index: BusIndex) -> Self {
        Self { index }
    }

    /// Send a message.
    ///
    /// Caller must constrain `enabled` to be boolean.
    pub fn send<AB, E>(
        &self,
        builder: &mut AB,
        message: impl IntoIterator<Item = E>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        E: Into<AB::Expr>,
    {
        // We embed the multiplicity `enabled` as an integer {0, 1}.
        builder.push_interaction(self.index, message, enabled, 1);
    }

    /// Receive a message.
    ///
    /// Caller must constrain `enabled` to be boolean.
    pub fn receive<AB, E>(
        &self,
        builder: &mut AB,
        message: impl IntoIterator<Item = E>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        E: Into<AB::Expr>,
    {
        // We embed the multiplicity `enabled` as an integer {0, -1}.
        builder.push_interaction(self.index, message, -enabled.into(), 1);
    }

    /// Send or receive determined by `interaction_type`.
    ///
    /// Caller must constrain `enabled` to be boolean.
    pub fn send_or_receive<AB, E>(
        &self,
        builder: &mut AB,
        interaction_type: PermutationInteractionType,
        message: impl IntoIterator<Item = E>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        E: Into<AB::Expr>,
    {
        match interaction_type {
            PermutationInteractionType::Send => self.send(builder, message, enabled),
            PermutationInteractionType::Receive => self.receive(builder, message, enabled),
        }
    }

    /// Send or receive a message determined by the expression `direction`.
    ///
    /// Direction = 1 means send, direction = -1 means receive, and direction = 0 means disabled.
    ///
    /// Caller must constrain that direction is in {-1, 0, 1}.
    pub fn interact<AB, E>(
        &self,
        builder: &mut AB,
        message: impl IntoIterator<Item = E>,
        direction: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        E: Into<AB::Expr>,
    {
        // We embed the multiplicity `direction` as an integer {-1, 0, 1}.
        builder.push_interaction(self.index, message, direction.into(), 1);
    }
}

/// Parameters to ensure sufficient soundness of the LogUp part of the protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct LogUpSecurityParameters {
    /// A bound on the total number of interactions.
    /// Determines a constraint at keygen that is checked by the verifier.
    pub max_interaction_count: u32,
    /// A bound on the base-2 logarithm of the length of the longest interaction. Checked in
    /// keygen.
    pub log_max_message_length: u32,
    /// The number of proof-of-work bits for the LogUp proof-of-work phase.
    pub pow_bits: usize,
}

impl LogUpSecurityParameters {
    /// The number of bits of security with grinding.
    pub fn bits_of_security<F: Field>(&self) -> u32 {
        // See Section 4 of [docs/Soundness_of_Interactions_via_LogUp.pdf].
        let log_order = u32::try_from(F::order().bits() - 1).unwrap();
        log_order
            - log2_ceil_usize(2 * self.max_interaction_count as usize) as u32  // multiply by two to account for the poles as well
            - self.log_max_message_length
            + u32::try_from(self.pow_bits).unwrap()
    }
    pub fn max_message_length(&self) -> usize {
        2usize
            .checked_pow(self.log_max_message_length)
            .expect("max_message_length overflowed usize")
    }
}
