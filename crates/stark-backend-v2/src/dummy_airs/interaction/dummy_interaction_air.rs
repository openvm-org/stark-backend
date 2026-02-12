//! Air with columns
//! | count | fields[..] |
//!
//! Chip will either send or receive the fields with multiplicity count.
//! The main Air has no constraints, the only constraints are specified by the Chip trait

use std::{iter, sync::Arc};

use itertools::izip;
use p3_air::{Air, BaseAir, BaseAirWithPublicValues};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    air_builders::PartitionedAirBuilder,
    interaction::{BusIndex, InteractionBuilder},
    prover::{
        stacked_pcs::stacked_commit, AirProvingContextV2, ColMajorMatrix, CommittedTraceDataV2,
        CpuBackendV2,
    },
    AirRef, ChipV2, PartitionedBaseAir, StarkProtocolConfig, SystemParams,
};

pub struct DummyInteractionCols;
impl DummyInteractionCols {
    pub fn count_col() -> usize {
        0
    }
    pub fn field_col(field_idx: usize) -> usize {
        field_idx + 1
    }
}

#[derive(Clone, Copy)]
pub struct DummyInteractionAir {
    field_width: usize,
    /// Send if true. Receive if false.
    pub is_send: bool,
    bus_index: BusIndex,
    pub count_weight: u32,
    /// If true, then | count | and | fields[..] | are in separate main trace partitions.
    pub partition: bool,
}

impl DummyInteractionAir {
    pub fn new(field_width: usize, is_send: bool, bus_index: BusIndex) -> Self {
        Self {
            field_width,
            is_send,
            bus_index,
            count_weight: 0,
            partition: false,
        }
    }

    pub fn partition(self) -> Self {
        Self {
            partition: true,
            ..self
        }
    }

    pub fn field_width(&self) -> usize {
        self.field_width
    }
}

impl<F> BaseAirWithPublicValues<F> for DummyInteractionAir {}
impl<F> PartitionedBaseAir<F> for DummyInteractionAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        if self.partition {
            vec![self.field_width]
        } else {
            vec![]
        }
    }
    fn common_main_width(&self) -> usize {
        if self.partition {
            1
        } else {
            1 + self.field_width
        }
    }
}
impl<F> BaseAir<F> for DummyInteractionAir {
    fn width(&self) -> usize {
        1 + self.field_width
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }
}

impl<AB: InteractionBuilder + PartitionedAirBuilder> Air<AB> for DummyInteractionAir {
    fn eval(&self, builder: &mut AB) {
        let (fields, count) = if self.partition {
            let local_0 = builder.common_main().row_slice(0).unwrap();
            let local_1 = builder.cached_mains()[0].row_slice(0).unwrap();
            let count = local_0[0];
            let fields = local_1.to_vec();
            (fields, count)
        } else {
            let main = builder.main();
            let local = main.row_slice(0).expect("window should have two elements");
            let count = local[DummyInteractionCols::count_col()];
            let fields: Vec<_> = (0..self.field_width)
                .map(|i| local[DummyInteractionCols::field_col(i)])
                .collect();
            (fields, count)
        };
        if self.is_send {
            builder.push_interaction(self.bus_index, fields, count, self.count_weight);
        } else {
            builder.push_interaction(
                self.bus_index,
                fields,
                AB::Expr::NEG_ONE * count,
                self.count_weight,
            );
        }
    }
}

/// Note: in principle, committing cached trace is out of scope of a chip. But this chip is for
/// usually testing, so we support it for convenience.
pub struct DummyInteractionChip {
    params: Option<SystemParams>,
    data: Option<DummyInteractionData>,
    pub air: DummyInteractionAir,
}

#[derive(Debug, Clone)]
pub struct DummyInteractionData {
    pub count: Vec<u32>,
    pub fields: Vec<Vec<u32>>,
}

impl DummyInteractionChip {
    pub fn new_without_partition(field_width: usize, is_send: bool, bus_index: BusIndex) -> Self {
        let air = DummyInteractionAir::new(field_width, is_send, bus_index);
        Self {
            params: None,
            data: None,
            air,
        }
    }
    pub fn new_with_partition(
        params: SystemParams,
        field_width: usize,
        is_send: bool,
        bus_index: BusIndex,
    ) -> Self {
        let air = DummyInteractionAir::new(field_width, is_send, bus_index).partition();
        Self {
            params: Some(params),
            data: None,
            air,
        }
    }
    pub fn load_data(&mut self, data: DummyInteractionData) {
        let DummyInteractionData { count, fields } = &data;
        let h = count.len();
        assert_eq!(fields.len(), h);
        let w = fields[0].len();
        assert_eq!(self.air.field_width, w);
        assert!(fields.iter().all(|r| r.len() == w));
        self.data = Some(data);
    }
    pub fn air<SC: StarkProtocolConfig>(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }
}

impl<SC: StarkProtocolConfig> ChipV2<(), CpuBackendV2<SC>> for DummyInteractionChip
where
    SC::F: PrimeCharacteristicRing,
{
    fn generate_proving_ctx(&self, _: ()) -> AirProvingContextV2<CpuBackendV2<SC>> {
        assert!(self.data.is_some());
        let data = self.data.clone().unwrap();
        if self.air.partition {
            self.generate_traces_with_partition::<SC>(data)
        } else {
            let trace = self.generate_traces_without_partition::<SC>(data);
            AirProvingContextV2::simple_no_pis(ColMajorMatrix::from_row_major(&trace))
        }
    }
}

impl DummyInteractionChip {
    fn generate_traces_with_partition<SC: StarkProtocolConfig>(
        &self,
        data: DummyInteractionData,
    ) -> AirProvingContextV2<CpuBackendV2<SC>>
    where
        SC::F: PrimeCharacteristicRing,
    {
        let DummyInteractionData {
            mut count,
            mut fields,
        } = data;
        let h = count.len();
        assert_eq!(fields.len(), h);
        let w = fields[0].len();
        assert_eq!(self.air.field_width, w);
        assert!(fields.iter().all(|r| r.len() == w));
        let h = h.next_power_of_two();
        count.resize(h, 0);
        fields.resize(h, vec![0; w]);
        let common_main_val: Vec<_> = count.into_iter().map(SC::F::from_u32).collect();
        let cached_trace_val: Vec<_> = fields
            .into_iter()
            .flatten()
            .map(SC::F::from_u32)
            .collect();
        let cached_trace_rm = RowMajorMatrix::new(cached_trace_val, w);
        let cached_trace = ColMajorMatrix::from_row_major(&cached_trace_rm);

        let params = self.params.as_ref().expect("params required for partition");
        let (commit, data) = stacked_commit::<SC::H>(
            params.l_skip,
            params.n_stack,
            params.log_blowup,
            params.k_whir(),
            &[&cached_trace],
        );

        let common_main_rm = RowMajorMatrix::new(common_main_val, 1);
        AirProvingContextV2 {
            cached_mains: vec![CommittedTraceDataV2 {
                commitment: commit,
                trace: cached_trace,
                data: Arc::new(data),
            }],
            common_main: ColMajorMatrix::from_row_major(&common_main_rm),
            public_values: vec![],
        }
    }

    fn generate_traces_without_partition<SC: StarkProtocolConfig>(
        &self,
        data: DummyInteractionData,
    ) -> RowMajorMatrix<SC::F>
    where
        SC::F: PrimeCharacteristicRing,
    {
        let DummyInteractionData { count, fields } = data;
        let h = count.len();
        assert_eq!(fields.len(), h);
        let w = fields[0].len();
        assert_eq!(self.air.field_width, w);
        assert!(fields.iter().all(|r| r.len() == w));
        let common_main_val: Vec<_> = izip!(count, fields)
            .flat_map(|(count, fields)| iter::once(count).chain(fields))
            .chain(iter::repeat(0))
            .take((w + 1) * h.next_power_of_two())
            .map(SC::F::from_u32)
            .collect();
        RowMajorMatrix::new(common_main_val, w + 1)
    }
}
