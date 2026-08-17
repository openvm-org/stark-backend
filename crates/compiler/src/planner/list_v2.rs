use std::collections::{HashMap, HashSet};

use crate::{
    graph_ir::BufId,
    planner::{
        ctx::PlanCtx,
        plan::{StreamInstr, StreamMemoryPlan},
        AbstractTimingGraph, PlanError,
    },
};

struct MemoryInterferenceGraph {
    interf: HashMap<BufId, Vec<BufId>>,
}

#[derive(Clone)]
struct ScheduleState<'a> {
    cur_mem_used: usize,
    max_peak_mem: usize,
    cur_max_time: f64,
    buf_users: HashMap<BufId, HashSet<usize>>, // buf to nodes
    stream_of: Vec<usize>,                     // stream of node
    stream_end_t: Vec<f64>,
    node_requires: Vec<HashSet<BufId>>,
    ready_queue: HashSet<usize>, // ready set of nodes to be sceduled
    node_start_times: Vec<f64>,  // of the nodes that are already scheduled, ranges can overlap if
    // on different streams
    g: &'a AbstractTimingGraph,
}

impl<'a> ScheduleState<'a> {
    fn new(g: &'a AbstractTimingGraph, streams: usize) -> Self {
        ScheduleState {
            cur_mem_used: 0,
            max_peak_mem: 0,
            cur_max_time: 0.0,
            buf_users: g
                .buf_users
                .iter()
                .map(|(bid, users)| (*bid, users.iter().cloned().collect()))
                .collect(),
            stream_of: vec![0; g.num_nodes],
            stream_end_t: vec![0.0; streams],
            node_requires: g
                .node_consumes
                .iter()
                .map(|bids| bids.iter().cloned().collect())
                .collect(),
            ready_queue: g.inital_ready_nodes.iter().cloned().collect(),
            node_start_times: vec![0.0; g.num_nodes],
            g,
        }
    }

    fn empty(g: &'a AbstractTimingGraph) -> Self {
        todo!()
    }

    fn make_interference_graph(&self) -> MemoryInterferenceGraph {
        let mut interf = HashMap::new();

        todo!();

        MemoryInterferenceGraph { interf }
    }

    fn make_schedule(mut self) -> StreamMemoryPlan {
        todo!()
    }

    /// the cost of putting node on stream
    fn cost(&self, node: usize, on_stream: usize, max_memory_bound: usize) -> f64 {
        let mut max_producer_end_t = 0.0;
        for bid in self.g.node_consumes[node].iter() {
            for n in &self.g.buf_producers[bid] {
                let end_t = self.node_start_times[*n] + self.g.node_times[*n];
                if end_t > max_producer_end_t {
                    max_producer_end_t = end_t;
                }
            }
        }
        let time_cost = max_producer_end_t.max(self.stream_end_t[on_stream]) - self.cur_max_time;

        let mut killed_bytes = 0;
        let mut produced_bytes = 0;

        for bid in self.g.node_consumes[node].iter() {
            if self.buf_users[bid].len() == 1 && self.buf_users[bid].contains(&node) {
                killed_bytes += self.g.buf_info[bid.0].concrete_size;
            }
        }
        for bid in self.g.node_produces[node].iter() {
            produced_bytes += self.g.buf_info[bid.0].concrete_size;
        }

        let memory_cost = produced_bytes as f64 - killed_bytes as f64;

        if self.cur_mem_used + produced_bytes >= max_memory_bound {
            f64::INFINITY
        } else {
            memory_cost / 1024.0 + time_cost
        }
    }

    fn put_on(&mut self, node: usize, on_stream: usize) {
        self.stream_of[node] = on_stream;

        let mut max_producer_end_t = 0.0;
        for bid in self.g.node_consumes[node].iter() {
            for n in &self.g.buf_producers[bid] {
                let end_t = self.node_start_times[*n] + self.g.node_times[*n];
                if end_t > max_producer_end_t {
                    max_producer_end_t = end_t;
                }
            }
        }

        self.stream_end_t[on_stream] =
            max_producer_end_t.max(self.stream_end_t[on_stream]) + self.g.node_times[node];

        let mut killed_bytes = 0;
        let mut produced_bytes = 0;

        for bid in self.g.node_consumes[node].iter() {
            if self.buf_users[bid].len() == 1 && self.buf_users[bid].contains(&node) {
                killed_bytes += self.g.buf_info[bid.0].concrete_size;
            }
            self.buf_users.remove(bid);
        }
        for bid in self.g.node_produces[node].iter() {
            produced_bytes += self.g.buf_info[bid.0].concrete_size;
            for n in self.g.buf_users[bid].iter() {
                self.node_requires[*n].remove(bid);
                if self.node_requires[*n].len() == 0 {
                    self.ready_queue.insert(*n);
                }
            }
        }
        self.cur_mem_used += produced_bytes;
        self.cur_mem_used -= killed_bytes;
        self.max_peak_mem = self.max_peak_mem.max(self.cur_mem_used);

        self.cur_max_time = self.cur_max_time.max(self.stream_end_t[on_stream]);
        self.ready_queue.remove(&node);
    }

    fn schedule(mut self, max_memory_bound: usize, k: usize) -> Vec<Self> {
        let mut costs = vec![];
        for node in self.ready_queue.iter() {
            let (min_stream, cost) = (0..self.stream_end_t.len())
                .map(|s| (s, self.cost(*node, s, max_memory_bound)))
                .min_by(|(_, c1), (_, c2)| c1.total_cmp(c2))
                .unwrap();
            costs.push((*node, min_stream, cost));
        }

        costs.sort_by(|(_, _, c1), (_, _, c2)| c1.total_cmp(c2));
        let mut schedules = vec![];
        for (i, (node, on_stream, c)) in costs[0..k].iter().enumerate() {
            if *c == f64::INFINITY {
                continue;
            }
            let mut new_schedule = if i == k - 1 {
                let mut sched = Self::empty(self.g);

                std::mem::swap(&mut self, &mut sched);
                sched
            } else {
                self.clone()
            };
            new_schedule.put_on(*node, *on_stream);
            schedules.push(new_schedule);
        }

        schedules
    }

    fn done(&self) -> bool {
        self.ready_queue.is_empty()
    }

    fn cur_cost(&self) -> f64 {
        self.cur_max_time + self.cur_mem_used as f64 / 1024.0
    }
}

pub fn plan_v2(
    g: &AbstractTimingGraph,
    num_streams: usize,
    max_memory_bound: usize,
    num_beams: usize,
) -> Result<StreamMemoryPlan, PlanError> {
    let mut finished_beams = vec![];

    let mut beams = vec![ScheduleState::new(g, num_streams); num_beams];

    let mut next_beams = vec![];

    while finished_beams.len() < num_beams {
        for b in beams.drain(..) {
            for new_sched in b.schedule(max_memory_bound, num_streams) {
                if new_sched.done() {
                    finished_beams.push(new_sched);
                } else {
                    next_beams.push(new_sched);
                }
            }
        }
        next_beams.sort_by(|b1, b2| b1.cur_cost().total_cmp(&b2.cur_cost()));
        next_beams.drain(num_beams..);
        std::mem::swap(&mut next_beams, &mut beams);
    }

    todo!()
}
