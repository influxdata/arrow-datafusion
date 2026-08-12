// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Benchmark for `UnionExec` construction cost as a function of child count.
//!
//! Scenarios:
//! - `shared_arc`: every child returns the same `Arc<Schema>`
//! - `content_equal`: pointer-distinct but identical schemas per child
//! - `last_differs`: identical except the last child (worst case for any
//!   equality fast path: the scan is wasted, then the full merge runs)

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::union::UnionExec;
use datafusion_physical_plan::ExecutionPlan;

const NUM_FIELDS: usize = 10;
const METADATA_PER_FIELD: usize = 2;

fn base_schema() -> Schema {
    let fields: Vec<Field> = (0..NUM_FIELDS)
        .map(|i| {
            let mut md = HashMap::new();
            for m in 0..METADATA_PER_FIELD {
                md.insert(format!("key_{m}"), format!("value_{i}_{m}"));
            }
            Field::new(format!("col_{i}"), DataType::Int64, true).with_metadata(md)
        })
        .collect();
    Schema::new(fields)
}

fn children_shared_arc(n: usize) -> Vec<Arc<dyn ExecutionPlan>> {
    let schema: SchemaRef = Arc::new(base_schema());
    (0..n)
        .map(|_| Arc::new(EmptyExec::new(Arc::clone(&schema))) as Arc<dyn ExecutionPlan>)
        .collect()
}

fn children_content_equal(n: usize) -> Vec<Arc<dyn ExecutionPlan>> {
    let schema = base_schema();
    (0..n)
        .map(|_| {
            Arc::new(EmptyExec::new(Arc::new(schema.clone()))) as Arc<dyn ExecutionPlan>
        })
        .collect()
}

fn children_last_differs(n: usize) -> Vec<Arc<dyn ExecutionPlan>> {
    let mut children = children_content_equal(n - 1);
    let schema = base_schema();
    let mut fields: Vec<Field> =
        schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    let last = fields.pop().unwrap();
    let mut md = last.metadata().clone();
    md.insert("divergent".to_string(), "true".to_string());
    fields.push(last.with_metadata(md));
    children.push(Arc::new(EmptyExec::new(Arc::new(Schema::new(fields)))) as _);
    children
}

fn bench_union_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("union_exec_try_new");
    for n in [100, 1000, 4000] {
        let shared = children_shared_arc(n);
        let content = children_content_equal(n);
        let differs = children_last_differs(n);

        group.bench_with_input(BenchmarkId::new("shared_arc", n), &shared, |b, ch| {
            b.iter(|| UnionExec::try_new(ch.clone()).unwrap())
        });
        group.bench_with_input(
            BenchmarkId::new("content_equal", n),
            &content,
            |b, ch| b.iter(|| UnionExec::try_new(ch.clone()).unwrap()),
        );
        group.bench_with_input(BenchmarkId::new("last_differs", n), &differs, |b, ch| {
            b.iter(|| UnionExec::try_new(ch.clone()).unwrap())
        });
    }
    group.finish();

    // Reconstruct the union once per child, as optimizer rewrites do.
    let mut group = c.benchmark_group("union_exec_rebuild_per_child");
    for n in [100, 1000] {
        let children = children_content_equal(n);
        let union: Arc<dyn ExecutionPlan> = UnionExec::try_new(children.clone()).unwrap();
        group.bench_with_input(BenchmarkId::new("content_equal", n), &n, |b, _| {
            b.iter(|| {
                let mut plan = Arc::clone(&union);
                for _ in 0..n {
                    plan = plan.with_new_children(children.clone()).unwrap();
                }
                plan
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_union_construction);
criterion_main!(benches);
