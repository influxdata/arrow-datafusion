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

use std::{borrow::Borrow, fmt::Debug, sync::Arc};

use arrow::datatypes::SchemaRef;
use datafusion_common::{stats::Precision, ColumnStatistics, Statistics};
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use itertools::Itertools;

pub(super) fn partition_count(plan: &dyn ExecutionPlan) -> usize {
    plan.output_partitioning().partition_count()
}

pub(super) fn pretty_fmt_fields(schema: &SchemaRef) -> String {
    schema.fields().iter().map(|field| field.name()).join(",")
}

pub(super) fn make_column_statistics_inexact(
    column_statistics: Vec<ColumnStatistics>,
) -> Vec<ColumnStatistics> {
    column_statistics
        .into_iter()
        .map(
            |ColumnStatistics {
                 null_count,
                 min_value,
                 max_value,
                 sum_value,
                 distinct_count,
             }| {
                ColumnStatistics {
                    null_count: make_inexact_or_keep_absent(null_count),
                    min_value: make_inexact_or_keep_absent(min_value),
                    max_value: make_inexact_or_keep_absent(max_value),
                    sum_value: make_inexact_or_keep_absent(sum_value),
                    distinct_count: make_inexact_or_keep_absent(distinct_count),
                }
            },
        )
        .collect_vec()
}

fn make_inexact_or_keep_absent<T: Debug + Clone + Eq + PartialOrd>(
    val: Precision<T>,
) -> Precision<T> {
    match val {
        Precision::Exact(val) => Precision::Inexact(val),
        _ => val,
    }
}

/// Merge a collection of [`Statistics`] into a single stat.
///
/// This takes statistics references, which may or may not be arc'ed.
pub(super) fn merge_stats_collection<
    T: Borrow<Statistics> + Clone + Into<Arc<Statistics>>,
>(
    mut stats: impl Iterator<Item = T>,
    target_schema: &SchemaRef,
) -> Statistics {
    let Some(start) = stats.next() else {
        // stats is empty
        return Statistics::new_unknown(target_schema);
    };
    stats.fold(start.borrow().clone(), |a, b| merge_stats(a, b.borrow()))
}

/// Merge together two [`Statistics`].
fn merge_stats(a: Statistics, b: &Statistics) -> Statistics {
    assert_eq!(
        a.column_statistics.len(),
        b.column_statistics.len(),
        "failed to merge statistics, due to different column schema"
    );

    Statistics {
        num_rows: a.num_rows.add(&b.num_rows),
        total_byte_size: a.total_byte_size.add(&b.total_byte_size),
        column_statistics: a
            .column_statistics
            .into_iter()
            .zip(b.column_statistics.iter())
            .map(|(a, b)| merge_col_stats(a, b))
            .collect_vec(),
    }
}

fn merge_col_stats(a: ColumnStatistics, b: &ColumnStatistics) -> ColumnStatistics {
    ColumnStatistics {
        null_count: a.null_count.add(&b.null_count),
        min_value: a.min_value.min(&b.min_value),
        max_value: a.max_value.max(&b.max_value),
        sum_value: a.sum_value.add(&b.sum_value),
        distinct_count: Precision::Absent,
    }
}
