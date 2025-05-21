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

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::Statistics;
use datafusion_physical_expr::{
    analyze, intervals::utils::check_support, AnalysisContext,
};
use datafusion_physical_plan::{filter::collect_new_statistics, PhysicalExpr};

/// Calculates [`Statistics`] by applying selectivity (either default, or estimated) to input statistics.
///
/// This is a (slightly) modified from a private function in datafusion. Refer to:
/// <https://github.com/apache/datafusion/blob/07a310fea8d287805d1490e9dc3c1b7b1c2775d8/datafusion/physical-plan/src/filter.rs#L175-L216>
pub(super) fn apply_filter(
    input_stats: Statistics,
    input_schema: &SchemaRef,
    predicate: &Arc<dyn PhysicalExpr>,
    default_selectivity: u8,
) -> datafusion_common::Result<Statistics> {
    if !check_support(predicate, input_schema) {
        let selectivity = default_selectivity as f64 / 100.0;
        let mut stats = input_stats.to_inexact();
        stats.num_rows = stats.num_rows.with_estimated_selectivity(selectivity);
        stats.total_byte_size = stats
            .total_byte_size
            .with_estimated_selectivity(selectivity);
        return Ok(stats);
    }

    let num_rows = input_stats.num_rows;
    let total_byte_size = input_stats.total_byte_size;
    let input_analysis_ctx = AnalysisContext::try_from_statistics(
        input_schema,
        &input_stats.column_statistics,
    )?;

    let analysis_ctx = analyze(predicate, input_analysis_ctx, input_schema)?;

    // Estimate (inexact) selectivity of predicate
    let selectivity = analysis_ctx.selectivity.unwrap_or(1.0);
    let num_rows = num_rows.with_estimated_selectivity(selectivity);
    let total_byte_size = total_byte_size.with_estimated_selectivity(selectivity);

    let column_statistics =
        collect_new_statistics(&input_stats.column_statistics, analysis_ctx.boundaries);
    Ok(Statistics {
        num_rows,
        total_byte_size,
        column_statistics,
    })
}
