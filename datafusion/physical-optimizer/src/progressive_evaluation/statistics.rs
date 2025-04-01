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

use datafusion_common::stats::Precision;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion, TreeNodeVisitor};
use datafusion_common::{
    internal_datafusion_err, Result as DatafusionResult, ScalarValue,
};
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_physical_plan::aggregates::AggregateExec;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::joins::{
    CrossJoinExec, HashJoinExec, NestedLoopJoinExec, SymmetricHashJoinExec,
};
use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::ColumnStatistics;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};

/// Similar to our in-house `StatisticsVisitor`, except it performs the statistics extraction per partition,
/// and not on the entire node.
#[derive(Debug)]
pub struct PartitionStatisticsVisitor<'a> {
    column_name: &'a str,
    partition: usize,
    statistics: Option<ColumnStatistics>,
}

impl<'a> PartitionStatisticsVisitor<'a> {
    pub fn new(column_name: &'a str, partition: usize) -> Self {
        Self {
            column_name,
            partition,
            statistics: Some(ColumnStatistics::new_unknown()),
        }
    }

    /// Consumes self, returning the found [`ColumnStatistics`].
    ///
    /// Returning None means the [`PartitionStatisticsVisitor`] could not extract the stats.
    pub fn result(&mut self) -> Option<ColumnStatistics> {
        std::mem::take(&mut self.statistics)
    }

    /// Merge stats.
    fn merge_stats<'i>(
        mut stats: impl Iterator<Item = Option<&'i ColumnStatistics>>,
    ) -> Option<ColumnStatistics> {
        let Some(start) = stats.next() else {
            // stats is empty
            return None;
        };
        stats.fold(start.cloned(), |a, b| {
            if let (Some(a), Some(b)) = (a, b) {
                Some(merge_stats(a, b))
            } else {
                None // we hit something that prevented stats extraction
            }
        })
    }

    /// Find partition within multiple children, and return the partition stats.
    fn find_stats_per_partition_within_multiple_children(
        &mut self,
        node: &Arc<dyn ExecutionPlan>,
    ) -> DatafusionResult<Option<ColumnStatistics>> {
        // figure out which input of the union to use
        let mut child_partition = self.partition;
        for child in node.children() {
            let num_child_partitions = partition_count(child);
            if child_partition < num_child_partitions {
                self.partition = child_partition;
                child.visit(self)?;
                return Ok(self.result());
            }
            child_partition -= num_child_partitions;
        }
        unreachable!("didn't find the partition in children");
    }

    /// Applied per node, merge all statistics across partitions in node.
    fn merge_column_statistics_across_partitions(
        &mut self,
        node: &Arc<dyn ExecutionPlan>,
    ) -> DatafusionResult<Option<ColumnStatistics>> {
        let cnt_partitions = partition_count_children(node);
        let mut extracted_stats = Vec::with_capacity(cnt_partitions);

        for child in node.children() {
            for partition in 0..partition_count(child) {
                let mut extractor = Self::new(self.column_name, partition);
                child.visit(&mut extractor)?;
                extracted_stats.push(extractor.result());
            }
        }

        Ok(Self::merge_stats(
            extracted_stats.iter().map(|a| a.as_ref()),
        ))
    }

    /// Extract column statistics from file_group of parquet exec.
    ///
    /// This is required since we need the statistics per partition.
    ///
    /// This is a hack until upstream code is in place.
    fn extract_statistics_from_file_group(
        &self,
        datasrc_exec: &DataSourceExec,
    ) -> DatafusionResult<Option<ColumnStatistics>> {
        let Ok(col_idx) = datasrc_exec.schema().index_of(self.column_name) else {
            // maybe alias
            return Ok(None);
        };

        // Extract partitioned files from the ParquetExec
        let Some(base_config) = datasrc_exec
            .data_source()
            .as_any()
            .downcast_ref::<FileScanConfig>()
        else {
            return Ok(None);
        };

        let file_group_partition = base_config.file_groups.get(self.partition).ok_or(
            internal_datafusion_err!(
                "requested partition does not exist in datasource exec"
            ),
        )?;

        Ok(file_group_partition
            .statistics()
            .as_ref()
            .and_then(|stats| stats.column_statistics.get(col_idx).cloned()))
    }

    /// Extract from datasource, with consideration of partitioning.
    fn extract_from_data_source(
        &mut self,
        datasrc: &Arc<dyn ExecutionPlan>,
    ) -> DatafusionResult<Option<ColumnStatistics>> {
        if datasrc.as_any().downcast_ref::<EmptyExec>().is_some()
            || datasrc
                .as_any()
                .downcast_ref::<PlaceholderRowExec>()
                .is_some()
        {
            Ok(self.statistics.clone())
        } else if let Some(datasrc_exec) =
            datasrc.as_any().downcast_ref::<DataSourceExec>()
        {
            let datarsc_stats =
                match self.extract_statistics_from_file_group(datasrc_exec) {
                    Ok(Some(col_stats)) => Some(convert_min_max_to_exact(col_stats)),
                    _ => None,
                };
            Ok(datarsc_stats)
        } else {
            // specific constraint on our plans
            unreachable!("should not have another unsupported data source")
        }
    }
}

impl<'n> TreeNodeVisitor<'n> for PartitionStatisticsVisitor<'_> {
    type Node = Arc<dyn ExecutionPlan>;

    fn f_down(&mut self, node: &'n Self::Node) -> DatafusionResult<TreeNodeRecursion> {
        if !is_supported(node) {
            self.statistics = None;
            Ok(TreeNodeRecursion::Stop)
        } else if is_leaf(node) {
            self.statistics = self.extract_from_data_source(node)?;
            Ok(TreeNodeRecursion::Stop)
        } else if should_merge_partition_statistics(node) {
            self.statistics = self.merge_column_statistics_across_partitions(node)?;
            Ok(TreeNodeRecursion::Jump)
        } else if should_pass_thru_partition_statistics(node) {
            self.statistics =
                self.find_stats_per_partition_within_multiple_children(node)?;
            Ok(TreeNodeRecursion::Stop)
        } else {
            self.statistics = None;
            Ok(TreeNodeRecursion::Stop)
        }
    }
}

fn is_supported(plan: &Arc<dyn ExecutionPlan>) -> bool {
    !(plan.as_any().downcast_ref::<HashJoinExec>().is_some()
        || plan.as_any().downcast_ref::<SymmetricHashJoinExec>().is_some()
        || plan.as_any().downcast_ref::<CrossJoinExec>().is_some()
        || plan.as_any().downcast_ref::<NestedLoopJoinExec>().is_some()
        || plan.as_any().downcast_ref::<AggregateExec>().is_some() // aggregate values can be different than min/max
        || plan.as_any().downcast_ref::<RepartitionExec>().is_some())
}

/// Has multiple input partitions, and 1 output partition. Merge stats.
///
/// This is a heuristic (based on our plans) and contingent on [`is_supported`].
fn should_merge_partition_statistics(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let sort_preserving_partitioning =
        if let Some(sort_exec) = plan.as_any().downcast_ref::<SortExec>() {
            sort_exec.preserve_partitioning()
        } else {
            false
        };
    partition_count(plan) == 1
        && has_multiple_child_partitions(plan)
        && !sort_preserving_partitioning
}

/// The input partitions and output partitions are expected to have unchanged statistics.
///
/// This is a heuristic (based on our plans) and contingent on [`is_supported`].
fn should_pass_thru_partition_statistics(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.children().len() == 1 || partition_count(plan) == partition_count_children(plan)
}

fn is_leaf(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.children().is_empty()
}

fn has_multiple_child_partitions(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.children().len() > 1 || partition_count_children(plan) > 1
}

fn partition_count_children(plan: &Arc<dyn ExecutionPlan>) -> usize {
    plan.children()
        .iter()
        .map(|child| partition_count(child))
        .sum::<usize>()
}

fn partition_count(plan: &Arc<dyn ExecutionPlan>) -> usize {
    plan.output_partitioning().partition_count()
}

fn merge_stats(a: ColumnStatistics, b: &ColumnStatistics) -> ColumnStatistics {
    ColumnStatistics {
        null_count: a.null_count.add(&b.null_count),
        min_value: a.min_value.min(&b.min_value),
        max_value: a.max_value.max(&b.max_value),
        distinct_count: Precision::Absent,
        sum_value: Precision::Absent,
    }
}

/// DataFusion statistics can't distinguish "inexact" from "known to be within the range"
///
/// Converts inexact values to exact if possible for the purposes of
/// analysis by [`PartitionStatisticsVisitor`]
fn convert_min_max_to_exact(mut column_statistics: ColumnStatistics) -> ColumnStatistics {
    column_statistics.min_value = to_exact(column_statistics.min_value);
    column_statistics.max_value = to_exact(column_statistics.max_value);
    column_statistics
}

/// Convert [`Precision::Inexact`] to [`Precision::Exact`] if possible
fn to_exact(v: Precision<ScalarValue>) -> Precision<ScalarValue> {
    match v {
        Precision::Exact(v) | Precision::Inexact(v) => Precision::Exact(v),
        Precision::Absent => Precision::Absent,
    }
}

/// Return min max of a ColumnStatistics with precise values
pub fn column_statistics_min_max(
    column_statistics: ColumnStatistics,
) -> Option<(ScalarValue, ScalarValue)> {
    match (column_statistics.min_value, column_statistics.max_value) {
        (Precision::Exact(min), Precision::Exact(max)) => Some((min, max)),
        // the statistics values are absent or imprecise
        _ => None,
    }
}
