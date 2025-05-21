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

use datafusion_common::{
    error::Result,
    tree_node::{Transformed, TreeNodeRecursion},
};
use datafusion_datasource::{
    file_groups::FileGroup, file_scan_config::FileScanConfig, source::DataSourceExec,
    PartitionedFile,
};
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_plan::{sorts::sort::SortExec, ExecutionPlan};

use itertools::Itertools;

use crate::progressive_evaluation::extract_ranges::extract_ranges_from_files;

// This is an optimization for the most recent value query `ORDER BY time DESC/ASC LIMIT n`
// See: https://github.com/influxdata/influxdb_iox/issues/12205
// The observation is recent data is mostly in first file, so the plan should avoid reading the others unless necessary
//
/// This function is to split all files in the same ParquetExec into different groups/DF partitions and
/// set the `preserve_partitioning` so they will be executed sequentially. The files will later be re-ordered
/// (if non-overlapping) by lexical range.
pub fn split_parquet_files(
    plan: Arc<dyn ExecutionPlan>,
    ordering_req: &LexOrdering,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    if let Some(sort_exec) = plan.as_any().downcast_ref::<SortExec>() {
        if !sort_exec
            .properties()
            .equivalence_properties()
            .ordering_satisfy(ordering_req)
        {
            // halt on DAG branch
            Ok(Transformed::new(plan, false, TreeNodeRecursion::Stop))
        } else {
            // continue down
            Ok(Transformed::no(plan))
        }
    } else if let Some(parquet_exec) = plan.as_any().downcast_ref::<DataSourceExec>() {
        if let Some(transformed) =
            transform_parquet_exec_single_file_each_group(parquet_exec, ordering_req)?
        {
            Ok(Transformed::yes(transformed))
        } else {
            Ok(Transformed::no(plan))
        }
    } else {
        Ok(Transformed::no(plan))
    }
}

/// Transform a ParquetExec with N files in various groupings,
/// into a ParquetExec into N groups each include one file.
///
/// The function is only called when the plan does not include DeduplicateExec and includes only one ParquetExec.
///
/// This function will return error if
///   - There are no statsitics for the given column (including the when the column is missing from the file
///     and produce null values that leads to absent statistics)
///   - Some files overlap (the min/max time ranges are disjoint)
///   - There is a DeduplicateExec in the plan which means the data of the plan overlaps
///
/// The output ParquetExec's are ordered such that the file with the most recent time ranges is read first
///
/// For example
/// ```text
/// ParquetExec(groups=[[file1,file2], [file3]])
/// ```
/// Is rewritten so each file is in its own group and the files are ordered by time range
/// ```text
/// ParquetExec(groups=[[file1], [file2], [file3]])
/// ```
fn transform_parquet_exec_single_file_each_group(
    datasrc_exec: &DataSourceExec,
    ordering_req: &LexOrdering,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    if datasrc_exec
        .properties()
        .output_partitioning()
        .partition_count()
        == 1
    {
        return Ok(None);
    }

    // Extract partitioned files from the ParquetExec
    let Some(base_config) = datasrc_exec
        .data_source()
        .as_any()
        .downcast_ref::<FileScanConfig>()
    else {
        return Ok(None);
    };
    let files = base_config
        .file_groups
        .iter()
        .flat_map(|group| group.files())
        .collect_vec();
    let schema = Arc::clone(&base_config.file_schema);

    // extract disjoint lexical ranges
    // if cannot find, then is not disjoint
    let Some(lexical_ranges) = extract_ranges_from_files(ordering_req, &files, schema)?
    else {
        return Ok(None);
    };

    // reorder partitioned files by lexical indices
    let indices = lexical_ranges.indices();
    assert_eq!(
        indices.len(),
        files.len(),
        "should have every file listed in the sorted indices"
    );
    let mut new_partitioned_file_groups = files
        .into_iter()
        .enumerate()
        .map(|(file_idx, file)| {
            (
                indices
                    .iter()
                    .position(|sorted_idx| *sorted_idx == file_idx)
                    .expect("file should be listed in indices"),
                file,
            )
        })
        .collect::<Vec<_>>();
    new_partitioned_file_groups.sort_by_key(|(idx, _)| *idx);

    // create new file grouping
    let new_partitioned_file_groups = new_partitioned_file_groups
        .into_iter()
        .map(|(_, file)| {
            // each file group has 1 file
            build_file_group_with_stats(file)
        })
        .collect_vec();

    // Assigned new partitioned file groups to the new base config
    let mut new_base_config = base_config.clone();
    new_base_config.file_groups = new_partitioned_file_groups;

    Ok(Some(DataSourceExec::from_data_source(new_base_config)))
}

fn build_file_group_with_stats(file: &PartitionedFile) -> FileGroup {
    let mut group = FileGroup::new(vec![file.clone()]);
    if let Some(stats) = &file.statistics {
        group = group.with_statistics(Arc::clone(stats))
    }
    group
}

#[cfg(test)]
pub(crate) mod test_utils {
    use arrow::datatypes::DataType;
    use arrow::datatypes::Field;
    use arrow::datatypes::SchemaRef;
    use datafusion_common::ScalarValue;
    use datafusion_common::{stats::Precision, Statistics};
    use datafusion_datasource::file_scan_config::FileScanConfig;
    use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
    use datafusion_datasource::source::DataSourceExec;
    use datafusion_datasource::PartitionedFile;
    use datafusion_datasource_parquet::source::ParquetSource;
    use datafusion_execution::object_store::ObjectStoreUrl;
    use datafusion_physical_expr::LexOrdering;
    use datafusion_physical_plan::{
        coalesce_batches::CoalesceBatchesExec,
        filter::FilterExec,
        joins::CrossJoinExec,
        limit::LocalLimitExec,
        projection::ProjectionExec,
        repartition::RepartitionExec,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
        union::UnionExec,
        Partitioning, PhysicalExpr,
    };
    use datafusion_physical_plan::{ColumnStatistics, ExecutionPlan};

    use itertools::Itertools;

    use std::{
        fmt::{self, Display, Formatter},
        sync::Arc,
    };

    /// Return a schema with a single column `a` of type int64.
    pub fn single_column_schema() -> SchemaRef {
        Arc::new(arrow::datatypes::Schema::new(vec![Field::new(
            "a",
            DataType::Int64,
            true,
        )]))
    }

    #[derive(Debug, Copy, Clone)]
    pub struct SortKeyRange {
        pub min: Option<i32>,
        pub max: Option<i32>,
        pub null_count: usize,
    }

    impl From<SortKeyRange> for ColumnStatistics {
        fn from(val: SortKeyRange) -> Self {
            Self {
                null_count: Precision::Exact(val.null_count),
                max_value: Precision::Exact(ScalarValue::Int32(val.max)),
                min_value: Precision::Exact(ScalarValue::Int32(val.min)),
                sum_value: Precision::Absent,
                distinct_count: Precision::Absent,
            }
        }
    }

    impl Display for SortKeyRange {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            write!(f, "({:?})->({:?})", self.min, self.max)?;
            if self.null_count > 0 {
                write!(f, " null_count={}", self.null_count)?;
            }
            Ok(())
        }
    }

    /// Create a single parquet, with a given ordering and using the statistics from the [`SortKeyRange`]
    pub fn parquet_exec_with_sort_with_statistics(
        output_ordering: Vec<LexOrdering>,
        key_ranges: &[&SortKeyRange],
    ) -> Arc<dyn ExecutionPlan> {
        parquet_exec_with_sort_with_statistics_and_schema(
            &single_column_schema(),
            output_ordering,
            key_ranges,
        )
    }

    pub type RangeForMultipleColumns = Vec<SortKeyRange>; // vec![col0, col1, col2]
    pub struct PartitionedFilesAndRanges {
        pub per_file: Vec<RangeForMultipleColumns>,
    }

    /// Create a single parquet exec, with multiple parquet, with a given ordering and using the statistics from the [`SortKeyRange`].
    /// Assumes a single column schema.
    pub fn parquet_exec_with_sort_with_statistics_and_schema(
        schema: &SchemaRef,
        output_ordering: Vec<LexOrdering>,
        key_ranges_for_single_column_multiple_files: &[&SortKeyRange], // VecPerFile<KeyForSingleColumn>
    ) -> Arc<dyn ExecutionPlan> {
        let per_file_ranges = PartitionedFilesAndRanges {
            per_file: key_ranges_for_single_column_multiple_files
                .iter()
                .map(|single_col_range_per_file| vec![**single_col_range_per_file])
                .collect_vec(),
        };

        let file_scan_config = file_scan_config(schema, output_ordering, per_file_ranges);

        DataSourceExec::from_data_source(file_scan_config)
    }

    /// Create a file scan config with a given file [`SchemaRef`], ordering,
    /// and [`ColumnStatistics`] for multiple columns.
    pub fn file_scan_config(
        schema: &SchemaRef,
        output_ordering: Vec<LexOrdering>,
        multiple_column_key_ranges_per_file: PartitionedFilesAndRanges,
    ) -> FileScanConfig {
        let PartitionedFilesAndRanges { per_file } = multiple_column_key_ranges_per_file;
        let mut statistics = Statistics::new_unknown(schema);
        let mut file_groups = Vec::with_capacity(per_file.len());

        // cummulative statistics for the entire parquet exec, per sort key
        let num_sort_keys = per_file[0].len();
        let mut cum_null_count = vec![0; num_sort_keys];
        let mut cum_min = vec![None; num_sort_keys];
        let mut cum_max = vec![None; num_sort_keys];

        // iterate thru files, creating the PartitionedFile and the associated statistics
        for (file_idx, multiple_column_key_ranges_per_file) in
            per_file.into_iter().enumerate()
        {
            // gather stats for all columns
            let mut per_file_col_stats = Vec::with_capacity(num_sort_keys);
            for (col_idx, key_range) in
                multiple_column_key_ranges_per_file.into_iter().enumerate()
            {
                let SortKeyRange {
                    min,
                    max,
                    null_count,
                } = key_range;

                // update per file stats
                per_file_col_stats.push(ColumnStatistics {
                    null_count: Precision::Exact(null_count),
                    min_value: Precision::Exact(ScalarValue::Int32(min)),
                    max_value: Precision::Exact(ScalarValue::Int32(max)),
                    ..Default::default()
                });

                // update cummulative statistics for entire parquet exec
                cum_min[col_idx] = match (cum_min[col_idx], min) {
                    (None, x) => x,
                    (x, None) => x,
                    (Some(a), Some(b)) => Some(std::cmp::min(a, b)),
                };
                cum_max[col_idx] = match (cum_max[col_idx], max) {
                    (None, x) => x,
                    (x, None) => x,
                    (Some(a), Some(b)) => Some(std::cmp::max(a, b)),
                };
                cum_null_count[col_idx] += null_count;
            }

            // Create single file with statistics.
            let mut file = PartitionedFile::new(format!("{file_idx}.parquet"), 100);
            file.statistics = Some(Arc::new(Statistics {
                num_rows: Precision::Absent,
                total_byte_size: Precision::Absent,
                column_statistics: per_file_col_stats,
            }));
            file_groups.push(vec![file].into());
        }

        // add stats, for the whole parquet exec, for all columns
        for col_idx in 0..num_sort_keys {
            statistics.column_statistics[col_idx] = ColumnStatistics {
                null_count: Precision::Exact(cum_null_count[col_idx]),
                min_value: Precision::Exact(ScalarValue::Int32(cum_min[col_idx])),
                max_value: Precision::Exact(ScalarValue::Int32(cum_max[col_idx])),
                ..Default::default()
            };
        }

        FileScanConfigBuilder::new(
            ObjectStoreUrl::parse("test:///").unwrap(),
            Arc::clone(schema),
            Arc::new(ParquetSource::new(Default::default())),
        )
        .with_file_groups(file_groups)
        .with_output_ordering(output_ordering)
        .with_statistics(statistics)
        .build()
    }

    pub fn union_exec(input: Vec<Arc<dyn ExecutionPlan>>) -> Arc<dyn ExecutionPlan> {
        Arc::new(UnionExec::new(input))
    }

    pub fn sort_exec(
        sort_exprs: &LexOrdering,
        input: &Arc<dyn ExecutionPlan>,
        preserve_partitioning: bool,
    ) -> Arc<dyn ExecutionPlan> {
        let new_sort = SortExec::new(sort_exprs.clone(), Arc::clone(input))
            .with_preserve_partitioning(preserve_partitioning);
        Arc::new(new_sort)
    }

    pub fn coalesce_exec(input: &Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        Arc::new(CoalesceBatchesExec::new(Arc::clone(input), 10))
    }

    pub fn filter_exec(
        input: &Arc<dyn ExecutionPlan>,
        predicate: Arc<dyn PhysicalExpr>,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(FilterExec::try_new(predicate, Arc::clone(input)).unwrap())
    }

    pub fn limit_exec(
        input: &Arc<dyn ExecutionPlan>,
        fetch: usize,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(LocalLimitExec::new(Arc::clone(input), fetch))
    }

    pub fn proj_exec(
        input: &Arc<dyn ExecutionPlan>,
        projects: Vec<(Arc<dyn PhysicalExpr>, String)>,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(ProjectionExec::try_new(projects, Arc::clone(input)).unwrap())
    }

    pub fn spm_exec(
        input: &Arc<dyn ExecutionPlan>,
        sort_exprs: &LexOrdering,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(SortPreservingMergeExec::new(
            sort_exprs.clone(),
            Arc::clone(input),
        ))
    }

    pub fn repartition_exec(
        input: &Arc<dyn ExecutionPlan>,
        partitioning: Partitioning,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(RepartitionExec::try_new(Arc::clone(input), partitioning).unwrap())
    }

    pub fn crossjoin_exec(
        left: &Arc<dyn ExecutionPlan>,
        right: &Arc<dyn ExecutionPlan>,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(CrossJoinExec::new(Arc::clone(left), Arc::clone(right)))
    }
}
