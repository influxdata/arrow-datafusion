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
        group = group.with_statistics(Arc::unwrap_or_clone(stats.to_owned()))
    }
    group
}
