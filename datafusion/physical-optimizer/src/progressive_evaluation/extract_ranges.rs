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

//! Extract [`NonOverlappingOrderedLexicalRanges`] from different sources.

use arrow::compute::SortOptions;
use arrow::datatypes::Schema;
use datafusion_common::tree_node::TreeNode;
use datafusion_common::{ColumnStatistics, Result, ScalarValue};
use datafusion_datasource::PartitionedFile;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use std::sync::Arc;

use super::lexical_ranges::{LexicalRange, NonOverlappingOrderedLexicalRanges};
use super::statistics::{column_statistics_min_max, PartitionStatisticsVisitor};

/// Attempt to extract LexicalRanges for the given sort keys and input plan
///
/// Output will have N ranges where N is the number of output partitions
///
/// Returns None if not possible to determine ranges
pub fn extract_disjoint_ranges_from_plan(
    exprs: &LexOrdering,
    input_plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<NonOverlappingOrderedLexicalRanges>> {
    // if the ordering does not match, then we cannot confirm proper ranges
    if !input_plan
        .properties()
        .equivalence_properties()
        .ordering_satisfy(exprs)
    {
        return Ok(None);
    }

    let num_input_partitions = partition_count(input_plan);

    // one builder for each output partition
    let mut builders = vec![LexicalRange::builder(); num_input_partitions];
    for sort_expr in exprs.iter() {
        let Some(column) = sort_expr.expr.as_any().downcast_ref::<Column>() else {
            return Ok(None);
        };
        let col_name = column.name();

        for (partition, builder) in
            builders.iter_mut().enumerate().take(num_input_partitions)
        {
            let Some((min, max)) = min_max_for_partition(
                col_name,
                &sort_expr.options,
                partition,
                input_plan,
            )?
            else {
                return Ok(None);
            };
            builder.push(min, max);
        }
    }

    let sort_options = exprs.iter().map(|e| e.options).collect::<Vec<_>>();
    let ranges_per_partition = builders
        .into_iter()
        .map(|builder| builder.build())
        .collect::<Vec<_>>();

    NonOverlappingOrderedLexicalRanges::try_new(&sort_options, ranges_per_partition)
}

fn partition_count(plan: &Arc<dyn ExecutionPlan>) -> usize {
    plan.output_partitioning().partition_count()
}

/// Returns the min and max value for the specified partition.
///
/// Eventually this would be represented by the Statistics structure in
/// DataFusion.
fn min_max_for_partition(
    col_name: &str,
    sort_options: &SortOptions,
    partition: usize,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<(ScalarValue, ScalarValue)>> {
    // Check if the column is a constant value according to the equivalence properties (TODO)

    let mut extractor = PartitionStatisticsVisitor::new(col_name, partition);
    plan.visit(&mut extractor)?;
    let stats = extractor.result();

    if let Some(ColumnStatistics {
        min_value,
        max_value,
        null_count,
        ..
    }) = stats
    {
        let (Some(min), Some(max)) = (min_value.get_value(), max_value.get_value())
        else {
            return Ok(None);
        };

        let mut min = min.clone();
        let mut max = max.clone();
        if *null_count.get_value().unwrap_or(&0) > 0 {
            let nulls_as_min = !sort_options.descending && sort_options.nulls_first // ASC nulls first
                || sort_options.descending && !sort_options.nulls_first; // DESC nulls last

            // TODO(wiedld): use ScalarValue::Null after using type coersion in ConvertedRows?
            // For now, return None if fails to convert.
            let Some(null) = use_scalar_none(&min) else {
                return Ok(None);
            };

            if nulls_as_min {
                min = null;
            } else {
                max = null;
            }
        }

        Ok(Some((min, max)))
    } else {
        Ok(None)
    }
}

/// Attempt to extract LexicalRanges for the given sort keys and partitioned files
///
/// Output will have N ranges where N is the number of partitioned files
///
/// Returns None if not possible to determine disjoint ranges
pub fn extract_ranges_from_files(
    exprs: &LexOrdering,
    files: &Vec<&PartitionedFile>,
    schema: Arc<Schema>,
) -> Result<Option<NonOverlappingOrderedLexicalRanges>> {
    let num_input_partitions = files.len();

    // one builder for each output partition
    let mut builders = vec![LexicalRange::builder(); num_input_partitions];
    for sort_expr in exprs.iter() {
        let Some(column) = sort_expr.expr.as_any().downcast_ref::<Column>() else {
            return Ok(None);
        };
        let col_name = column.name();

        for (file, builder) in files.iter().zip(builders.iter_mut()) {
            let Some((min, max)) = min_max_for_partitioned_file(col_name, file, &schema)?
            else {
                return Ok(None);
            };
            builder.push(min, max);
        }
    }

    let sort_options = exprs.iter().map(|e| e.options).collect::<Vec<_>>();
    let ranges_per_partition = builders
        .into_iter()
        .map(|builder| builder.build())
        .collect::<Vec<_>>();

    NonOverlappingOrderedLexicalRanges::try_new(&sort_options, ranges_per_partition)
}

/// Returns the min and max value for the specified partitioned file.
///
/// Eventually this will not be required, since we will generalize the solution to not require DAG plan modification.
fn min_max_for_partitioned_file(
    col_name: &str,
    file: &PartitionedFile,
    schema: &Arc<Schema>,
) -> Result<Option<(ScalarValue, ScalarValue)>> {
    let (Some((col_idx, _)), Some(file_stats)) =
        (schema.fields().find(col_name), &file.statistics)
    else {
        return Ok(None);
    };
    let col_stats = file_stats.column_statistics[col_idx].clone();
    Ok(column_statistics_min_max(col_stats))
}

/// TODO: remove this function.
///
/// This is due to our [`ConvertedRows`] not yet handling type coersion of [`ScalarValue::Null`].
fn use_scalar_none(value: &ScalarValue) -> Option<ScalarValue> {
    match value {
        ScalarValue::Boolean(_) => Some(ScalarValue::Boolean(None)),
        ScalarValue::Float16(_) => Some(ScalarValue::Float16(None)),
        ScalarValue::Float32(_) => Some(ScalarValue::Float32(None)),
        ScalarValue::Float64(_) => Some(ScalarValue::Float64(None)),
        ScalarValue::Decimal128(_, a, b) => Some(ScalarValue::Decimal128(None, *a, *b)),
        ScalarValue::Decimal256(_, a, b) => Some(ScalarValue::Decimal256(None, *a, *b)),
        ScalarValue::Int8(_) => Some(ScalarValue::Int8(None)),
        ScalarValue::Int16(_) => Some(ScalarValue::Int16(None)),
        ScalarValue::Int32(_) => Some(ScalarValue::Int32(None)),
        ScalarValue::Int64(_) => Some(ScalarValue::Int64(None)),
        ScalarValue::UInt8(_) => Some(ScalarValue::UInt8(None)),
        ScalarValue::UInt16(_) => Some(ScalarValue::UInt16(None)),
        ScalarValue::UInt32(_) => Some(ScalarValue::UInt32(None)),
        ScalarValue::UInt64(_) => Some(ScalarValue::UInt64(None)),
        ScalarValue::Utf8(_) => Some(ScalarValue::Utf8(None)),
        ScalarValue::Utf8View(_) => Some(ScalarValue::Utf8View(None)),
        ScalarValue::LargeUtf8(_) => Some(ScalarValue::LargeUtf8(None)),
        ScalarValue::Binary(_) => Some(ScalarValue::Binary(None)),
        ScalarValue::BinaryView(_) => Some(ScalarValue::BinaryView(None)),
        ScalarValue::FixedSizeBinary(i, _) => {
            Some(ScalarValue::FixedSizeBinary(*i, None))
        }
        ScalarValue::LargeBinary(_) => Some(ScalarValue::LargeBinary(None)),
        ScalarValue::Date32(_) => Some(ScalarValue::Date32(None)),
        ScalarValue::Date64(_) => Some(ScalarValue::Date64(None)),
        ScalarValue::Time32Second(_) => Some(ScalarValue::Time32Second(None)),
        ScalarValue::Time32Millisecond(_) => Some(ScalarValue::Time32Millisecond(None)),
        ScalarValue::Time64Microsecond(_) => Some(ScalarValue::Time64Microsecond(None)),
        ScalarValue::Time64Nanosecond(_) => Some(ScalarValue::Time64Nanosecond(None)),
        ScalarValue::TimestampSecond(_, tz) => {
            Some(ScalarValue::TimestampSecond(None, tz.clone()))
        }
        ScalarValue::TimestampMillisecond(_, tz) => {
            Some(ScalarValue::TimestampMillisecond(None, tz.clone()))
        }
        ScalarValue::TimestampMicrosecond(_, tz) => {
            Some(ScalarValue::TimestampMicrosecond(None, tz.clone()))
        }
        ScalarValue::TimestampNanosecond(_, tz) => {
            Some(ScalarValue::TimestampNanosecond(None, tz.clone()))
        }
        ScalarValue::IntervalYearMonth(_) => Some(ScalarValue::IntervalYearMonth(None)),
        ScalarValue::IntervalDayTime(_) => Some(ScalarValue::IntervalDayTime(None)),
        ScalarValue::IntervalMonthDayNano(_) => {
            Some(ScalarValue::IntervalMonthDayNano(None))
        }
        ScalarValue::DurationSecond(_) => Some(ScalarValue::DurationSecond(None)),
        ScalarValue::DurationMillisecond(_) => {
            Some(ScalarValue::DurationMillisecond(None))
        }
        ScalarValue::DurationMicrosecond(_) => {
            Some(ScalarValue::DurationMicrosecond(None))
        }
        ScalarValue::DurationNanosecond(_) => Some(ScalarValue::DurationNanosecond(None)),

        // for now, don't support others.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::cmp;

    use super::*;

    use arrow::{
        compute::SortOptions,
        datatypes::{DataType, Field},
    };
    use datafusion_common::{stats::Precision, ColumnStatistics, Statistics};
    use datafusion_datasource::{
        file_groups::FileGroup, file_scan_config::FileScanConfigBuilder,
        source::DataSourceExec,
    };
    use datafusion_datasource_parquet::source::ParquetSource;
    use datafusion_execution::object_store::ObjectStoreUrl;
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
    use datafusion_physical_plan::{
        expressions::col, sorts::sort::SortExec, union::UnionExec,
    };
    use itertools::Itertools;

    struct KeyRange {
        min: Option<i32>,
        max: Option<i32>,
        null_count: usize,
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]))
    }

    /// Create a single parquet
    fn parquet_exec_with_sort_with_statistics(
        output_ordering: Vec<LexOrdering>,
        key_ranges: &[&KeyRange],
    ) -> Arc<dyn ExecutionPlan> {
        let mut statistics = Statistics::new_unknown(&schema());
        let mut file_groups = Vec::with_capacity(key_ranges.len());

        let mut cum_null_count = 0;
        let mut cum_min = None;
        let mut cum_max = None;
        for key_range in key_ranges {
            let KeyRange {
                min,
                max,
                null_count,
            } = key_range;

            // update cummulative statistics for entire parquet exec
            cum_min = match (cum_min, min) {
                (None, x) => *x,
                (x, None) => x,
                (Some(a), Some(b)) => Some(cmp::min(a, *b)),
            };
            cum_max = match (cum_max, max) {
                (None, x) => *x,
                (x, None) => x,
                (Some(a), Some(b)) => Some(cmp::max(a, *b)),
            };
            cum_null_count += *null_count;

            // Create file with statistics.
            let mut file = PartitionedFile::new("x".to_string(), 100);
            let stats = Statistics {
                num_rows: Precision::Absent,
                total_byte_size: Precision::Absent,
                column_statistics: vec![ColumnStatistics {
                    null_count: Precision::Exact(*null_count),
                    min_value: Precision::Exact(ScalarValue::Int32(*min)),
                    max_value: Precision::Exact(ScalarValue::Int32(*max)),
                    ..Default::default()
                }],
            };
            file.statistics = Some(Arc::new(stats.clone()));
            file_groups.push(FileGroup::new(vec![file]).with_statistics(stats));
        }

        // add stats, for the whole parquet exec, for a single column
        statistics.column_statistics[0] = ColumnStatistics {
            null_count: Precision::Exact(cum_null_count),
            min_value: Precision::Exact(ScalarValue::Int32(cum_min)),
            max_value: Precision::Exact(ScalarValue::Int32(cum_max)),
            ..Default::default()
        };

        let config = Arc::new(
            FileScanConfigBuilder::new(
                ObjectStoreUrl::parse("test:///").unwrap(),
                schema(),
                Arc::new(ParquetSource::new(Default::default())),
            )
            .with_file_groups(file_groups)
            .with_output_ordering(output_ordering)
            .with_statistics(statistics)
            .build(),
        );

        Arc::new(DataSourceExec::new(config))
    }

    fn union_exec(input: Vec<Arc<dyn ExecutionPlan>>) -> Arc<dyn ExecutionPlan> {
        Arc::new(UnionExec::new(input))
    }

    fn sort_exec(
        sort_exprs: LexOrdering,
        input: Arc<dyn ExecutionPlan>,
        preserve_partitioning: bool,
    ) -> Arc<dyn ExecutionPlan> {
        let new_sort = SortExec::new(sort_exprs, input)
            .with_preserve_partitioning(preserve_partitioning);
        Arc::new(new_sort)
    }

    fn str_lexical_ranges(lex_ranges: &[LexicalRange]) -> String {
        lex_ranges
            .iter()
            .map(|lr| lr.to_string())
            .collect_vec()
            .join(", ")
    }

    /// Build a test physical plan like:
    /// UNION
    ///     ParquetExec (range_a)
    ///     SORT
    ///         UNION
    ///             ParquetExec (range_b_1)
    ///             ParquetExec (range_b_2)
    fn build_test_case(
        ordering: &LexOrdering,
        key_ranges: &[KeyRange; 3],
    ) -> Arc<dyn ExecutionPlan> {
        let [range_a, range_b_1, range_b_2] = key_ranges;

        let datasrc_a =
            parquet_exec_with_sort_with_statistics(vec![ordering.clone()], &[range_a]);

        let datasrc_b1 =
            parquet_exec_with_sort_with_statistics(vec![ordering.clone()], &[range_b_1]);
        let datasrc_b2 =
            parquet_exec_with_sort_with_statistics(vec![ordering.clone()], &[range_b_2]);
        let b = sort_exec(
            ordering.clone(),
            union_exec(vec![datasrc_b1, datasrc_b2]),
            false,
        );

        union_exec(vec![datasrc_a, b])
    }

    #[test]
    fn test_extract_disjoint_ranges() -> Result<()> {
        let plan_ranges_disjoint = &[
            KeyRange {
                min: Some(1000),
                max: Some(2000),
                null_count: 0,
            },
            KeyRange {
                min: Some(2001),
                max: Some(3000),
                null_count: 0,
            },
            KeyRange {
                min: Some(3001),
                max: Some(3500),
                null_count: 0,
            },
        ];
        let plan_ranges_overlapping = &[
            KeyRange {
                min: Some(1000),
                max: Some(2010),
                null_count: 0,
            },
            KeyRange {
                min: Some(2001),
                max: Some(3000),
                null_count: 0,
            },
            KeyRange {
                min: Some(3001),
                max: Some(3500),
                null_count: 0,
            },
        ];

        // Test: ASC, is disjoint (not overlapping)
        // (1000)->(2000), (2001)->(3500)
        let options = SortOptions {
            descending: false,
            ..Default::default()
        };
        let sort_expr = PhysicalSortExpr::new(col("a", &schema())?, options);
        let lex_ordering = LexOrdering::new(vec![sort_expr]);
        let plan = build_test_case(&lex_ordering, plan_ranges_disjoint);
        let Some(actual) = extract_disjoint_ranges_from_plan(&lex_ordering, &plan)?
        else {
            unreachable!("should find disjoint")
        };
        assert_eq!(actual.indices(), &[0, 1], "should find proper ordering");
        assert_eq!(
            format!("{}", str_lexical_ranges(actual.value_ranges())),
            "(1000)->(2000), (2001)->(3500)",
            "should find proper ranges"
        );

        // Test: ASC, is overlapping
        // (1000)->(2010), (2001)->(3500)
        let options = SortOptions {
            descending: false,
            ..Default::default()
        };
        let sort_expr = PhysicalSortExpr::new(col("a", &schema())?, options);
        let lex_ordering = LexOrdering::new(vec![sort_expr]);
        let plan = build_test_case(&lex_ordering, plan_ranges_overlapping);
        let actual = extract_disjoint_ranges_from_plan(&lex_ordering, &plan)?;
        assert!(actual.is_none(), "should not find disjoint range");

        // Test: DESC, is disjoint (not overlapping)
        // (2001)->(3500), (1000)->(2000)
        let options = SortOptions {
            descending: true,
            ..Default::default()
        };
        let sort_expr = PhysicalSortExpr::new(col("a", &schema())?, options);
        let lex_ordering = LexOrdering::new(vec![sort_expr]);
        let plan = build_test_case(&lex_ordering, plan_ranges_disjoint);
        let Some(actual) = extract_disjoint_ranges_from_plan(&lex_ordering, &plan)?
        else {
            unreachable!("should find disjoint")
        };
        assert_eq!(actual.indices(), &[1, 0], "should find proper ordering"); // NOTE THE INVERSE ORDER
        assert_eq!(
            format!("{}", str_lexical_ranges(actual.value_ranges())),
            "(2001)->(3500), (1000)->(2000)",
            "should find proper ranges"
        );

        // Test: DESC, is overlapping
        // (2001)->(3500), (1000)->(2010)
        let options = SortOptions {
            descending: false,
            ..Default::default()
        };
        let sort_expr = PhysicalSortExpr::new(col("a", &schema())?, options);
        let lex_ordering = LexOrdering::new(vec![sort_expr]);
        let plan = build_test_case(&lex_ordering, plan_ranges_overlapping);
        let actual = extract_disjoint_ranges_from_plan(&lex_ordering, &plan)?;
        assert!(actual.is_none(), "should not find disjoint range");

        Ok(())
    }

    /// Same as `test_extract_disjoint_ranges`, but include nulls first.
    #[test]
    fn test_extract_disjoint_ranges_nulls_first() -> Result<()> {
        // NULL is min_value if ASC
        let plan_ranges_disjoint_if_asc = &[
            KeyRange {
                min: Some(1000),
                max: Some(2000),
                null_count: 1, // null is min value
            },
            KeyRange {
                min: Some(2001),
                max: Some(3000),
                null_count: 0,
            },
            KeyRange {
                min: Some(3001),
                max: Some(3500),
                null_count: 0,
            },
        ];
        let plan_ranges_overlap_if_asc = &[
            KeyRange {
                min: Some(1000),
                max: Some(2000),
                null_count: 0,
            },
            KeyRange {
                min: Some(2001),
                max: Some(3000),
                null_count: 0,
            },
            KeyRange {
                min: Some(3001),
                max: Some(3500),
                null_count: 1, // null is min value, this will be overlapping
            },
        ];

        // NULL is max_value if DESC
        let plan_ranges_disjoint_if_desc = &[
            KeyRange {
                min: Some(1000),
                max: Some(2000),
                null_count: 0,
            },
            KeyRange {
                min: Some(2001),
                max: Some(3000),
                null_count: 0,
            },
            KeyRange {
                min: Some(3001),
                max: Some(3500),
                null_count: 1, // null is max value
            },
        ];
        let plan_ranges_overlap_if_desc = &[
            KeyRange {
                min: Some(1000),
                max: Some(2000),
                null_count: 1, // null is max value, this will be overlapping
            },
            KeyRange {
                min: Some(2001),
                max: Some(3000),
                null_count: 0,
            },
            KeyRange {
                min: Some(3001),
                max: Some(3500),
                null_count: 0,
            },
        ];

        // Test: ASC, is disjoint (not overlapping)
        // (1000)->(2000), (2001)->(3500), and partition<(1000)->(2000)> has the NULL (min value)
        let options = SortOptions {
            descending: false,
            nulls_first: true,
        };
        let sort_expr = PhysicalSortExpr::new(col("a", &schema())?, options);
        let lex_ordering = LexOrdering::new(vec![sort_expr]);
        let plan = build_test_case(&lex_ordering, plan_ranges_disjoint_if_asc);
        let Some(actual) = extract_disjoint_ranges_from_plan(&lex_ordering, &plan)?
        else {
            unreachable!("should find disjoint")
        };
        assert_eq!(actual.indices(), &[0, 1], "should find proper ordering");
        assert_eq!(
            format!("{}", str_lexical_ranges(actual.value_ranges())),
            "(NULL)->(2000), (2001)->(3500)",
            "should find proper ranges"
        );

        // Test: ASC, is overlapping
        // (1000)->(2000), (2001)->(3500), and partition<(2001)->(3500)> has the NULL (min value)
        let options = SortOptions {
            descending: false,
            nulls_first: true,
        };
        let sort_expr = PhysicalSortExpr::new(col("a", &schema())?, options);
        let lex_ordering = LexOrdering::new(vec![sort_expr]);
        let plan = build_test_case(&lex_ordering, plan_ranges_overlap_if_asc);
        let actual = extract_disjoint_ranges_from_plan(&lex_ordering, &plan)?;
        assert!(actual.is_none(), "should not find disjoint range");

        // Test: DESC, is disjoint (not overlapping)
        // (2001)->(3500), (1000)->(2000), and partition<(2001)->(3500)> has the NULL (max value)
        let options = SortOptions {
            descending: true,
            nulls_first: true,
        };
        let sort_expr = PhysicalSortExpr::new(col("a", &schema())?, options);
        let lex_ordering = LexOrdering::new(vec![sort_expr]);
        let plan = build_test_case(&lex_ordering, plan_ranges_disjoint_if_desc);
        let Some(actual) = extract_disjoint_ranges_from_plan(&lex_ordering, &plan)?
        else {
            unreachable!("should find disjoint")
        };
        assert_eq!(actual.indices(), &[1, 0], "should find proper ordering"); // NOTE THE INVERSE ORDER
        assert_eq!(
            format!("{}", str_lexical_ranges(actual.value_ranges())),
            "(2001)->(NULL), (1000)->(2000)",
            "should find proper ranges"
        );

        // Test: DESC, is overlapping
        // (2001)->(3500), (1000)->(2000), and partition<(1000)->(2000)> has the NULL (max value)
        let options = SortOptions {
            descending: true,
            nulls_first: true,
        };
        let sort_expr = PhysicalSortExpr::new(col("a", &schema())?, options);
        let lex_ordering = LexOrdering::new(vec![sort_expr]);
        let plan = build_test_case(&lex_ordering, plan_ranges_overlap_if_desc);
        let actual = extract_disjoint_ranges_from_plan(&lex_ordering, &plan)?;
        assert!(actual.is_none(), "should not find disjoint range");

        Ok(())
    }

    /// Same as `test_extract_disjoint_ranges`, but include nulls last.
    #[test]
    fn test_extract_disjoint_ranges_nulls_last() -> Result<()> {
        // NULL is max_value if ASC
        let plan_ranges_disjoint_if_asc = &[
            KeyRange {
                min: Some(1000),
                max: Some(2000),
                null_count: 0,
            },
            KeyRange {
                min: Some(2001),
                max: Some(3000),
                null_count: 0,
            },
            KeyRange {
                min: Some(3001),
                max: Some(3500),
                null_count: 1, // null is max value
            },
        ];
        let plan_ranges_overlap_if_asc = &[
            KeyRange {
                min: Some(1000),
                max: Some(2000),
                null_count: 1, // null is max value, this will be overlapping
            },
            KeyRange {
                min: Some(2001),
                max: Some(3000),
                null_count: 0,
            },
            KeyRange {
                min: Some(3001),
                max: Some(3500),
                null_count: 0,
            },
        ];

        // NULL is min_value if DESC
        let plan_ranges_disjoint_if_desc = &[
            KeyRange {
                min: Some(1000),
                max: Some(2000),
                null_count: 1, // null is min value
            },
            KeyRange {
                min: Some(2001),
                max: Some(3000),
                null_count: 0,
            },
            KeyRange {
                min: Some(3001),
                max: Some(3500),
                null_count: 0,
            },
        ];
        let plan_ranges_overlap_if_desc = &[
            KeyRange {
                min: Some(1000),
                max: Some(2000),
                null_count: 0,
            },
            KeyRange {
                min: Some(2001),
                max: Some(3000),
                null_count: 0,
            },
            KeyRange {
                min: Some(3001),
                max: Some(3500),
                null_count: 1, // null is min value, this will be overlapping
            },
        ];

        // Test: ASC, is disjoint (not overlapping)
        // (1000)->(2000), (2001)->(3500), and partition<(2001)->(3500)> has the NULL (max value)
        let options = SortOptions {
            descending: false,
            nulls_first: false,
        };
        let sort_expr = PhysicalSortExpr::new(col("a", &schema())?, options);
        let lex_ordering = LexOrdering::new(vec![sort_expr]);
        let plan = build_test_case(&lex_ordering, plan_ranges_disjoint_if_asc);
        let Some(actual) = extract_disjoint_ranges_from_plan(&lex_ordering, &plan)?
        else {
            unreachable!("should find disjoint")
        };
        assert_eq!(actual.indices(), &[0, 1], "should find proper ordering");
        assert_eq!(
            format!("{}", str_lexical_ranges(actual.value_ranges())),
            "(1000)->(2000), (2001)->(NULL)",
            "should find proper ranges"
        );

        // Test: ASC, is overlapping
        // (1000)->(2000), (2001)->(3500), and partition<(1000)->(2000)> has the NULL (max value)
        let options = SortOptions {
            descending: false,
            nulls_first: false,
        };
        let sort_expr = PhysicalSortExpr::new(col("a", &schema())?, options);
        let lex_ordering = LexOrdering::new(vec![sort_expr]);
        let plan = build_test_case(&lex_ordering, plan_ranges_overlap_if_asc);
        let actual = extract_disjoint_ranges_from_plan(&lex_ordering, &plan)?;
        assert!(actual.is_none(), "should not find disjoint range");

        // Test: DESC, is disjoint (not overlapping)
        // (2001)->(3500), (1000)->(2000), and partition<(1000)->(2000)> has the NULL (min value)
        let options = SortOptions {
            descending: true,
            nulls_first: false,
        };
        let sort_expr = PhysicalSortExpr::new(col("a", &schema())?, options);
        let lex_ordering = LexOrdering::new(vec![sort_expr]);
        let plan = build_test_case(&lex_ordering, plan_ranges_disjoint_if_desc);
        let Some(actual) = extract_disjoint_ranges_from_plan(&lex_ordering, &plan)?
        else {
            unreachable!("should find disjoint")
        };
        assert_eq!(actual.indices(), &[1, 0], "should find proper ordering"); // NOTE THE INVERSE ORDER
        assert_eq!(
            format!("{}", str_lexical_ranges(actual.value_ranges())),
            "(2001)->(3500), (NULL)->(2000)",
            "should find proper ranges"
        );

        // Test: DESC, is overlapping
        // (2001)->(3500), (1000)->(2000), and partition<(2001)->(3500)> has the NULL (min value)
        let options = SortOptions {
            descending: true,
            nulls_first: false,
        };
        let sort_expr = PhysicalSortExpr::new(col("a", &schema())?, options);
        let lex_ordering = LexOrdering::new(vec![sort_expr]);
        let plan = build_test_case(&lex_ordering, plan_ranges_overlap_if_desc);
        let actual = extract_disjoint_ranges_from_plan(&lex_ordering, &plan)?;
        assert!(actual.is_none(), "should not find disjoint range");

        Ok(())
    }

    /// Build a test physical plan like:
    /// UNION
    ///     ParquetExec (range_a)
    ///     ParquetExec (range_b_1, range_b_2)
    fn build_test_case_multiple_partition_parquet_exec(
        ordering: &LexOrdering,
        key_ranges: &[KeyRange; 3],
    ) -> Arc<dyn ExecutionPlan> {
        let [range_a, range_b_1, range_b_2] = key_ranges;

        let datasrc_a =
            parquet_exec_with_sort_with_statistics(vec![ordering.clone()], &[range_a]);
        let datasrc_b = parquet_exec_with_sort_with_statistics(
            vec![ordering.clone()],
            &[range_b_1, range_b_2],
        );

        union_exec(vec![datasrc_a, datasrc_b])
    }

    #[test]
    fn test_extract_multiple_partitions_union_parquet_exec() -> Result<()> {
        let plan_ranges_disjoint = &[
            KeyRange {
                min: Some(1000),
                max: Some(2000),
                null_count: 0,
            },
            KeyRange {
                min: Some(2001),
                max: Some(3000),
                null_count: 0,
            },
            KeyRange {
                min: Some(3001),
                max: Some(3500),
                null_count: 0,
            },
        ];
        let plan_ranges_overlapping = &[
            KeyRange {
                min: Some(1000),
                max: Some(2010),
                null_count: 0,
            },
            KeyRange {
                min: Some(2001),
                max: Some(3000),
                null_count: 0,
            },
            KeyRange {
                min: Some(3001),
                max: Some(3500),
                null_count: 0,
            },
        ];

        // Test: ASC, is disjoint (not overlapping)
        let options = SortOptions {
            descending: false,
            ..Default::default()
        };
        let sort_expr = PhysicalSortExpr::new(col("a", &schema())?, options);
        let lex_ordering = LexOrdering::new(vec![sort_expr]);
        let plan = build_test_case_multiple_partition_parquet_exec(
            &lex_ordering,
            plan_ranges_disjoint,
        );
        let Some(actual) = extract_disjoint_ranges_from_plan(&lex_ordering, &plan)?
        else {
            unreachable!("should find disjoint")
        };
        assert_eq!(actual.indices(), &[0, 1, 2], "should find proper ordering");
        assert_eq!(
            format!("{}", str_lexical_ranges(actual.value_ranges())),
            "(1000)->(2000), (2001)->(3000), (3001)->(3500)",
            "should find proper ranges"
        );

        // Test: ASC, is overlapping
        let options = SortOptions {
            descending: false,
            ..Default::default()
        };
        let sort_expr = PhysicalSortExpr::new(col("a", &schema())?, options);
        let lex_ordering = LexOrdering::new(vec![sort_expr]);
        let plan = build_test_case_multiple_partition_parquet_exec(
            &lex_ordering,
            plan_ranges_overlapping,
        );
        let actual = extract_disjoint_ranges_from_plan(&lex_ordering, &plan)?;
        assert!(actual.is_none(), "should not find disjoint range");

        Ok(())
    }
}
