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

//! [`NonOverlappingOrderedLexicalRanges`] represents ranges of lexically ordered values.

use arrow::array::ArrayRef;
use arrow::compute::SortOptions;
use arrow::row::{Row, RowConverter, Rows, SortField};
use datafusion_common::{error::DataFusionError, Result, ScalarValue};
use std::fmt::Display;
use std::sync::Arc;

/// # Lexical Space
///
/// The "Lexical Space" is all possible values of a sort order (set of sort
/// expressions).
///
/// For example, given data with a sort order of `A ASC, B ASC`
/// (`A` ascending, `B` ascending), then the lexical space is all the unique
/// combinations of `(A, B)`.
///
/// # Lexical Range
///
/// The "lexical range" of an input in this lexical space is
/// the minimum and maximum sort key values for that range.
///
/// For example, for data like
/// | `a` | `b` |
/// |--------|--------|
/// | 1 | 100 |
/// | 1 | 200 |
/// | 1 | 300 |
/// | 2 | 100 |
/// | 2 | 200 |
/// | 3 | 50 |
///
/// The lexical range is `min --> max`: `(1,100) --> (3,50)`
#[derive(Debug, Default, Clone)]
pub struct LexicalRange {
    /// The minimum value in the lexical space (one for each sort key)
    min: Vec<ScalarValue>,
    /// The maximum value in the lexical space (one for each sort key)
    max: Vec<ScalarValue>,
}

impl LexicalRange {
    pub fn builder() -> LexicalRangeBuilder {
        LexicalRangeBuilder::new()
    }
}

impl Display for LexicalRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({})->({})",
            self.min
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(","),
            self.max
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

/// Builder for [`LexicalRange`]
#[derive(Debug, Default, Clone)]
pub struct LexicalRangeBuilder {
    inner: LexicalRange,
}

impl LexicalRangeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(self) -> LexicalRange {
        self.inner
    }

    /// Adds min and max to the end of the in-progress ranges
    pub fn push(&mut self, min: ScalarValue, max: ScalarValue) {
        self.inner.min.push(min);
        self.inner.max.push(max);
    }
}

/// Represents ranges of lexically ordered values in a sort order.
///
/// One element for each input partition

#[derive(Debug)]
pub struct NonOverlappingOrderedLexicalRanges {
    /// Corresponding lexical range per partition, already reordered to match indices
    value_ranges: Vec<LexicalRange>,

    /// The order of the input partitions (rows of ranges) that would provide ensure they are
    /// sorted in lexical order
    indices: Vec<usize>,
}

impl Display for NonOverlappingOrderedLexicalRanges {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LexicalRanges\n  {:?}", self.indices,)
    }
}

impl NonOverlappingOrderedLexicalRanges {
    /// Attempt to create a new [`NonOverlappingOrderedLexicalRanges`]
    ///
    /// Returns None if:
    /// * - There are no ranges
    /// * - The ranges are not disjoint (aka they overlap in the lexical space)
    ///
    /// Returns Err if there is an error converting values
    pub fn try_new(
        sort_options: &[SortOptions],
        ranges_per_partition: Vec<LexicalRange>,
    ) -> Result<Option<Self>> {
        if ranges_per_partition.is_empty() {
            return Ok(None);
        }
        // convert to min/maxes, as VecPerPartition<VecPerSortKey>
        let (mins, maxes) = ranges_per_partition.clone().into_iter().fold(
            (vec![], vec![]),
            |(mut mins, mut maxes), partition_range| {
                let LexicalRange {
                    min: min_per_sort_key,
                    max: max_per_sort_key,
                } = partition_range;
                mins.push(min_per_sort_key);
                maxes.push(max_per_sort_key);
                (mins, maxes)
            },
        );
        let rows = ConvertedRows::try_new(sort_options, mins, maxes)?;

        let mut indices = (0..rows.num_rows()).collect::<Vec<_>>();
        indices.sort_by_key(|&i| rows.min_row(i));

        // check that the ranges are disjoint
        if !rows.are_disjoint(&indices) {
            return Ok(None);
        };

        // reorder lexical ranges
        let mut value_ranges = ranges_per_partition
            .into_iter()
            .enumerate()
            .map(|(input_partition_idx, range)| {
                let reordering_idx = indices
                    .iter()
                    .position(|reorder_idx| *reorder_idx == input_partition_idx)
                    .expect("missing partition in reordering indices");
                (reordering_idx, range)
            })
            .collect::<Vec<_>>();
        value_ranges.sort_by_key(|(reord_i, _)| *reord_i);

        let value_ranges = value_ranges
            .into_iter()
            .map(|(_, range)| range)
            .collect::<Vec<_>>();

        Ok(Some(Self {
            value_ranges,
            indices,
        }))
    }

    /// Returns the indices that describe the order input partitions must be reordered
    /// to be non overlapping and ordered in lexical order.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Returns the lexical ranges already re-ordered
    pub fn value_ranges(&self) -> &[LexicalRange] {
        &self.value_ranges
    }
}

/// Result of converting multiple-column ScalarValue rows to columns.
#[derive(Debug)]
struct ConvertedRows {
    /// Use the same row converter for mins and maxes, otherwise they cannot be compared.
    converter: RowConverter,

    mins: Rows,
    maxes: Rows,
}

impl ConvertedRows {
    /// Create new [`ConvertedRows`] from the vector of sort keys and specified options.
    ///
    /// Keys are in the format `VecPerPartition<VecPerSortKey<value>>`
    fn try_new(
        sort_options: &[SortOptions],
        min_keys: Vec<Vec<ScalarValue>>,
        max_keys: Vec<Vec<ScalarValue>>,
    ) -> Result<Self> {
        assert_eq!(sort_options.len(), min_keys[0].len());
        assert_eq!(sort_options.len(), max_keys[0].len());

        // build converter using min keys
        let arrays = pivot_to_arrays(min_keys)?;
        let converter_fields = arrays
            .iter()
            .zip(sort_options.iter())
            .map(|(a, options)| {
                SortField::new_with_options(a.data_type().clone(), *options)
            })
            .collect::<Vec<_>>();
        let converter = RowConverter::new(converter_fields)?;
        let mins = converter.convert_columns(&arrays)?;

        // build maxes
        let arrays = pivot_to_arrays(max_keys)?;
        let maxes = converter.convert_columns(&arrays)?;

        Ok(Self {
            converter,
            mins,
            maxes,
        })
    }

    fn num_rows(&self) -> usize {
        self.mins.num_rows()
    }

    /// Return the min (as Row) at the specified index
    fn min_row(&self, index: usize) -> Row<'_> {
        self.mins.row(index)
    }

    /// Return the max (as Row) at the specified index
    fn max_row(&self, index: usize) -> Row<'_> {
        self.maxes.row(index)
    }

    /// Return the min value at the specified index
    fn min_value(&self, index: usize) -> Result<Vec<ArrayRef>> {
        let values = self
            .converter
            .convert_rows([self.min_row(index)])
            .map_err(|e| DataFusionError::ArrowError(e, None))?;
        Ok(values.iter().map(Arc::clone).collect::<Vec<_>>())
    }

    /// Return the max value at the specified index
    fn max_value(&self, index: usize) -> Result<Vec<ArrayRef>> {
        let values = self
            .converter
            .convert_rows([self.max_row(index)])
            .map_err(|e| DataFusionError::ArrowError(e, None))?;
        Ok(values.iter().map(Arc::clone).collect::<Vec<_>>())
    }

    // Return true if ranges are disjoint when order according to ordered_partition_idx.
    fn are_disjoint(&self, ordered_by_min_partition_indices: &[usize]) -> bool {
        for index_index in 1..ordered_by_min_partition_indices.len() {
            let index = ordered_by_min_partition_indices[index_index];
            let prev_index = ordered_by_min_partition_indices[index_index - 1];

            // Ordering is by sort key, and may be desc.
            // Therefore need to confirm that the min & max of the current range is greater than the previous range.
            let start_exclusive = self.min_row(index) > self.min_row(prev_index)
                && self.min_row(index) > self.max_row(prev_index);
            let end_exclusive = self.max_row(index) > self.min_row(prev_index)
                && self.max_row(index) > self.max_row(prev_index);
            if !(start_exclusive && end_exclusive) {
                return false;
            }
        }
        true
    }
}

/// Convert a multi-column ScalarValue row to columns
fn pivot_to_arrays(keys: Vec<Vec<ScalarValue>>) -> Result<Vec<ArrayRef>> {
    let mut arrays = vec![];
    for col in 0..keys[0].len() {
        let mut column = vec![];
        for row in &keys {
            // todo avoid this clone (with take)
            column.push(row[col].clone());
        }
        arrays.push(ScalarValue::iter_to_array(column)?)
    }
    Ok(arrays)
}

#[cfg(test)]
mod tests {
    use arrow::compute::SortOptions;
    use datafusion_common::ScalarValue;
    use itertools::Itertools;

    use super::{LexicalRange, NonOverlappingOrderedLexicalRanges};

    struct TestCase {
        partitions: Vec<TestPartition>,
        num_sort_keys: usize,
        name: &'static str,
        expected_ranges_per_partition: Vec<&'static str>, // before ordering
        expect_disjoint: bool,
        expected_ordered_indices: Vec<ExpectedOrderedIndices>, // after ordering
    }

    impl TestCase {
        fn run(self) {
            // Test: confirm found proper lexical ranges
            let lexical_ranges = self.build_lexical_ranges();
            let expected = self
                .expected_ranges_per_partition
                .iter()
                .map(|str| str.to_string())
                .collect_vec();
            let actual = lexical_ranges
                .iter()
                .map(|range| format!("{range}"))
                .collect_vec();
            assert_eq!(
                actual, expected,
                "ERROR {}: expected ranges {:?} but found {:?}",
                self.name, expected, actual
            );

            // Test: confirm found proper non overlapping (or not) ranges per given sort ordering
            [
                TestSortOption::AscNullsFirst,
                TestSortOption::AscNullsLast,
                TestSortOption::DescNullsFirst,
                TestSortOption::DescNullsLast,
            ].into_iter().for_each(|sort_ordering| {
                if let Some(nonoverlapping) = NonOverlappingOrderedLexicalRanges::try_new(&sort_ordering.sort_options(self.num_sort_keys).as_slice(), lexical_ranges.clone()).expect("should not error") {
                    assert!(self.expect_disjoint, "ERROR {} for {:?}: expected ranges to overlap, instead found disjoint ranges", self.name, &sort_ordering);

                    let expected_ordered_indices = self.find_expected_indices(&sort_ordering);
                    assert_eq!(expected_ordered_indices, nonoverlapping.indices(), "ERROR {} for {:?}: expected to find indices ordered {:?}, instead found ordering {:?}", self.name, &sort_ordering, expected_ordered_indices, nonoverlapping.indices());
                } else {
                    assert!(!self.expect_disjoint, "ERROR {} for {:?}: expected to find disjoint ranges, instead could either not detect ranges or found overlapping ranges", self.name, &sort_ordering);
                };
            });
        }

        fn build_lexical_ranges(&self) -> Vec<LexicalRange> {
            self.partitions
                .iter()
                .map(|partition| {
                    let mut builder = LexicalRange::builder();
                    for SortKeyRange { min, max } in &partition.range_per_sort_key {
                        builder.push(min.clone(), max.clone());
                    }
                    builder.build()
                })
                .collect_vec()
        }

        fn find_expected_indices(&self, sort_ordering: &TestSortOption) -> &[usize] {
            self.expected_ordered_indices
                .iter()
                .find(|ord| ord.sort_ordering == *sort_ordering)
                .expect("should have expected outcome")
                .expected_indices
                .as_ref()
        }
    }

    struct TestPartition {
        range_per_sort_key: Vec<SortKeyRange>,
    }

    /// Range of a sort key. Note that this is not impacted by directionality of ordering (e.g. [`SortOptions`]).
    struct SortKeyRange {
        min: ScalarValue,
        max: ScalarValue,
    }

    fn build_partition_with_single_sort_key(
        ints: (Option<i64>, Option<i64>),
    ) -> TestPartition {
        let range_per_sort_key = vec![SortKeyRange {
            min: ScalarValue::Int64(ints.0),
            max: ScalarValue::Int64(ints.1),
        }];
        TestPartition { range_per_sort_key }
    }

    fn build_partition_with_multiple_sort_keys(
        ints: (Option<i64>, Option<i64>),
        strings: (Option<String>, Option<String>),
        times: (Option<i64>, Option<i64>),
    ) -> TestPartition {
        let range_per_sort_key = vec![
            SortKeyRange {
                min: ScalarValue::Int64(ints.0),
                max: ScalarValue::Int64(ints.1),
            },
            SortKeyRange {
                min: ScalarValue::Utf8(strings.0),
                max: ScalarValue::Utf8(strings.1),
            },
            SortKeyRange {
                min: ScalarValue::TimestampNanosecond(times.0, None),
                max: ScalarValue::TimestampNanosecond(times.1, None),
            },
        ];
        TestPartition { range_per_sort_key }
    }

    #[derive(Debug, PartialEq)]
    enum TestSortOption {
        AscNullsLast,
        AscNullsFirst,
        DescNullsLast,
        DescNullsFirst,
    }

    impl TestSortOption {
        fn sort_options(&self, len: usize) -> Vec<SortOptions> {
            match self {
                Self::AscNullsLast => std::iter::repeat_n(
                    SortOptions {
                        descending: false,
                        nulls_first: false,
                    },
                    len,
                )
                .collect_vec(),
                Self::AscNullsFirst => std::iter::repeat_n(
                    SortOptions {
                        descending: false,
                        nulls_first: true,
                    },
                    len,
                )
                .collect_vec(),
                Self::DescNullsLast => std::iter::repeat_n(
                    SortOptions {
                        descending: true,
                        nulls_first: false,
                    },
                    len,
                )
                .collect_vec(),
                Self::DescNullsFirst => std::iter::repeat_n(
                    SortOptions {
                        descending: true,
                        nulls_first: true,
                    },
                    len,
                )
                .collect_vec(),
            }
        }
    }

    struct ExpectedOrderedIndices {
        /// the ordering (e.g. [`SortOptions`]) applied to all columns in the sort key.
        sort_ordering: TestSortOption,
        /// Expected outcome ordering with this sort_ordering applied.
        expected_indices: Vec<usize>,
    }

    impl From<(TestSortOption, Vec<usize>)> for ExpectedOrderedIndices {
        fn from(value: (TestSortOption, Vec<usize>)) -> Self {
            Self {
                sort_ordering: value.0,
                expected_indices: value.1,
            }
        }
    }

    #[test]
    fn test_disjointness_single_key() {
        let cases = [
            TestCase {
                partitions: vec![
                    build_partition_with_single_sort_key((Some(1), Some(10))),
                    build_partition_with_single_sort_key((Some(2), Some(10))),
                    build_partition_with_single_sort_key((Some(0), Some(0))),
                ],
                num_sort_keys: 1,
                name: "order_by_single_sort_key__overlapping",
                expected_ranges_per_partition: vec!["(1)->(10)", "(2)->(10)", "(0)->(0)"],
                expect_disjoint: false,
                expected_ordered_indices: vec![],
            },
            TestCase {
                partitions: vec![
                    build_partition_with_single_sort_key((Some(1), Some(10))),
                    build_partition_with_single_sort_key((Some(11), Some(20))),
                    build_partition_with_single_sort_key((Some(0), Some(0))),
                ],
                num_sort_keys: 1,
                name: "order_by_single_sort_key__disjoint",
                expected_ranges_per_partition: vec![
                    "(1)->(10)",
                    "(11)->(20)",
                    "(0)->(0)",
                ],
                expect_disjoint: true,
                expected_ordered_indices: vec![
                    (TestSortOption::AscNullsFirst, vec![2, 0, 1]).into(),
                    (TestSortOption::AscNullsLast, vec![2, 0, 1]).into(),
                    (TestSortOption::DescNullsFirst, vec![1, 0, 2]).into(),
                    (TestSortOption::DescNullsLast, vec![1, 0, 2]).into(),
                ],
            },
        ];

        cases.into_iter().for_each(|test_case| test_case.run());
    }

    #[test]
    fn test_disjointness_multiple_sort_keys() {
        let cases = [
            /* Using the first sort key, an integer, as the decider. */
            TestCase {
                partitions: vec![
                    build_partition_with_multiple_sort_keys(
                        (Some(1), Some(10)),
                        (Some("same".into()), Some("same".into())),
                        (Some(1), Some(1)),
                    ),
                    build_partition_with_multiple_sort_keys(
                        (Some(2), Some(10)),
                        (Some("same".into()), Some("same".into())),
                        (Some(1), Some(1)),
                    ),
                    build_partition_with_multiple_sort_keys(
                        (Some(0), Some(0)),
                        (Some("same".into()), Some("same".into())),
                        (Some(1), Some(1)),
                    ),
                ],
                num_sort_keys: 3,
                name: "order_by_first_sort_key__overlapping",
                expected_ranges_per_partition: vec![
                    "(1,same,1)->(10,same,1)",
                    "(2,same,1)->(10,same,1)",
                    "(0,same,1)->(0,same,1)",
                ],
                expect_disjoint: false,
                expected_ordered_indices: vec![],
            },
            TestCase {
                partitions: vec![
                    build_partition_with_multiple_sort_keys(
                        (Some(1), Some(10)),
                        (Some("same".into()), Some("same".into())),
                        (Some(1), Some(1)),
                    ),
                    build_partition_with_multiple_sort_keys(
                        (Some(11), Some(20)),
                        (Some("same".into()), Some("same".into())),
                        (Some(1), Some(1)),
                    ),
                    build_partition_with_multiple_sort_keys(
                        (Some(0), Some(0)),
                        (Some("same".into()), Some("same".into())),
                        (Some(1), Some(1)),
                    ),
                ],
                num_sort_keys: 3,
                name: "order_by_first_sort_key__disjoint",
                expected_ranges_per_partition: vec![
                    "(1,same,1)->(10,same,1)",
                    "(11,same,1)->(20,same,1)",
                    "(0,same,1)->(0,same,1)",
                ],
                expect_disjoint: true,
                expected_ordered_indices: vec![
                    (TestSortOption::AscNullsFirst, vec![2, 0, 1]).into(),
                    (TestSortOption::AscNullsLast, vec![2, 0, 1]).into(),
                    (TestSortOption::DescNullsFirst, vec![1, 0, 2]).into(),
                    (TestSortOption::DescNullsLast, vec![1, 0, 2]).into(),
                ],
            },
            /* Using the middle sort key, a string, as the decider. */
            TestCase {
                partitions: vec![
                    build_partition_with_multiple_sort_keys(
                        (Some(1), Some(1)),
                        (Some("a".into()), Some("d".into())),
                        (Some(1), Some(1)),
                    ),
                    build_partition_with_multiple_sort_keys(
                        (Some(1), Some(1)),
                        (Some("f".into()), Some("g".into())),
                        (Some(1), Some(1)),
                    ),
                    build_partition_with_multiple_sort_keys(
                        (Some(1), Some(1)),
                        (Some("c".into()), Some("e".into())),
                        (Some(1), Some(1)),
                    ),
                ],
                num_sort_keys: 3,
                name: "order_by_middle_sort_key__overlapping",
                expected_ranges_per_partition: vec![
                    "(1,a,1)->(1,d,1)",
                    "(1,f,1)->(1,g,1)",
                    "(1,c,1)->(1,e,1)",
                ],
                expect_disjoint: false,
                expected_ordered_indices: vec![],
            },
            TestCase {
                partitions: vec![
                    build_partition_with_multiple_sort_keys(
                        (Some(1), Some(1)),
                        (Some("a".into()), Some("b".into())),
                        (Some(1), Some(1)),
                    ),
                    build_partition_with_multiple_sort_keys(
                        (Some(1), Some(1)),
                        (Some("f".into()), Some("g".into())),
                        (Some(1), Some(1)),
                    ),
                    build_partition_with_multiple_sort_keys(
                        (Some(1), Some(1)),
                        (Some("c".into()), Some("e".into())),
                        (Some(1), Some(1)),
                    ),
                ],
                num_sort_keys: 3,
                name: "order_by_middle_sort_key__disjoint",
                expected_ranges_per_partition: vec![
                    "(1,a,1)->(1,b,1)",
                    "(1,f,1)->(1,g,1)",
                    "(1,c,1)->(1,e,1)",
                ],
                expect_disjoint: true,
                expected_ordered_indices: vec![
                    (TestSortOption::AscNullsFirst, vec![0, 2, 1]).into(),
                    (TestSortOption::AscNullsLast, vec![0, 2, 1]).into(),
                    (TestSortOption::DescNullsFirst, vec![1, 2, 0]).into(),
                    (TestSortOption::DescNullsLast, vec![1, 2, 0]).into(),
                ],
            },
            /* Using the last sort key, a nanosecond timestamp, as the decider. */
            TestCase {
                partitions: vec![
                    build_partition_with_multiple_sort_keys(
                        (Some(1), Some(1)),
                        (Some("same".into()), Some("same".into())),
                        (Some(50000000), Some(50000001)),
                    ),
                    build_partition_with_multiple_sort_keys(
                        (Some(1), Some(1)),
                        (Some("same".into()), Some("same".into())),
                        (Some(700000), Some(50000001)),
                    ),
                    build_partition_with_multiple_sort_keys(
                        (Some(1), Some(1)),
                        (Some("same".into()), Some("same".into())),
                        (Some(100000000), Some(100000001)),
                    ),
                ],
                num_sort_keys: 3,
                name: "order_by_last_sort_key__overlapping",
                expected_ranges_per_partition: vec![
                    "(1,same,50000000)->(1,same,50000001)",
                    "(1,same,700000)->(1,same,50000001)",
                    "(1,same,100000000)->(1,same,100000001)",
                ],
                expect_disjoint: false,
                expected_ordered_indices: vec![],
            },
            TestCase {
                partitions: vec![
                    build_partition_with_multiple_sort_keys(
                        (Some(1), Some(1)),
                        (Some("same".into()), Some("same".into())),
                        (Some(50000000), Some(50000001)),
                    ),
                    build_partition_with_multiple_sort_keys(
                        (Some(1), Some(1)),
                        (Some("same".into()), Some("same".into())),
                        (Some(700000), Some(7000001)),
                    ),
                    build_partition_with_multiple_sort_keys(
                        (Some(1), Some(1)),
                        (Some("same".into()), Some("same".into())),
                        (Some(100000000), Some(100000001)),
                    ),
                ],
                num_sort_keys: 3,
                name: "order_by_last_sort_key__disjoint",
                expected_ranges_per_partition: vec![
                    "(1,same,50000000)->(1,same,50000001)",
                    "(1,same,700000)->(1,same,7000001)",
                    "(1,same,100000000)->(1,same,100000001)",
                ],
                expect_disjoint: true,
                expected_ordered_indices: vec![
                    (TestSortOption::AscNullsFirst, vec![1, 0, 2]).into(),
                    (TestSortOption::AscNullsLast, vec![1, 0, 2]).into(),
                    (TestSortOption::DescNullsFirst, vec![2, 0, 1]).into(),
                    (TestSortOption::DescNullsLast, vec![2, 0, 1]).into(),
                ],
            },
        ];

        cases.into_iter().for_each(|test_case| test_case.run());
    }

    #[test]
    fn test_disjointness_with_nulls() {
        let cases = [
            TestCase {
                partitions: vec![
                    build_partition_with_single_sort_key((Some(1), Some(10))),
                    build_partition_with_single_sort_key((Some(2), Some(10))),
                    build_partition_with_single_sort_key((None, None)),
                ],
                num_sort_keys: 1,
                name: "order_by_nulls__overlapping",
                expected_ranges_per_partition: vec![
                    "(1)->(10)",
                    "(2)->(10)",
                    "(NULL)->(NULL)",
                ],
                expect_disjoint: false,
                expected_ordered_indices: vec![],
            },
            TestCase {
                partitions: vec![
                    build_partition_with_single_sort_key((Some(1), Some(10))),
                    build_partition_with_single_sort_key((Some(11), Some(20))),
                    build_partition_with_single_sort_key((None, None)),
                ],
                num_sort_keys: 1,
                name: "order_by_nulls__disjoint",
                expected_ranges_per_partition: vec![
                    "(1)->(10)",
                    "(11)->(20)",
                    "(NULL)->(NULL)",
                ],
                expect_disjoint: true,
                expected_ordered_indices: vec![
                    (TestSortOption::AscNullsFirst, vec![2, 0, 1]).into(),
                    (TestSortOption::AscNullsLast, vec![0, 1, 2]).into(),
                    (TestSortOption::DescNullsFirst, vec![2, 1, 0]).into(),
                    (TestSortOption::DescNullsLast, vec![1, 0, 2]).into(),
                ],
            },
        ];

        cases.into_iter().for_each(|test_case| test_case.run());
    }
}
