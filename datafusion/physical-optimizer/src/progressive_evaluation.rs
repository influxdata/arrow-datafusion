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

//! TODO: physical optimizer run to conditionally swap the SPM with the ProgressiveEvalExec

mod extract_ranges;
mod lexical_ranges;
mod statistics;
mod util;

use itertools::Itertools;
use std::sync::Arc;

use datafusion_common::{
    tree_node::{Transformed, TreeNode},
    Result,
};
use datafusion_physical_plan::{
    sorts::sort_preserving_merge::SortPreservingMergeExec, union::UnionExec,
    ExecutionPlan,
};
use extract_ranges::extract_disjoint_ranges_from_plan;
use util::split_parquet_files;

use crate::PhysicalOptimizerRule;

#[allow(dead_code)]
#[derive(Debug)]
struct InsertProgressiveEval;

impl PhysicalOptimizerRule for InsertProgressiveEval {
    fn name(&self) -> &str {
        "TBD"
    }

    fn schema_check(&self) -> bool {
        false
    }

    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &datafusion_common::config::ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|plan| {
            // Find SortPreservingMergeExec
            let Some(sort_preserving_merge_exec) =
                plan.as_any().downcast_ref::<SortPreservingMergeExec>()
            else {
                return Ok(Transformed::no(plan));
            };

            // Split file groups to maximize potential disjoint ranges.
            let new_inputs: Vec<Arc<dyn ExecutionPlan>> = sort_preserving_merge_exec
                .children()
                .into_iter()
                .map(|spm_child| {
                    Arc::clone(spm_child)
                        .transform_down(|plan| {
                            split_parquet_files(plan, sort_preserving_merge_exec.expr())
                        })
                        .map(|t| t.data)
                })
                .try_collect()?;
            let transformed_input_plan = Arc::new(UnionExec::new(new_inputs)) as _;

            // try to extract the lexical ranges for the input partitions
            let Ok(Some(_lexical_ranges)) = extract_disjoint_ranges_from_plan(
                sort_preserving_merge_exec.expr(),
                &transformed_input_plan,
            ) else {
                return Ok(Transformed::no(plan));
            };

            // confirm we still have the ordering needed for the SPM
            assert!(transformed_input_plan
                .properties()
                .equivalence_properties()
                .ordering_satisfy(sort_preserving_merge_exec.expr()));

            // Replace SortPreservingMergeExec with ProgressiveEvalExec
            // TODO: have the ProgressiveEvalExec perform that partition mapping
            // let progresive_eval_exec = Arc::new(ProgressiveEvalExec::new(
            //     transformed_input_plan,
            //     lexical_ranges,
            //     sort_preserving_merge_exec.fetch(),
            // ));
            // Ok(Transformed::yes(progresive_eval_exec))

            Ok(Transformed::no(plan))
        })
        .map(|t| t.data)
    }
}
