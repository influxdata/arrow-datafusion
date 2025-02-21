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

use std::fmt::Debug;
use std::sync::Arc;

use crate::utils::{
    add_sort_above, is_datasource, is_sort, is_sort_preserving_merge, is_union, is_window, maybe_add_sort_above
};

use arrow::datatypes::SchemaRef;
use datafusion_common::tree_node::{
    ConcreteTreeNode, Transformed, TreeNode, TreeNodeRecursion,
};
use datafusion_common::{plan_err, HashSet, JoinSide, Result};
use datafusion_expr::JoinType;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr::{EquivalenceProperties, PhysicalSortRequirement};
use datafusion_physical_expr_common::sort_expr::{LexOrdering, LexRequirement};
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::joins::utils::{
    calculate_join_output_ordering, ColumnIndex,
};
use datafusion_physical_plan::joins::{HashJoinExec, SortMergeJoinExec};
use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::source::DataSourceExec;
use datafusion_physical_plan::streaming::StreamingTableExec;
use datafusion_physical_plan::tree_node::PlanContext;
use datafusion_physical_plan::{displayable, ExecutionPlan, ExecutionPlanProperties};

/// This is a "data class" we use within the [`EnforceSorting`] rule to push
/// down [`SortExec`] in the plan. In some cases, we can reduce the total
/// computational cost by pushing down `SortExec`s through some executors. The
/// object carries the parent required ordering and the (optional) `fetch` value
/// of the parent node as its data.
///
/// [`EnforceSorting`]: crate::enforce_sorting::EnforceSorting
#[derive(Debug, Default, Clone)]
pub struct ParentRequirements {
    ordering_requirement: Option<LexRequirement>,
    fetch: Option<usize>,
}

pub type SortPushDown = PlanContext<ParentRequirements>;

/// Assigns the ordering requirement of the root node to the its children.
pub fn assign_initial_requirements(node: SortPushDown) -> SortPushDown {
    let SortPushDown {
        children,
        plan,
        data,
    } = node;
    let reqs = plan.required_input_ordering();
    let fetch = data.fetch;

    let new_children = children.into_iter().zip(reqs).map(|(child_requirement, node_order_req)| {
        let pushdown = ParentRequirements {
            ordering_requirement: node_order_req,
            fetch,
        };
        SortPushDown::new(child_requirement.plan, pushdown, child_requirement.children)
    }).collect();

    SortPushDown::new(plan, data, new_children)
}

/// Pushdown sorts using a top-down traversal.
pub fn pushdown_sorts(mut requirements: SortPushDown) -> Result<Transformed<SortPushDown>> {
    if !is_sort(&requirements.plan) {
        Ok(Transformed::no(requirements))
    } else {
        // perform sort pushdown:

        // assign sort requirements to child
        requirements = assign_initial_requirements(requirements);

        // remove the sort itself (sort has a single child)
        requirements = requirements.children.swap_remove(0);
        requirements = requirements.update_plan_from_children()?;

        // attempt pushdown
        requirements.transform_down(pushdown_sorts_helper)
    }
}

/// Pushdown sorts using a top-down traversal.
pub fn pushdown_sorts_helper(
    mut requirements: SortPushDown,
) -> Result<Transformed<SortPushDown>> {
    let plan = &requirements.plan;
    // gather the requirements assigned from the parent
    let parent_reqs = requirements
        .data
        .ordering_requirement
        .clone()
        .unwrap_or_default();

    // compare child/current plan reqs, vs parent
    let current_eq_props = plan.equivalence_properties();
    let child_satisfies_parent = current_eq_props.ordering_satisfy_requirement(&parent_reqs);
    let parent_stricter_than_child = current_eq_props.output_ordering().map(|child_ordering| current_eq_props.requirements_compatible(&parent_reqs, &child_ordering.into())).unwrap_or_default();

    // let maintains_ordering = plan.maintains_input_order().iter().all(|maintains| *maintains);
    println!("\n\npushdown_sorts_helper:\n{}\nparent_requirements={:?}\nsatisfy_parent={:?}", displayable(plan.as_ref()).indent(true), parent_reqs, child_satisfies_parent);

    if is_sort(plan) && (child_satisfies_parent || parent_stricter_than_child) {
        println!("Case 1");
        // Case 1: unify to a single sort order pushdown.
        // Outcome: carry on sort pushdown with new ordering.

        // If this `SortExec` satisfies parent requirements => use that.
        // Otherwise, use stricter parent.
        let sort_reqs = if child_satisfies_parent {
            plan.equivalence_properties().output_ordering().expect("sort exec should have output ordering")
        } else {
            parent_reqs.into()
        };

        // It's possible current plan (`SortExec`) has a fetch value.
        // And if both of them have fetch values, we should use the minimum one.
        let sort_fetch = plan.fetch();
        if let Some(fetch) = sort_fetch {
            if let Some(requirement_fetch) = requirements.data.fetch {
                requirements.data.fetch = Some(fetch.min(requirement_fetch));
            }
        }
        let fetch = requirements.data.fetch.or(sort_fetch);

        // remove current sort
        requirements = requirements.children.swap_remove(0); // sort exec has only 1 input
        requirements = requirements.update_plan_from_children()?;

        // reset requirements
        requirements.data.fetch = fetch;
        requirements.data.ordering_requirement = Some(sort_reqs.into());

        // call same function again, now with sort removed
        requirements = assign_initial_requirements(requirements);
        pushdown_sorts_helper(requirements)
    } else if is_sort(plan) {
        println!("Case 2, add sort");
        // Case 2: we have a new sort to push down.
        // Outcome: (1) add back current sort being pushed down
        // Outcome: (2) continue pushdown with new sort

        assert!(!parent_reqs.is_empty(), "should have parent ordering requirements, since no requirement would be satisfied");
        let fetch = requirements.data.fetch;
        requirements = add_sort_above(requirements, parent_reqs, fetch);
        requirements = requirements.update_plan_from_children()?;
        // on next transform_down, it will hit `Case 1` for the new sort to push

        // Reset the parent requirements on the children
        // which should be no ordering required, since is current node is a sort
        requirements = assign_initial_requirements(requirements); // given child node requirements for this sort
        Ok(Transformed::yes(requirements))
    } else if parent_reqs.is_empty() || (is_sort_preserving_merge(plan) && child_satisfies_parent) {
        println!("Case 3");
        // Case 3: we are not pushing down a sort. Continue down to find the next sort.
        requirements = assign_initial_requirements(requirements);
        Ok(Transformed::no(requirements))
    } else if let Some(adjusted_ordering_reqs) =
    pushdown_requirement_to_children(&plan, &parent_reqs)? {
            println!("Case 4");
            // Case 4: can continue current sort pushdown
            let fetch = requirements.data.fetch;
    
            // Update requirements with the children's ordering requirements (which may be stricter)
            let children = std::mem::take(&mut requirements.children);
            let new_children = children.into_iter().zip(adjusted_ordering_reqs).map(|(child_requirement, child_order_req)| {
                let pushdown = ParentRequirements {
                    ordering_requirement: child_order_req,
                    fetch,
                };
                SortPushDown::new(child_requirement.plan, pushdown, child_requirement.children)
            }).collect();
            requirements.children = new_children;
            requirements = requirements.update_plan_from_children()?;
    
            Ok(Transformed::yes(requirements))
    } else {
        println!("Case 5, add sort");
        // Case 5: we have performed the current pushdown as far as possible.
        // Outcome: add back the sort being pushed down

        assert!(!parent_reqs.is_empty(), "should have carried sort order requirements");
        let fetch = requirements.data.fetch;

        requirements = maybe_add_sort_above(requirements, parent_reqs, fetch);

        // this should set the requirements for subplan
        requirements = assign_initial_requirements(requirements);
        Ok(Transformed::yes(requirements))
    }
}

/// Calculate the pushdown ordering requirements for children.
/// 
/// If sort cannot be pushed down, return None.
/// 
/// If sort can be pushed down, return the sort requirements of the chilren
/// which may be more stringent than the parent's needs.
fn pushdown_requirement_to_children(
    plan: &Arc<dyn ExecutionPlan>,
    parent_required: &LexRequirement,
) -> Result<Option<Vec<Option<LexRequirement>>>> {
    let output_req = LexRequirement::from(
        plan.properties()
            .output_ordering()
            .cloned()
            .unwrap_or(LexOrdering::default()),
    );
    let is_satisfied = plan
        .properties()
        .eq_properties
        .ordering_satisfy_requirement(&parent_required);
    let parent_is_stricter = plan
        .properties()
        .eq_properties
        .requirements_compatible(parent_required, &output_req);
    let maintains_input_order = plan.maintains_input_order();

    println!("pushdown_requirement_to_children for {}, is_satisfied={:?}", plan.name(), is_satisfied);

    if is_window(plan) {
        let required_input_ordering = plan.required_input_ordering();
        let request_child = required_input_ordering[0].clone().unwrap_or_default();
        let child_plan = plan.children().swap_remove(0);

        match determine_children_requirement(parent_required, &request_child, child_plan)
        {
            RequirementsCompatibility::Satisfy => {
                let req = (!request_child.is_empty())
                    .then(|| LexRequirement::new(request_child.to_vec()));
                Ok(Some(vec![req]))
            }
            RequirementsCompatibility::Compatible(adjusted) => Ok(Some(vec![adjusted])),
            RequirementsCompatibility::NonCompatible => Ok(None),
        }
    } else if is_sort(&plan) {
        if parent_is_stricter {
            debug_assert!(!parent_required.is_empty());
            Ok(Some(vec![Some(LexRequirement::new(
                parent_required.to_vec(),
            ))]))
        } else {
            Ok(None)
        }
    } else if plan.fetch().is_some()
        && plan.supports_limit_pushdown()
        && maintains_input_order.iter().all(|maintain| *maintain)
    {
        // println!("A => pushdown thru {}", plan.name());
        // Push down through operator with fetch when:
        // - requirement is aligned with output ordering
        // - it preserves ordering during execution
        if is_satisfied || parent_is_stricter {
            let req = (!parent_required.is_empty())
                .then(|| LexRequirement::new(parent_required.to_vec()));
            Ok(Some(vec![req]))
        } else {
            Ok(None)
        }
    } else if is_datasource(plan) || plan.as_any().is::<StreamingTableExec>() {
        if is_satisfied {
            let req = (!parent_required.is_empty())
                .then(|| LexRequirement::new(parent_required.to_vec()));
            Ok(Some(vec![req]))
        } else {
            Ok(None)
        }
    } else if is_union(plan) {
        // UnionExec does not have real sort requirements for its input. Here we change the adjusted_request_ordering to UnionExec's output ordering and
        // propagate the sort requirements down to correct the unnecessary descendant SortExec under the UnionExec
        let req = (!parent_required.is_empty()).then(|| parent_required.clone());
        Ok(Some(vec![req; plan.children().len()]))
    } else if let Some(smj) = plan.as_any().downcast_ref::<SortMergeJoinExec>() {
        // If the current plan is SortMergeJoinExec
        let left_columns_len = smj.left().schema().fields().len();
        let parent_required_expr = LexOrdering::from(parent_required.clone());
        match expr_source_side(
            parent_required_expr.as_ref(),
            smj.join_type(),
            left_columns_len,
        ) {
            Some(JoinSide::Left) => try_pushdown_requirements_to_join(
                smj,
                parent_required,
                parent_required_expr.as_ref(),
                JoinSide::Left,
            ),
            Some(JoinSide::Right) => {
                let right_offset =
                    smj.schema().fields.len() - smj.right().schema().fields.len();
                let new_right_required =
                    shift_right_required(parent_required, right_offset)?;
                let new_right_required_expr = LexOrdering::from(new_right_required);
                try_pushdown_requirements_to_join(
                    smj,
                    parent_required,
                    new_right_required_expr.as_ref(),
                    JoinSide::Right,
                )
            }
            _ => {
                // Can not decide the expr side for SortMergeJoinExec, can not push down
                Ok(None)
            }
        }
    } else if maintains_input_order.is_empty()
        || !maintains_input_order.iter().any(|o| *o)
        || plan.as_any().is::<RepartitionExec>()
        || plan.as_any().is::<FilterExec>()
        // TODO: Add support for Projection push down
        || plan.as_any().is::<ProjectionExec>()
        || pushdown_would_violate_requirements(parent_required, plan.as_ref())
    {
        println!("B => don't pushdown {}", plan.name());
        // If the current plan is a leaf node or can not maintain any of the input ordering, can not pushed down requirements.
        // For RepartitionExec, we always choose to not push down the sort requirements even the RepartitionExec(input_partition=1) could maintain input ordering.
        // Pushing down is not beneficial
        Ok(None)
    } else if is_sort_preserving_merge(plan) {
        let new_ordering = LexOrdering::from(parent_required.clone());
        let mut spm_eqs = plan.equivalence_properties().clone();
        // Sort preserving merge will have new ordering, one requirement above is pushed down to its below.
        spm_eqs = spm_eqs.with_reorder(new_ordering);
        // Do not push-down through SortPreservingMergeExec when
        // ordering requirement invalidates requirement of sort preserving merge exec.
        if !spm_eqs.ordering_satisfy(&plan.output_ordering().cloned().unwrap_or_default())
        {
            Ok(None)
        } else {
            // Can push-down through SortPreservingMergeExec, because parent requirement is finer
            // than SortPreservingMergeExec output ordering.
            let req = (!parent_required.is_empty())
                .then(|| LexRequirement::new(parent_required.to_vec()));
            Ok(Some(vec![req]))
        }
    } else if let Some(hash_join) = plan.as_any().downcast_ref::<HashJoinExec>() {
        handle_hash_join(hash_join, parent_required)
    } else {
        println!("C => custom pushdown thru {}", plan.name());
        handle_custom_pushdown(plan, parent_required, maintains_input_order)
    }
    // TODO: Add support for Projection push down
}

/// Return true if pushing the sort requirements through a node would violate
/// the input sorting requirements for the plan
fn pushdown_would_violate_requirements(
    parent_required: &LexRequirement,
    child: &dyn ExecutionPlan,
) -> bool {
    child
        .required_input_ordering()
        .iter()
        .any(|child_required| {
            let Some(child_required) = child_required.as_ref() else {
                // no requirements, so pushing down would not violate anything
                return false;
            };
            // check if the plan's requirements would still e satisfied if we pushed
            // down the parent requirements
            child_required
                .iter()
                .zip(parent_required.iter())
                .all(|(c, p)| !c.compatible(p))
        })
}

/// Determine children requirements:
/// - If children requirements are more specific, do not push down parent
///   requirements.
/// - If parent requirements are more specific, push down parent requirements.
/// - If they are not compatible, need to add a sort.
fn determine_children_requirement(
    parent_required: &LexRequirement,
    request_child: &LexRequirement,
    child_plan: &Arc<dyn ExecutionPlan>,
) -> RequirementsCompatibility {
    if child_plan
        .equivalence_properties()
        .requirements_compatible(request_child, parent_required)
    {
        // Child requirements are more specific, no need to push down.
        RequirementsCompatibility::Satisfy
    } else if child_plan
        .equivalence_properties()
        .requirements_compatible(parent_required, request_child)
    {
        // Parent requirements are more specific, adjust child's requirements
        // and push down the new requirements:
        let adjusted = (!parent_required.is_empty())
            .then(|| LexRequirement::new(parent_required.to_vec()));
        RequirementsCompatibility::Compatible(adjusted)
    } else {
        RequirementsCompatibility::NonCompatible
    }
}

fn try_pushdown_requirements_to_join(
    smj: &SortMergeJoinExec,
    parent_required: &LexRequirement,
    sort_expr: &LexOrdering,
    push_side: JoinSide,
) -> Result<Option<Vec<Option<LexRequirement>>>> {
    let left_eq_properties = smj.left().equivalence_properties();
    let right_eq_properties = smj.right().equivalence_properties();
    let mut smj_required_orderings = smj.required_input_ordering();
    let right_requirement = smj_required_orderings.swap_remove(1);
    let left_requirement = smj_required_orderings.swap_remove(0);
    let left_ordering = &smj.left().output_ordering().cloned().unwrap_or_default();
    let right_ordering = &smj.right().output_ordering().cloned().unwrap_or_default();

    let (new_left_ordering, new_right_ordering) = match push_side {
        JoinSide::Left => {
            let left_eq_properties =
                left_eq_properties.clone().with_reorder(sort_expr.clone());
            if left_eq_properties
                .ordering_satisfy_requirement(&left_requirement.unwrap_or_default())
            {
                // After re-ordering requirement is still satisfied
                (sort_expr, right_ordering)
            } else {
                return Ok(None);
            }
        }
        JoinSide::Right => {
            let right_eq_properties =
                right_eq_properties.clone().with_reorder(sort_expr.clone());
            if right_eq_properties
                .ordering_satisfy_requirement(&right_requirement.unwrap_or_default())
            {
                // After re-ordering requirement is still satisfied
                (left_ordering, sort_expr)
            } else {
                return Ok(None);
            }
        }
        JoinSide::None => return Ok(None),
    };
    let join_type = smj.join_type();
    let probe_side = SortMergeJoinExec::probe_side(&join_type);
    let new_output_ordering = calculate_join_output_ordering(
        new_left_ordering,
        new_right_ordering,
        join_type,
        smj.on(),
        smj.left().schema().fields.len(),
        &smj.maintains_input_order(),
        Some(probe_side),
    );
    let mut smj_eqs = smj.properties().equivalence_properties().clone();
    // smj will have this ordering when its input changes.
    smj_eqs = smj_eqs.with_reorder(new_output_ordering.unwrap_or_default());
    let should_pushdown = smj_eqs.ordering_satisfy_requirement(parent_required);
    Ok(should_pushdown.then(|| {
        let mut required_input_ordering = smj.required_input_ordering();
        let new_req = Some(LexRequirement::from(sort_expr.clone()));
        match push_side {
            JoinSide::Left => {
                required_input_ordering[0] = new_req;
            }
            JoinSide::Right => {
                required_input_ordering[1] = new_req;
            }
            JoinSide::None => unreachable!(),
        }
        required_input_ordering
    }))
}

fn expr_source_side(
    required_exprs: &LexOrdering,
    join_type: JoinType,
    left_columns_len: usize,
) -> Option<JoinSide> {
    match join_type {
        JoinType::Inner
        | JoinType::Left
        | JoinType::Right
        | JoinType::Full
        | JoinType::LeftMark => {
            let all_column_sides = required_exprs
                .iter()
                .filter_map(|r| {
                    r.expr.as_any().downcast_ref::<Column>().map(|col| {
                        if col.index() < left_columns_len {
                            JoinSide::Left
                        } else {
                            JoinSide::Right
                        }
                    })
                })
                .collect::<Vec<_>>();

            // If the exprs are all coming from one side, the requirements can be pushed down
            if all_column_sides.len() != required_exprs.len() {
                None
            } else if all_column_sides
                .iter()
                .all(|side| matches!(side, JoinSide::Left))
            {
                Some(JoinSide::Left)
            } else if all_column_sides
                .iter()
                .all(|side| matches!(side, JoinSide::Right))
            {
                Some(JoinSide::Right)
            } else {
                None
            }
        }
        JoinType::LeftSemi | JoinType::LeftAnti => required_exprs
            .iter()
            .all(|e| e.expr.as_any().downcast_ref::<Column>().is_some())
            .then_some(JoinSide::Left),
        JoinType::RightSemi | JoinType::RightAnti => required_exprs
            .iter()
            .all(|e| e.expr.as_any().downcast_ref::<Column>().is_some())
            .then_some(JoinSide::Right),
    }
}

fn shift_right_required(
    parent_required: &LexRequirement,
    left_columns_len: usize,
) -> Result<LexRequirement> {
    let new_right_required = parent_required
        .iter()
        .filter_map(|r| {
            let col = r.expr.as_any().downcast_ref::<Column>()?;
            col.index().checked_sub(left_columns_len).map(|offset| {
                r.clone()
                    .with_expr(Arc::new(Column::new(col.name(), offset)))
            })
        })
        .collect::<Vec<_>>();
    if new_right_required.len() == parent_required.len() {
        Ok(LexRequirement::new(new_right_required))
    } else {
        plan_err!(
            "Expect to shift all the parent required column indexes for SortMergeJoin"
        )
    }
}

/// Handles the custom pushdown of parent-required sorting requirements down to
/// the child execution plans, considering whether the input order is maintained.
///
/// # Arguments
///
/// * `plan` - A reference to an `ExecutionPlan` for which the pushdown will be applied.
/// * `parent_required` - The sorting requirements expected by the parent node.
/// * `maintains_input_order` - A vector of booleans indicating whether each child
///   maintains the input order.
///
/// # Returns
///
/// Returns `Ok(Some(Vec<Option<LexRequirement>>))` if the sorting requirements can be
/// pushed down, `Ok(None)` if not. On error, returns a `Result::Err`.
fn handle_custom_pushdown(
    plan: &Arc<dyn ExecutionPlan>,
    parent_required: &LexRequirement,
    maintains_input_order: Vec<bool>,
) -> Result<Option<Vec<Option<LexRequirement>>>> {
    // If there's no requirement from the parent or the plan has no children, return early
    if parent_required.is_empty() || plan.children().is_empty() {
        return Ok(None);
    }

    // Collect all unique column indices used in the parent-required sorting expression
    let all_indices: HashSet<usize> = parent_required
        .iter()
        .flat_map(|order| {
            collect_columns(&order.expr)
                .iter()
                .map(|col| col.index())
                .collect::<HashSet<_>>()
        })
        .collect();

    // Get the number of fields in each child's schema
    let len_of_child_schemas: Vec<usize> = plan
        .children()
        .iter()
        .map(|c| c.schema().fields().len())
        .collect();

    // Find the index of the child that maintains input order
    let Some(maintained_child_idx) = maintains_input_order
        .iter()
        .enumerate()
        .find(|(_, m)| **m)
        .map(|pair| pair.0)
    else {
        return Ok(None);
    };

    // Check if all required columns come from the child that maintains input order
    let start_idx = len_of_child_schemas[..maintained_child_idx]
        .iter()
        .sum::<usize>();
    let end_idx = start_idx + len_of_child_schemas[maintained_child_idx];
    let all_from_maintained_child =
        all_indices.iter().all(|i| i >= &start_idx && i < &end_idx);

    // If all columns are from the maintained child, update the parent requirements
    if all_from_maintained_child {
        let sub_offset = len_of_child_schemas
            .iter()
            .take(maintained_child_idx)
            .sum::<usize>();
        // Transform the parent-required expression for the child schema by adjusting columns
        let updated_parent_req = parent_required
            .iter()
            .map(|req| {
                let child_schema = plan.children()[maintained_child_idx].schema();
                let updated_columns = Arc::clone(&req.expr)
                    .transform_up(|expr| {
                        if let Some(col) = expr.as_any().downcast_ref::<Column>() {
                            let new_index = col.index() - sub_offset;
                            Ok(Transformed::yes(Arc::new(Column::new(
                                child_schema.field(new_index).name(),
                                new_index,
                            ))))
                        } else {
                            Ok(Transformed::no(expr))
                        }
                    })?
                    .data;
                Ok(PhysicalSortRequirement::new(updated_columns, req.options))
            })
            .collect::<Result<Vec<_>>>()?;

        // Prepare the result, populating with the updated requirements for children that maintain order
        let result = maintains_input_order
            .iter()
            .map(|&maintains_order| {
                if maintains_order {
                    Some(LexRequirement::new(updated_parent_req.clone()))
                } else {
                    None
                }
            })
            .collect();

        Ok(Some(result))
    } else {
        Ok(None)
    }
}

// For hash join we only maintain the input order for the right child
// for join type: Inner, Right, RightSemi, RightAnti
fn handle_hash_join(
    plan: &HashJoinExec,
    parent_required: &LexRequirement,
) -> Result<Option<Vec<Option<LexRequirement>>>> {
    // If there's no requirement from the parent or the plan has no children
    // or the join type is not Inner, Right, RightSemi, RightAnti, return early
    if parent_required.is_empty() || !plan.maintains_input_order()[1] {
        return Ok(None);
    }

    // Collect all unique column indices used in the parent-required sorting expression
    let all_indices: HashSet<usize> = parent_required
        .iter()
        .flat_map(|order| {
            collect_columns(&order.expr)
                .into_iter()
                .map(|col| col.index())
                .collect::<HashSet<_>>()
        })
        .collect();

    let column_indices = build_join_column_index(plan);
    let projected_indices: Vec<_> = if let Some(projection) = &plan.projection {
        projection.iter().map(|&i| &column_indices[i]).collect()
    } else {
        column_indices.iter().collect()
    };
    let len_of_left_fields = projected_indices
        .iter()
        .filter(|ci| ci.side == JoinSide::Left)
        .count();

    let all_from_right_child = all_indices.iter().all(|i| *i >= len_of_left_fields);

    // If all columns are from the right child, update the parent requirements
    if all_from_right_child {
        // Transform the parent-required expression for the child schema by adjusting columns
        let updated_parent_req = parent_required
            .iter()
            .map(|req| {
                let child_schema = plan.children()[1].schema();
                let updated_columns = Arc::clone(&req.expr)
                    .transform_up(|expr| {
                        if let Some(col) = expr.as_any().downcast_ref::<Column>() {
                            let index = projected_indices[col.index()].index;
                            Ok(Transformed::yes(Arc::new(Column::new(
                                child_schema.field(index).name(),
                                index,
                            ))))
                        } else {
                            Ok(Transformed::no(expr))
                        }
                    })?
                    .data;
                Ok(PhysicalSortRequirement::new(updated_columns, req.options))
            })
            .collect::<Result<Vec<_>>>()?;

        // Populating with the updated requirements for children that maintain order
        Ok(Some(vec![
            None,
            Some(LexRequirement::new(updated_parent_req)),
        ]))
    } else {
        Ok(None)
    }
}

// this function is used to build the column index for the hash join
// push down sort requirements to the right child
fn build_join_column_index(plan: &HashJoinExec) -> Vec<ColumnIndex> {
    let map_fields = |schema: SchemaRef, side: JoinSide| {
        schema
            .fields()
            .iter()
            .enumerate()
            .map(|(index, _)| ColumnIndex { index, side })
            .collect::<Vec<_>>()
    };

    match plan.join_type() {
        JoinType::Inner | JoinType::Right => {
            map_fields(plan.left().schema(), JoinSide::Left)
                .into_iter()
                .chain(map_fields(plan.right().schema(), JoinSide::Right))
                .collect::<Vec<_>>()
        }
        JoinType::RightSemi | JoinType::RightAnti => {
            map_fields(plan.right().schema(), JoinSide::Right)
        }
        _ => unreachable!("unexpected join type: {}", plan.join_type()),
    }
}

/// Define the Requirements Compatibility
#[derive(Debug)]
enum RequirementsCompatibility {
    /// Requirements satisfy
    Satisfy,
    /// Requirements compatible
    Compatible(Option<LexRequirement>),
    /// Requirements not compatible
    NonCompatible,
}
