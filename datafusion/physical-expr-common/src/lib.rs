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

//! Physical Expr Common packages for [DataFusion]
//! This package contains high level PhysicalExpr trait
//!
//! [DataFusion]: <https://crates.io/crates/datafusion>

// Disable clippy lints that were introduced after this code was written
#![allow(clippy::needless_return)]
#![allow(clippy::needless_lifetimes)]
#![allow(clippy::unnecessary_lazy_evaluations)]
#![allow(clippy::empty_line_after_doc_comments)]

pub mod binary_map;
pub mod binary_view_map;
pub mod datum;
pub mod physical_expr;
pub mod sort_expr;
pub mod tree_node;
pub mod utils;
