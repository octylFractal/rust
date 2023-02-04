//! Provides the implementation of the `derived_clone` attribute.
//!
//! This attribute is used to implement a derived `Clone` implementation for a struct. It is delayed
//! from normal derive expansion because it requires full type information, which is not available
//! during derive expansion.

use rustc_ast::Attribute;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::HirId;
use rustc_index::vec::IndexVec;
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_infer::traits::{FulfillmentError, ObligationCause, ObligationCauseCode};
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_middle::{
    mir::*,
    thir::*,
    ty::{self, util::IntTypeExt},
};
use rustc_span::{sym, Span};
use rustc_target::abi::VariantIdx;
use rustc_trait_selection::traits::error_reporting::TypeErrCtxtExt;
use rustc_trait_selection::traits::fully_normalize;

const SELF_LOCAL: Local = Local::from_u32(1);

struct CloneMirBuilder<'tcx> {
    tcx: TyCtxt<'tcx>,
    infcx: InferCtxt<'tcx>,
    body_hir_id: HirId,
    span: Span,
    param_env: ty::ParamEnv<'tcx>,
    adt: ty::AdtDef<'tcx>,
    substs: ty::SubstsRef<'tcx>,
    body: Body<'tcx>,
}

pub fn build_clone_mir<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_def: ty::WithOptConstParam<LocalDefId>,
    hir_id: HirId,
    body_hir_id: HirId,
    params: &IndexVec<ParamId, Param<'tcx>>,
    return_ty: Ty<'tcx>,
    return_ty_span: Span,
    span: Span,
    attr: &Attribute,
) -> Body<'tcx> {
    let param_env = tcx.param_env(fn_def.did);
    let mut body = Body {
        basic_blocks: BasicBlocks::new(IndexVec::new()),
        source: MirSource::item(fn_def.did.to_def_id()),
        phase: MirPhase::Built,
        source_scopes: IndexVec::new(),
        generator: None,
        local_decls: LocalDecls::new(),
        user_type_annotations: IndexVec::new(),
        arg_count: params.len(),
        spread_arg: None,
        var_debug_info: Vec::new(),
        span,
        required_consts: Vec::new(),
        is_polymorphic: false,
        tainted_by_errors: None,
        injection_phase: None,
        pass_count: 0,
    };
    body.source_scopes.push(SourceScopeData {
        span,
        parent_scope: None,
        inlined: None,
        inlined_parent_scope: None,
        local_data: ClearCrossCrate::Set(SourceScopeLocalData {
            lint_root: hir_id,
            safety: Safety::Safe,
        }),
    });

    let self_param = &params[ParamId::from_u32(0)];
    if let Some(ref pat) = self_param.pat && let Some(name) = pat.simple_ident() {
        let source_info =
        SourceInfo::outermost(self_param.pat.as_ref().map_or(span, |pat| pat.span));
        body.var_debug_info.push(VarDebugInfo {
            name,
            source_info,
            value: VarDebugInfoContents::Place(Place::from(Local::from_u32(1))),
        });
    }

    // insert return and arg local
    body.local_decls.push(LocalDecl::new(return_ty, return_ty_span));
    body.local_decls.push(LocalDecl::new(self_param.ty, self_param.ty_span.unwrap_or(span)));

    let infcx = tcx.infer_ctxt().build();

    let &ty::Adt(adt, substs) = return_ty .kind() else {
        span_bug!(span, "clone derive should only be on ADTs, not {:?}", return_ty);
    };
    let substs = tcx.erase_regions(substs);
    let mut clone_mir_builder =
        CloneMirBuilder { tcx, infcx, body_hir_id, span, param_env, adt, substs, body };

    if clone_mir_builder.test_for_copy(attr, return_ty) {
        let mut block = BasicBlockData::new(Some(Terminator {
            source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: TerminatorKind::Return,
        }));
        // Deref first param (_1) into return (_0)
        block.statements.push(Statement {
            source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: StatementKind::Assign(Box::new((
                Place::return_place(),
                Rvalue::Use(Operand::Copy(Place {
                    local: SELF_LOCAL,
                    projection: tcx.intern_place_elems(&[ProjectionElem::Deref]),
                })),
            ))),
        });
        clone_mir_builder.body.basic_blocks_mut().push(block);
    } else {
        // We need to build calls to `clone` for each field.
        // First, we need to determine if we're working with a struct or an enum.
        // If it's an enum, we need to build a MIR switch to handle each variant.
        // If it's a struct, we can just build a call to `clone` for each field.
        let result = match adt.adt_kind() {
            ty::AdtKind::Struct => clone_mir_builder.build_struct_clone_mir(),
            ty::AdtKind::Enum => clone_mir_builder.build_enum_clone_mir(),

            ty::AdtKind::Union => {
                let err = tcx.sess.span_err(span, "cannot derive `Clone` for non-`Copy` unions");
                clone_mir_builder.body.tainted_by_errors = Some(err);
                Ok(())
            }
        };
        if let Err(fulfillment_errors) = result {
            let err = clone_mir_builder.infcx.err_ctxt().report_fulfillment_errors(
                &fulfillment_errors,
                Some(rustc_hir::BodyId { hir_id: body_hir_id }),
            );
            clone_mir_builder.body.tainted_by_errors = Some(err);
        }
    }

    clone_mir_builder.body
}

struct VariantCloneBlocks {
    /// The first block of the variant clone
    first_block: BasicBlock,
    /// The return block of the variant clone
    return_block: BasicBlock,
}

impl<'tcx> CloneMirBuilder<'tcx> {
    fn cause(&self) -> ObligationCause<'tcx> {
        ObligationCause::new(
            self.span,
            self.body_hir_id.owner.def_id,
            ObligationCauseCode::MiscObligation,
        )
    }

    /// Test for `ty: Copy`, conservatively returning `false` if unbound regions are encountered.
    fn test_for_copy(&self, attr: &Attribute, ty: Ty<'tcx>) -> bool {
        if let Some(meta_items) = attr.meta_item_list() && meta_items.iter().any(|mi| mi.has_name(sym::copy)) {
            return true;
        }
        ty.is_copy_considering_regions(self.tcx, self.param_env)
    }

    fn build_struct_clone_mir(&mut self) -> Result<(), Vec<FulfillmentError<'tcx>>> {
        let [variant] = &self.adt.variants().raw[..] else {
            bug!("struct did not have exactly one variant: {:?}", self.adt);
        };
        self.build_variant_clone_mir(
            VariantIdx::from_u32(0),
            variant,
            Place {
                local: SELF_LOCAL,
                projection: self.tcx.intern_place_elems(&[ProjectionElem::Deref]),
            },
        )
        .map(|_| ())
    }

    fn build_enum_clone_mir(&mut self) -> Result<(), Vec<FulfillmentError<'tcx>>> {
        // push SwitchInt block + discr local early
        let switch_int_block_idx = self.body.basic_blocks_mut().push(BasicBlockData::new(None));
        let discr_local = self
            .body
            .local_decls
            .push(LocalDecl::new(self.adt.repr().discr_type().to_ty(self.tcx), self.span));
        let return_block_idx =
            self.body.basic_blocks_mut().push(BasicBlockData::new(Some(Terminator {
                source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
                kind: TerminatorKind::Return,
            })));
        let mut switch_blocks = self
            .adt
            .discriminants(self.tcx)
            .map(|(v_idx, disc)| {
                let variant = &self.adt.variants()[v_idx];
                let VariantCloneBlocks { first_block, return_block } = self
                    .build_variant_clone_mir(
                        v_idx,
                        variant,
                        Place {
                            local: SELF_LOCAL,
                            projection: self.tcx.intern_place_elems(&[
                                ProjectionElem::Deref,
                                ProjectionElem::Downcast(Some(variant.name), v_idx),
                            ]),
                        },
                    )?;

                self.body.basic_blocks_mut()[return_block].terminator = Some(Terminator {
                    source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
                    kind: TerminatorKind::Goto { target: return_block_idx },
                });

                Ok::<_, Vec<FulfillmentError<'tcx>>>((disc.val, first_block))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if switch_blocks.len() >= 2 {
            // Need to add FalseEdges for borrowck, I guess
            // We'll replace every switch target block except the second to last one with a new block that
            // FalseEdges#imaginary to the next FalseEdge block.

            let second_to_last_edge = switch_blocks.len() - 2;
            for elem in &mut switch_blocks[0..second_to_last_edge] {
                let block_idx = self.body.basic_blocks_mut().push(BasicBlockData::new(None));
                self.body.basic_blocks_mut()[block_idx].terminator = Some(Terminator {
                    source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
                    kind: TerminatorKind::FalseEdge {
                        real_target: elem.1,
                        imaginary_target: BasicBlock::from(u32::from(block_idx) + 1),
                    },
                });
                elem.1 = block_idx;
            }
            // and the last block is imaginary to the last real switch block.
            let idx = self.body.basic_blocks_mut().push(BasicBlockData::new(Some(Terminator {
                source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
                kind: TerminatorKind::FalseEdge {
                    real_target: switch_blocks[second_to_last_edge].1,
                    imaginary_target: switch_blocks[second_to_last_edge + 1].1,
                },
            })));
            switch_blocks[second_to_last_edge].1 = idx;
        }
        // push unreachable block
        let unreachable_block_idx =
            self.body.basic_blocks_mut().push(BasicBlockData::new(Some(Terminator {
                source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
                kind: TerminatorKind::Unreachable,
            })));

        let mut switch_block = &mut self.body.basic_blocks_mut()[switch_int_block_idx];
        switch_block.statements.push(Statement {
            source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: StatementKind::FakeRead(Box::new((
                FakeReadCause::ForMatchedPlace(None),
                Place::from(SELF_LOCAL),
            ))),
        });
        switch_block.statements.push(Statement {
            source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: StatementKind::Assign(Box::new((
                Place::from(discr_local),
                Rvalue::Discriminant(Place {
                    local: SELF_LOCAL,
                    projection: self.tcx.intern_place_elems(&[ProjectionElem::Deref]),
                }),
            ))),
        });
        switch_block.terminator = Some(Terminator {
            source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: TerminatorKind::SwitchInt {
                discr: Operand::Copy(Place::from(discr_local)),
                targets: SwitchTargets::new(switch_blocks.into_iter(), unreachable_block_idx),
            },
        });

        Ok(())
    }

    /// Builds the MIR for cloning a single variant. Used for both structs and enums.
    ///
    /// Returns the first block, for ease of use in the enum builder.
    fn build_variant_clone_mir(
        &mut self,
        variant_idx: VariantIdx,
        variant: &'tcx ty::VariantDef,
        variant_place: Place<'tcx>,
    ) -> Result<VariantCloneBlocks, Vec<FulfillmentError<'tcx>>> {
        let block_base = self.body.basic_blocks.len();
        let field_cnt = variant.fields.len();
        // vec of locals for the cloned fields
        let mut clone_locals = Vec::with_capacity(field_cnt);
        let mut last_ref_local = None;
        for (idx, f) in variant.fields.iter().enumerate() {
            let field_ty = fully_normalize(
                &self.infcx,
                self.cause(),
                self.param_env,
                f.ty(self.tcx, self.substs),
            )?;
            let erased_field_ty = self.tcx.erase_regions(field_ty);
            // allocate slot for the field reference
            let ref_local = self.body.local_decls.push(LocalDecl::new(
                self.tcx.mk_imm_ref(self.tcx.mk_region(ty::RegionKind::ReErased), erased_field_ty),
                self.span,
            ));
            // allocate slot for the field clone
            let clone_local =
                self.body.local_decls.push(LocalDecl::new(erased_field_ty, self.span));
            clone_locals.push(clone_local);
            // get our field_ty's clone function
            let typeck_results = self.tcx.typeck(self.body_hir_id.owner.def_id);
            let Some(&clone_fn) = typeck_results.clone_fns.get(&f.did.expect_local()) else {
                span_bug!(self.span, "no `clone` method found for `{}`", field_ty)
            };
            // push block with the clone
            let mut block = BasicBlockData::new(Some(Terminator {
                source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
                kind: TerminatorKind::Call {
                    func: Operand::Constant(Box::new(Constant {
                        span: self.span,
                        user_ty: None,
                        literal: ConstantKind::Val(
                            interpret::ConstValue::ZeroSized,
                            self.tcx.erase_regions(self.tcx.mk_fn_def(clone_fn, [erased_field_ty])),
                        ),
                    })),
                    args: vec![Operand::Move(Place::from(ref_local))],
                    destination: Place::from(clone_local),
                    target: Some(BasicBlock::from(block_base + idx + 1)),
                    cleanup: (idx > 0).then(|| BasicBlock::from(block_base + 2 * field_cnt - idx)),
                    from_hir_call: false,
                    fn_span: self.span,
                },
            }));
            // Dead the previous block's ref_local
            if let Some(last_ref_local) = last_ref_local {
                block.statements.push(Statement {
                    source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
                    kind: StatementKind::StorageDead(last_ref_local),
                });
            }
            last_ref_local = Some(ref_local);
            // Need to Live our locals
            block.statements.push(Statement {
                source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
                kind: StatementKind::StorageLive(ref_local),
            });
            block.statements.push(Statement {
                source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
                kind: StatementKind::StorageLive(clone_local),
            });
            block.statements.push(Statement {
                source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
                kind: StatementKind::Assign(Box::new((
                    Place::from(ref_local),
                    Rvalue::Ref(
                        self.tcx.mk_region(ty::RegionKind::ReErased),
                        BorrowKind::Shared,
                        variant_place.project_deeper(
                            &[ProjectionElem::Field(Field::from(idx), erased_field_ty)],
                            self.tcx,
                        ),
                    ),
                ))),
            });
            self.body.basic_blocks_mut().push(block);
        }
        // now that the fields are cloned, we build the final result
        let mut block = BasicBlockData::new(Some(Terminator {
            source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: TerminatorKind::Return,
        }));
        // Dead the last ref_local
        if let Some(last_ref_local) = last_ref_local {
            block.statements.push(Statement {
                source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
                kind: StatementKind::StorageDead(last_ref_local),
            });
        }
        let operands = clone_locals
            .iter()
            .map(|&clone_local| Operand::Move(Place::from(clone_local)))
            .collect();
        // Write the struct
        block.statements.push(Statement {
            source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: StatementKind::Assign(Box::new((
                Place::return_place(),
                Rvalue::Aggregate(
                    Box::new(AggregateKind::Adt(
                        self.adt.did(),
                        variant_idx,
                        self.substs,
                        None,
                        None,
                    )),
                    operands,
                ),
            ))),
        });
        // Dead the clone locals
        for &clone_local in &clone_locals {
            block.statements.push(Statement {
                source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
                kind: StatementKind::StorageDead(clone_local),
            });
        }
        let return_block = self.body.basic_blocks_mut().push(block);
        // and finally add cleanup blocks (skipping one, since we don't need cleanup for the last field)
        for idx in 1..field_cnt {
            let cleanup_block_idx = self.body.basic_blocks_mut().push(BasicBlockData::new(None));
            let mut cleanup_block = &mut self.body.basic_blocks_mut()[cleanup_block_idx];
            cleanup_block.terminator = Some(Terminator {
                source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
                kind: TerminatorKind::Drop {
                    place: Place::from(clone_locals[field_cnt - idx - 1]),
                    target: BasicBlock::from(cleanup_block_idx.as_u32() + 1),
                    unwind: None,
                },
            });
            cleanup_block.is_cleanup = true;
        }
        // and the last block is the panic resume block that the last cleanup block points to (if there are any)
        if field_cnt > 1 {
            let mut block = BasicBlockData::new(Some(Terminator {
                source_info: SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE },
                kind: TerminatorKind::Resume,
            }));
            block.is_cleanup = true;
            self.body.basic_blocks_mut().push(block);
        }
        Ok(VariantCloneBlocks { first_block: BasicBlock::from(block_base), return_block })
    }
}
