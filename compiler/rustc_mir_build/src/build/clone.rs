//! Provides the implementation of the `derived_clone` attribute.
//!
//! This attribute is used to implement a derived `Clone` implementation for a struct. It is delayed
//! from normal derive expansion because it requires full type information, which is not available
//! during derive expansion.

use rustc_hir::def_id::DefId;
use rustc_hir::HirId;
use rustc_hir_typeck::{FnCtxt, Inherited};
use rustc_index::vec::IndexVec;
use rustc_infer::traits::ObligationCauseCode;
use rustc_middle::{
    mir::{interpret::ConstValue, *},
    thir::*,
    ty::{self, util::IntTypeExt, AdtDef, AdtKind, RegionKind, SubstsRef, Ty, TyCtxt},
};
use rustc_span::symbol::Ident;
use rustc_span::{sym, Span};
use rustc_target::abi::VariantIdx;

const SELF_LOCAL: Local = Local::from_u32(1);

pub fn build_clone_mir<'tcx>(
    tcx: TyCtxt<'tcx>,
    did: DefId,
    hir_id: HirId,
    body_hir_id: HirId,
    _thir: &Thir<'tcx>,
    _expr: ExprId,
    params: &IndexVec<ParamId, Param<'tcx>>,
    return_ty: Ty<'tcx>,
    return_ty_span: Span,
    span: Span,
) -> Body<'tcx> {
    let param_env = tcx.param_env(did);
    let mut body = Body {
        basic_blocks: BasicBlocks::new(IndexVec::new()),
        source: MirSource::item(did),
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

    // Ask tcx if it's copy, if so we have trivial MIR:
    if return_ty.is_copy_modulo_regions(tcx, param_env) {
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
        body.basic_blocks_mut().push(block);
    } else {
        // We need to build calls to `clone` for each field.
        Inherited::build(tcx, did.expect_local()).enter(|inh| {
            let fn_ctxt = FnCtxt::new(inh, param_env, body_hir_id.owner.def_id);
            let Some(clone_did) = tcx.lang_items().clone_trait() else {
                let err = tcx.sess.span_err(span, "cannot find `Clone` lang item");
                body.tainted_by_errors = Some(err);
                return;
            };
            // First, we need to determine if we're working with a struct or an enum.
            // If it's an enum, we need to build a MIR switch to handle each variant.
            // If it's a struct, we can just build a call to `clone` for each field.
            let &ty::Adt(adt, substs) = return_ty .kind() else {
                bug!("clone derive should only be on ADTs, not {:?}", return_ty);
            };
            let substs = tcx.erase_regions(substs);
            match adt.adt_kind() {
                AdtKind::Struct => {
                    build_struct_clone_mir(tcx, &mut body, span, adt, substs, &fn_ctxt, clone_did);
                }
                AdtKind::Enum => {
                    build_enum_clone_mir(tcx, &mut body, span, adt, substs, &fn_ctxt, clone_did);
                }
                AdtKind::Union => {
                    let err =
                        tcx.sess.span_err(span, "cannot derive `Clone` for non-`Copy` unions");
                    body.tainted_by_errors = Some(err);
                }
            }
        });
    }

    body
}

fn build_struct_clone_mir<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    span: Span,
    adt: AdtDef<'tcx>,
    substs: SubstsRef<'tcx>,
    fn_ctxt: &FnCtxt<'a, 'tcx>,
    clone_did: DefId,
) {
    let [variant] = &adt.variants().raw[..] else {
        bug!("struct did not have exactly one variant: {:?}", adt);
    };
    build_variant_clone_mir(
        tcx,
        body,
        span,
        adt,
        substs,
        fn_ctxt,
        clone_did,
        VariantIdx::from_u32(0),
        variant,
        Place { local: SELF_LOCAL, projection: tcx.intern_place_elems(&[ProjectionElem::Deref]) },
    );
}

fn build_enum_clone_mir<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    span: Span,
    adt: AdtDef<'tcx>,
    substs: SubstsRef<'tcx>,
    fn_ctxt: &FnCtxt<'a, 'tcx>,
    clone_did: DefId,
) {
    // push SwitchInt block + discr local early
    let switch_int_block_idx = body.basic_blocks_mut().push(BasicBlockData::new(None));
    let discr_local =
        body.local_decls.push(LocalDecl::new(adt.repr().discr_type().to_ty(tcx), span));
    let return_block_idx = body.basic_blocks_mut().push(BasicBlockData::new(Some(Terminator {
        source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
        kind: TerminatorKind::Return,
    })));
    let mut switch_blocks = adt
        .discriminants(tcx)
        .map(|(v_idx, disc)| {
            let variant = &adt.variants()[v_idx];
            let VariantCloneBlocks { first_block, return_block } = build_variant_clone_mir(
                tcx,
                body,
                span,
                adt,
                substs,
                fn_ctxt,
                clone_did,
                v_idx,
                variant,
                Place {
                    local: SELF_LOCAL,
                    projection: tcx.intern_place_elems(&[
                        ProjectionElem::Deref,
                        ProjectionElem::Downcast(Some(variant.name), v_idx),
                    ]),
                },
            );

            body.basic_blocks_mut()[return_block].terminator = Some(Terminator {
                source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
                kind: TerminatorKind::Goto { target: return_block_idx },
            });

            (disc.val, first_block)
        })
        .collect::<Vec<_>>();
    if switch_blocks.len() >= 2 {
        // Need to add FalseEdges for borrowck, I guess
        // We'll replace every switch target block except the second to last one with a new block that
        // FalseEdges#imaginary to the next FalseEdge block.

        let second_to_last_edge = switch_blocks.len() - 2;
        for elem in &mut switch_blocks[0..second_to_last_edge] {
            let block_idx = body.basic_blocks_mut().push(BasicBlockData::new(None));
            body.basic_blocks_mut()[block_idx].terminator = Some(Terminator {
                source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
                kind: TerminatorKind::FalseEdge {
                    real_target: elem.1,
                    imaginary_target: BasicBlock::from(u32::from(block_idx) + 1),
                },
            });
            elem.1 = block_idx;
        }
        // and the last block is imaginary to the last real switch block.
        let idx = body.basic_blocks_mut().push(BasicBlockData::new(Some(Terminator {
            source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: TerminatorKind::FalseEdge {
                real_target: switch_blocks[second_to_last_edge].1,
                imaginary_target: switch_blocks[second_to_last_edge + 1].1,
            },
        })));
        switch_blocks[second_to_last_edge].1 = idx;
    }
    // push unreachable block
    let unreachable_block_idx =
        body.basic_blocks_mut().push(BasicBlockData::new(Some(Terminator {
            source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: TerminatorKind::Unreachable,
        })));

    let mut switch_block = &mut body.basic_blocks_mut()[switch_int_block_idx];
    switch_block.statements.push(Statement {
        source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
        kind: StatementKind::FakeRead(Box::new((
            FakeReadCause::ForMatchedPlace(None),
            Place::from(SELF_LOCAL),
        ))),
    });
    switch_block.statements.push(Statement {
        source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
        kind: StatementKind::Assign(Box::new((
            Place::from(discr_local),
            Rvalue::Discriminant(Place {
                local: SELF_LOCAL,
                projection: tcx.intern_place_elems(&[ProjectionElem::Deref]),
            }),
        ))),
    });
    switch_block.terminator = Some(Terminator {
        source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
        kind: TerminatorKind::SwitchInt {
            discr: Operand::Copy(Place::from(discr_local)),
            targets: SwitchTargets::new(switch_blocks.into_iter(), unreachable_block_idx),
        },
    });
}

struct VariantCloneBlocks {
    /// The first block of the variant clone
    first_block: BasicBlock,
    /// The return block of the variant clone
    return_block: BasicBlock,
}

/// Builds the MIR for cloning a single variant. Used for both structs and enums.
///
/// Returns the first block, for ease of use in the enum builder.
fn build_variant_clone_mir<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    span: Span,
    adt: AdtDef<'tcx>,
    substs: SubstsRef<'tcx>,
    fn_ctxt: &FnCtxt<'a, 'tcx>,
    clone_did: DefId,
    variant_idx: VariantIdx,
    variant: &'tcx ty::VariantDef,
    variant_place: Place<'tcx>,
) -> VariantCloneBlocks {
    let block_base = body.basic_blocks.len();
    let field_cnt = variant.fields.len();
    // vec of locals for the cloned fields
    let mut clone_locals = Vec::with_capacity(field_cnt);
    let mut last_ref_local = None;
    for (idx, f) in variant.fields.iter().enumerate() {
        let field_ty = fn_ctxt.field_ty(span, f, substs);
        let erased_field_ty = tcx.erase_regions(field_ty);
        // allocate slot for the field reference
        let ref_local = body.local_decls.push(LocalDecl::new(
            tcx.mk_imm_ref(tcx.mk_region(RegionKind::ReErased), erased_field_ty),
            span,
        ));
        // allocate slot for the field clone
        let clone_local = body.local_decls.push(LocalDecl::new(erased_field_ty, span));
        clone_locals.push(clone_local);
        // get our field_ty's clone function
        let clone_fn = match fn_ctxt.lookup_method_in_trait_full(
            fn_ctxt.cause(span, ObligationCauseCode::MiscObligation),
            Ident::with_dummy_span(sym::clone),
            clone_did,
            field_ty,
            None,
        ) {
            Ok(v) => v,
            Err(errors) => {
                bug!("Could not find `Clone` impl for {:?}: {:?}", field_ty, errors)
            }
        };
        // push block with the clone
        let mut block = BasicBlockData::new(Some(Terminator {
            source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: TerminatorKind::Call {
                func: Operand::Constant(Box::new(Constant {
                    span,
                    user_ty: None,
                    literal: ConstantKind::Val(
                        ConstValue::ZeroSized,
                        tcx.erase_regions(tcx.mk_fn_def(clone_fn.def_id, clone_fn.substs)),
                    ),
                })),
                args: vec![Operand::Move(Place::from(ref_local))],
                destination: Place::from(clone_local),
                target: Some(BasicBlock::from(block_base + idx + 1)),
                cleanup: (idx > 0).then(|| BasicBlock::from(block_base + 2 * field_cnt - idx)),
                from_hir_call: false,
                fn_span: span,
            },
        }));
        // Dead the previous block's ref_local
        if let Some(last_ref_local) = last_ref_local {
            block.statements.push(Statement {
                source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
                kind: StatementKind::StorageDead(last_ref_local),
            });
        }
        last_ref_local = Some(ref_local);
        // Need to Live our locals
        block.statements.push(Statement {
            source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: StatementKind::StorageLive(ref_local),
        });
        block.statements.push(Statement {
            source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: StatementKind::StorageLive(clone_local),
        });
        block.statements.push(Statement {
            source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: StatementKind::Assign(Box::new((
                Place::from(ref_local),
                Rvalue::Ref(
                    tcx.mk_region(RegionKind::ReErased),
                    BorrowKind::Shared,
                    variant_place.project_deeper(
                        &[ProjectionElem::Field(Field::from(idx), erased_field_ty)],
                        tcx,
                    ),
                ),
            ))),
        });
        body.basic_blocks_mut().push(block);
    }
    // now that the fields are cloned, we build the final result
    let mut block = BasicBlockData::new(Some(Terminator {
        source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
        kind: TerminatorKind::Return,
    }));
    // Dead the last ref_local
    if let Some(last_ref_local) = last_ref_local {
        block.statements.push(Statement {
            source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: StatementKind::StorageDead(last_ref_local),
        });
    }
    let operands =
        clone_locals.iter().map(|&clone_local| Operand::Move(Place::from(clone_local))).collect();
    // Write the struct
    block.statements.push(Statement {
        source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
        kind: StatementKind::Assign(Box::new((
            Place::return_place(),
            Rvalue::Aggregate(
                Box::new(AggregateKind::Adt(adt.did(), variant_idx, substs, None, None)),
                operands,
            ),
        ))),
    });
    // Dead the clone locals
    for &clone_local in &clone_locals {
        block.statements.push(Statement {
            source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: StatementKind::StorageDead(clone_local),
        });
    }
    let return_block = body.basic_blocks_mut().push(block);
    // and finally add cleanup blocks (skipping one, since we don't need cleanup for the last field)
    for idx in 1..field_cnt {
        let cleanup_block_idx = body.basic_blocks_mut().push(BasicBlockData::new(None));
        let mut cleanup_block = &mut body.basic_blocks_mut()[cleanup_block_idx];
        cleanup_block.terminator = Some(Terminator {
            source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
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
            source_info: SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE },
            kind: TerminatorKind::Resume,
        }));
        block.is_cleanup = true;
        body.basic_blocks_mut().push(block);
    }
    VariantCloneBlocks { first_block: BasicBlock::from(block_base), return_block }
}
