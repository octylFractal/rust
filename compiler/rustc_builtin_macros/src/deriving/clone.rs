use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::path_std;
use rustc_ast::ast;
use rustc_ast::Generics;
use rustc_ast::{ItemKind, MetaItem};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::sym;
use rustc_span::Span;
use thin_vec::thin_vec;

pub fn expand_deriving_clone(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    // The simple form is `fn clone(&self) -> Self { *self }`, possibly with
    // some additional `AssertParamIsClone` assertions.
    //
    // We can use the simple form if either of the following are true.
    // - The type derives Copy and there are no generic parameters. (If we
    //   used the simple form with generics, we'd have to bound the generics
    //   with Clone + Copy, and then there'd be no Clone impl at all if the
    //   user fills in something that is Clone but not Copy. After
    //   specialization we can remove this no-generics limitation.)
    // - The item is a union. (Unions with generic parameters still can derive
    //   Clone because they require Copy for deriving, Clone alone is not
    //   enough. Whether Clone is implemented for fields is irrelevant so we
    //   don't assert it.)
    let bounds;
    let substructure = combine_substructure(Box::new(|cx, s, _| {
        // This should be replaced by the compiler, for now we'll just pass type checks with a copy
        // FIXME: make this somehow pass type-checks but compile error if not replaced in MIR stage?
        BlockOrExpr::new_expr(cx.expr_deref(s, cx.expr_self(s)))
    }));
    let mut is_copy_guaranteed = false;
    match item {
        Annotatable::Item(annitem) => match &annitem.kind {
            ItemKind::Struct(_, Generics { params, .. })
            | ItemKind::Enum(_, Generics { params, .. }) => {
                let container_id = cx.current_expansion.id.expn_data().parent.expect_local();
                let has_derive_copy = cx.resolver.has_derive_copy(container_id);
                is_copy_guaranteed = has_derive_copy
                    && !params
                        .iter()
                        .any(|param| matches!(param.kind, ast::GenericParamKind::Type { .. }));

                bounds = vec![];
            }
            ItemKind::Union(..) => {
                bounds = vec![Path(path_std!(marker::Copy))];
            }
            _ => cx.span_bug(span, "`#[derive(Clone)]` on wrong item kind"),
        },

        _ => cx.span_bug(span, "`#[derive(Clone)]` on trait item or impl item"),
    }

    let derived_clone_attr = if is_copy_guaranteed {
        // Tell the MIR builder that we are for-sure Copy.
        cx.attr_nested_word(sym::derived_clone, sym::copy, span)
    } else {
        cx.attr_word(sym::derived_clone, span)
    };

    let attrs = thin_vec![cx.attr_word(sym::inline, span), derived_clone_attr];
    let trait_def = TraitDef {
        span,
        path: path_std!(clone::Clone),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: bounds,
        supports_unions: true,
        methods: vec![MethodDef {
            name: sym::clone,
            generics: Bounds::empty(),
            explicit_self: true,
            nonself_args: Vec::new(),
            ret_ty: Self_,
            attributes: attrs,
            fieldless_variants_strategy: FieldlessVariantsStrategy::Default,
            combine_substructure: substructure,
        }],
        associated_types: Vec::new(),
        is_const,
    };

    trait_def.expand_ext(cx, mitem, item, push, true)
}
