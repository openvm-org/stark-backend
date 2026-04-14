use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DeriveInput, Fields};
#[cfg(feature = "lean")]
use syn::{GenericParam, Type};

fn codec_crate_root() -> proc_macro2::TokenStream {
    match proc_macro_crate::crate_name("openvm-stark-backend") {
        Ok(proc_macro_crate::FoundCrate::Itself) => quote!(crate),
        Ok(proc_macro_crate::FoundCrate::Name(name)) => {
            let ident = format_ident!("{}", name);
            quote!(::#ident)
        }
        Err(_) => quote!(::openvm_stark_backend),
    }
}

#[proc_macro_derive(Encode)]
pub fn encode_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let (impl_generics, type_generics, where_clause) = ast.generics.split_for_impl();
    let codec_root = codec_crate_root();

    let fields = match &ast.data {
        Data::Struct(data_struct) => &data_struct.fields,
        Data::Enum(_) => {
            return syn::Error::new(
                name.span(),
                "Encode derive macro only supports structs, not enums",
            )
            .to_compile_error()
            .into();
        }
        Data::Union(_) => {
            return syn::Error::new(
                name.span(),
                "Encode derive macro only supports structs, not unions",
            )
            .to_compile_error()
            .into();
        }
    };

    let encode_fields = match fields {
        Fields::Named(fields_named) => {
            let field_encodes = fields_named.named.iter().map(|field| {
                let field_name = &field.ident;
                quote! {
                    self.#field_name.encode(writer)?;
                }
            });
            quote! {
                #(#field_encodes)*
            }
        }
        Fields::Unnamed(fields_unnamed) => {
            let field_encodes = fields_unnamed.unnamed.iter().enumerate().map(|(idx, _)| {
                let index = syn::Index::from(idx);
                quote! {
                    self.#index.encode(writer)?;
                }
            });
            quote! {
                #(#field_encodes)*
            }
        }
        Fields::Unit => {
            quote! {}
        }
    };

    let expanded = quote! {
        impl #impl_generics #codec_root::codec::Encode for #name #type_generics #where_clause {
            fn encode<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
                #encode_fields
                Ok(())
            }
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(Decode)]
pub fn decode_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let (impl_generics, type_generics, where_clause) = ast.generics.split_for_impl();
    let codec_root = codec_crate_root();

    let fields = match &ast.data {
        Data::Struct(data_struct) => &data_struct.fields,
        Data::Enum(_) => {
            return syn::Error::new(
                name.span(),
                "Decode derive macro only supports structs, not enums",
            )
            .to_compile_error()
            .into();
        }
        Data::Union(_) => {
            return syn::Error::new(
                name.span(),
                "Decode derive macro only supports structs, not unions",
            )
            .to_compile_error()
            .into();
        }
    };

    let decode_fields = match fields {
        Fields::Named(fields_named) => {
            let field_decodes = fields_named.named.iter().map(|field| {
                let field_name = &field.ident;
                let field_ty = &field.ty;
                quote! {
                    #field_name: <#field_ty as #codec_root::codec::Decode>::decode(reader)?,
                }
            });
            quote! {
                {
                    #(#field_decodes)*
                }
            }
        }
        Fields::Unnamed(fields_unnamed) => {
            let field_decodes = fields_unnamed.unnamed.iter().map(|field| {
                let field_ty = &field.ty;
                quote! {
                    <#field_ty as #codec_root::codec::Decode>::decode(reader)?,
                }
            });
            quote! {
                (
                    #(#field_decodes)*
                )
            }
        }
        Fields::Unit => {
            quote! {}
        }
    };

    let expanded = quote! {
        impl #impl_generics #codec_root::codec::Decode for #name #type_generics #where_clause {
            fn decode<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
                Ok(Self #decode_fields)
            }
        }
    };

    TokenStream::from(expanded)
}

/// Generates `fn lean_columns() -> Vec<LeanEntry>` for Cols structs.
///
/// - `field: T` → `LeanEntry::Column("field")`
/// - `field: [T; N]` → `LeanEntry::Column("field_0")` .. `LeanEntry::Column("field_{N-1}")`
/// - `field: [[T; N]; M]` → `LeanEntry::Column("field_0_0")` ..
///   `LeanEntry::Column("field_{M-1}_{N-1}")`
/// - `field: SomeStruct<T, ..>` → `LeanEntry::SubAir { field_name, type_name, width }`
#[cfg(feature = "lean")]
#[proc_macro_derive(LeanColumns)]
pub fn lean_columns_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let (impl_generics, type_generics, where_clause) = ast.generics.split_for_impl();
    let lean_root = codec_crate_root();

    // Get the first type generic (e.g. `T`)
    let type_generic = ast
        .generics
        .params
        .iter()
        .find_map(|param| match param {
            GenericParam::Type(tp) => Some(&tp.ident),
            _ => None,
        })
        .expect("LeanColumns requires at least one type generic");

    let fields = match &ast.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return syn::Error::new(name.span(), "LeanColumns only supports named fields")
                    .to_compile_error()
                    .into();
            }
        },
        _ => {
            return syn::Error::new(name.span(), "LeanColumns only supports structs")
                .to_compile_error()
                .into();
        }
    };

    let push_stmts = fields.iter().map(|field| {
        let field_name = field.ident.as_ref().unwrap().to_string();
        let field_ty = &field.ty;

        if is_type_generic(field_ty, type_generic) {
            // Scalar field: T → Column("field")
            quote! {
                entries.push(#lean_root::lean::LeanEntry::Column(#field_name.to_string()));
            }
        } else if let Type::Array(arr) = field_ty {
            if is_type_generic(&arr.elem, type_generic) {
                // Array of T: [T; N] → Column("field_0") .. Column("field_{N-1}")
                let len = &arr.len;
                quote! {
                    for i in 0..#len {
                        entries.push(#lean_root::lean::LeanEntry::Column(
                            format!("{}_{}", #field_name, i),
                        ));
                    }
                }
            } else if let Type::Array(inner_arr) = &*arr.elem {
                if is_type_generic(&inner_arr.elem, type_generic) {
                    // Nested array of T: [[T; N]; M] → Column("field_0_0") .. Column("field_{M-1}_{N-1}")
                    let outer_len = &arr.len;
                    let inner_len = &inner_arr.len;
                    quote! {
                        for i in 0..#outer_len {
                            for j in 0..#inner_len {
                                entries.push(#lean_root::lean::LeanEntry::Column(
                                    format!("{}_{}_{}", #field_name, i, j),
                                ));
                            }
                        }
                    }
                } else {
                    // Array of array of structs: [[SomeStruct<T>; N]; M] → M*N SubAir entries
                    let elem_ty = &inner_arr.elem;
                    let type_name_str = extract_type_name(elem_ty);
                    let outer_len = &arr.len;
                    let inner_len = &inner_arr.len;
                    quote! {
                        for _ in 0..#outer_len {
                            for _ in 0..#inner_len {
                                entries.push(#lean_root::lean::LeanEntry::SubAir {
                                    field_name: #field_name.to_string(),
                                    type_name: #type_name_str.to_string(),
                                    width: std::mem::size_of::<#elem_ty>() / std::mem::size_of::<#type_generic>(),
                                });
                            }
                        }
                    }
                }
            } else {
                // Array of structs: [SomeStruct<T>; N] → N SubAir entries
                let elem_ty = &arr.elem;
                let type_name_str = extract_type_name(elem_ty);
                let len = &arr.len;
                quote! {
                    for _ in 0..#len {
                        entries.push(#lean_root::lean::LeanEntry::SubAir {
                            field_name: #field_name.to_string(),
                            type_name: #type_name_str.to_string(),
                            width: std::mem::size_of::<#elem_ty>() / std::mem::size_of::<#type_generic>(),
                        });
                    }
                }
            }
        } else {
            // Nested struct: SomeStruct<T, ..> → SubAir
            let type_name_str = extract_type_name(field_ty);
            quote! {
                entries.push(#lean_root::lean::LeanEntry::SubAir {
                    field_name: #field_name.to_string(),
                    type_name: #type_name_str.to_string(),
                    width: std::mem::size_of::<#field_ty>() / std::mem::size_of::<#type_generic>(),
                });
            }
        }
    });

    let expanded = quote! {
        impl #impl_generics #lean_root::lean::LeanColumns for #name #type_generics #where_clause {
            fn lean_columns() -> Vec<#lean_root::lean::LeanEntry> {
                let mut entries = Vec::new();
                #(#push_stmts)*
                entries
            }
        }
    };

    TokenStream::from(expanded)
}

/// Check if a type is exactly the given generic ident (e.g. `T`).
#[cfg(feature = "lean")]
fn is_type_generic(ty: &Type, generic: &syn::Ident) -> bool {
    if let Type::Path(tp) = ty {
        tp.qself.is_none() && tp.path.is_ident(generic)
    } else {
        false
    }
}

/// Extract the top-level type name from a type (e.g. `ExecutionState<T>` → `"ExecutionState"`).
#[cfg(feature = "lean")]
fn extract_type_name(ty: &Type) -> String {
    if let Type::Path(tp) = ty {
        if let Some(segment) = tp.path.segments.last() {
            return segment.ident.to_string();
        }
    }
    "Unknown".to_string()
}
