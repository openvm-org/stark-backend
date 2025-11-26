use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DeriveInput, Fields};

fn codec_crate_root() -> proc_macro2::TokenStream {
    match proc_macro_crate::crate_name("stark-backend-v2") {
        Ok(proc_macro_crate::FoundCrate::Itself) => quote!(crate),
        Ok(proc_macro_crate::FoundCrate::Name(name)) => {
            let ident = format_ident!("{}", name);
            quote!(::#ident)
        }
        Err(_) => quote!(::stark_backend_v2),
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
