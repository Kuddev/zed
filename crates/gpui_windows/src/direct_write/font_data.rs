use std::borrow::Cow;

use anyhow::{Context, Result};
use windows::{
    Win32::Graphics::DirectWrite::{
        IDWriteFactory5, IDWriteFontFile, IDWriteInMemoryFontFileLoader,
    },
    core::{ComObject, IUnknown, implement},
};

#[implement()]
struct FontData {
    bytes: Cow<'static, [u8]>,
}

pub(super) fn create_font_file(
    factory: &IDWriteFactory5,
    loader: &IDWriteInMemoryFontFileLoader,
    bytes: Cow<'static, [u8]>,
) -> Result<IDWriteFontFile> {
    create_reference(factory, loader, ComObject::new(FontData { bytes }))
}

fn font_byte_count(length: usize) -> Result<u32> {
    anyhow::ensure!(length != 0, "font data is empty");
    u32::try_from(length).context("font data exceeds the DirectWrite size limit")
}

fn create_reference(
    factory: &IDWriteFactory5,
    loader: &IDWriteInMemoryFontFileLoader,
    owner: ComObject<FontData>,
) -> Result<IDWriteFontFile> {
    let bytes = owner.get().bytes.as_ref();
    let byte_count = font_byte_count(bytes.len())?;
    let unknown = owner.to_interface::<IUnknown>();
    // Without an owner DirectWrite copies the complete font. Its retained COM
    // reference instead keeps this immutable Vec (or static slice) alive until
    // the last native font stream releases it, including after our caller exits.
    unsafe {
        loader.CreateInMemoryFontFileReference(factory, bytes.as_ptr().cast(), byte_count, &unknown)
    }
    .context("creating an in-memory font reference")
}

#[cfg(test)]
mod tests {
    use std::{ffi::c_void, ptr};

    use windows::{Win32::Graphics::DirectWrite::*, core::Interface};

    use super::*;

    static FONT: &[u8] = include_bytes!("../../../../assets/fonts/lilex/Lilex-Regular.ttf");

    fn factory_and_loader() -> Result<(IDWriteFactory5, IDWriteInMemoryFontFileLoader)> {
        let factory: IDWriteFactory5 =
            unsafe { DWriteCreateFactory(DWRITE_FACTORY_TYPE_ISOLATED)? };
        let loader = unsafe { factory.CreateInMemoryFontFileLoader()? };
        unsafe { factory.RegisterFontFileLoader(&loader)? };
        Ok((factory, loader))
    }

    fn stream_for(file: &IDWriteFontFile) -> Result<IDWriteFontFileStream> {
        let mut key = ptr::null_mut();
        let mut key_size = 0;
        unsafe {
            file.GetReferenceKey(&mut key, &mut key_size)?;
            Ok(file.GetLoader()?.CreateStreamFromKey(key, key_size)?)
        }
    }

    fn assert_original_bytes(stream: &IDWriteFontFileStream, original: *const u8) -> Result<()> {
        assert_eq!(unsafe { stream.GetFileSize()? }, FONT.len() as u64);
        let mut fragment: *mut c_void = ptr::null_mut();
        let mut context = ptr::null_mut();
        unsafe {
            stream.ReadFileFragment(&mut fragment, 0, FONT.len() as u64, &mut context)?;
        }
        let same_pointer = fragment.cast::<u8>().cast_const() == original;
        let same_bytes =
            unsafe { std::slice::from_raw_parts(fragment.cast::<u8>(), FONT.len()) } == FONT;
        unsafe { stream.ReleaseFileFragment(context) };
        assert!(same_pointer, "DirectWrite must reuse the input allocation");
        assert!(same_bytes, "font bytes must remain intact");
        Ok(())
    }

    fn assert_owner_lifetime(bytes: Cow<'static, [u8]>) -> Result<()> {
        let original = bytes.as_ptr();
        let (factory, loader) = factory_and_loader()?;
        let owner = ComObject::new(FontData { bytes });
        let weak = owner.to_interface::<IUnknown>().downgrade()?;
        let file = create_reference(&factory, &loader, owner)?;
        let builder = unsafe { factory.CreateFontSetBuilder()? };
        unsafe { builder.AddFontFile(&file)? };
        let set = unsafe { builder.CreateFontSet()? };
        let collection = unsafe { factory.CreateFontCollectionFromFontSet(&set)? };
        assert_eq!(unsafe { collection.GetFontFamilyCount() }, 1);
        let face = unsafe {
            collection
                .GetFontFamily(0)?
                .GetFirstMatchingFont(
                    DWRITE_FONT_WEIGHT_NORMAL,
                    DWRITE_FONT_STRETCH_NORMAL,
                    DWRITE_FONT_STYLE_NORMAL,
                )?
                .CreateFontFace()?
        };
        let stream = stream_for(&file)?;
        drop(collection);
        drop(set);
        drop(builder);
        drop(file);
        unsafe { factory.UnregisterFontFileLoader(&loader)? };
        drop(loader);
        drop(factory);
        assert!(
            weak.upgrade().is_some(),
            "native consumers must retain the owner"
        );
        assert_original_bytes(&stream, original)?;
        assert!(unsafe { face.GetGlyphCount() } > 0);
        drop(face);
        assert!(
            weak.upgrade().is_some(),
            "the remaining stream must retain the owner"
        );
        assert_original_bytes(&stream, original)?;
        drop(stream);
        assert!(
            weak.upgrade().is_none(),
            "the final stream must release the owner"
        );
        Ok(())
    }

    #[test]
    fn borrowed_font_reuses_static_bytes_and_releases_its_owner() -> Result<()> {
        assert_owner_lifetime(Cow::Borrowed(FONT))
    }

    #[test]
    fn owned_font_outlives_registration_handles_without_a_copy() -> Result<()> {
        assert_owner_lifetime(Cow::Owned(FONT.to_vec()))
    }

    #[test]
    fn public_registration_reuses_the_original_font() -> Result<()> {
        let (factory, loader) = factory_and_loader()?;
        let file = create_font_file(&factory, &loader, Cow::Borrowed(FONT))?;
        assert_original_bytes(&stream_for(&file)?, FONT.as_ptr())?;
        unsafe { factory.UnregisterFontFileLoader(&loader)? };
        Ok(())
    }

    #[test]
    fn invalid_font_releases_its_owner_after_builder_rejection() -> Result<()> {
        let (factory, loader) = factory_and_loader()?;
        let owner = ComObject::new(FontData {
            bytes: Cow::Owned(vec![0; 32]),
        });
        let weak = owner.to_interface::<IUnknown>().downgrade()?;
        let file = create_reference(&factory, &loader, owner)?;
        let builder = unsafe { factory.CreateFontSetBuilder()? };
        assert!(unsafe { builder.AddFontFile(&file) }.is_err());
        drop(builder);
        drop(file);
        unsafe { factory.UnregisterFontFileLoader(&loader)? };
        drop(loader);
        drop(factory);
        assert!(weak.upgrade().is_none());
        Ok(())
    }

    #[test]
    fn empty_font_releases_its_owner_on_error() -> Result<()> {
        let (factory, loader) = factory_and_loader()?;
        let owner = ComObject::new(FontData {
            bytes: Cow::Owned(Vec::new()),
        });
        let weak = owner.to_interface::<IUnknown>().downgrade()?;
        assert!(create_reference(&factory, &loader, owner).is_err());
        assert!(weak.upgrade().is_none());
        unsafe { factory.UnregisterFontFileLoader(&loader)? };
        Ok(())
    }

    #[test]
    fn font_length_does_not_truncate_to_the_native_integer_size() -> Result<()> {
        assert!(font_byte_count(0).is_err());
        assert_eq!(font_byte_count(1)?, 1);
        assert_eq!(font_byte_count(u32::MAX as usize)?, u32::MAX);
        #[cfg(target_pointer_width = "64")]
        assert!(font_byte_count(u32::MAX as usize + 1).is_err());
        Ok(())
    }
}
