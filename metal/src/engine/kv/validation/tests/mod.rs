use super::super::*;

#[test]
fn rejected_empty_update_keeps_shared_page() -> Result<()> {
    rejected_update(false)
}

#[test]
fn rejected_dtype_update_keeps_shared_page() -> Result<()> {
    rejected_update(true)
}

fn rejected_update(change_dtype: bool) -> Result<()> {
    let stream = Stream::new_gpu()?;
    for format in [KvPageFormat::Native, KvPageFormat::Int8PerTokenHead] {
        let mut cache = KvCache::new_paged_with_format(2, 2, format)?;
        let first = Array::from_f32(&[1.0, 0.5, -0.5, 0.25], &[1, 1, 1, 4])?;
        let context = cache.update(&first, &first, &stream)?;
        context
            .paged
            .as_ref()
            .ok_or(Error::NullHandle("pages"))?
            .page_dependency
            .async_eval(&stream)?;
        stream.synchronize()?;
        let mut snapshot = cache.snapshot_at(1)?;
        let page = cache.first_physical_page();
        let bad = if change_dtype {
            Array::from_native(
                stream.native().graph().astype(first.native(), mirtal::DType::Bfloat16)?,
            )?
        } else {
            Array::from_f32(&[], &[1, 1, 0, 4])?
        };
        assert!(cache.update(&bad, &bad, &stream).is_err());
        assert_eq!(cache.offset()?, 1);
        assert_eq!(cache.physical_page_count(), 1);
        assert_eq!(cache.first_physical_page(), page, "rejected update performed COW");
        assert_eq!(snapshot.first_physical_page(), page);
        let next = Array::from_f32(&[0.0, 1.0, 0.0, 0.5], &[1, 1, 1, 4])?;
        let current = cache.update(&next, &next, &stream)?;
        let branch = snapshot.update(&first, &first, &stream)?;
        assert_eq!(cache.offset()?, 2);
        assert_eq!(snapshot.offset()?, 2);
        assert_ne!(cache.first_physical_page(), snapshot.first_physical_page());
        for context in [current, branch] {
            context
                .paged
                .ok_or(Error::NullHandle("pages"))?
                .page_dependency
                .async_eval(&stream)?;
        }
        stream.synchronize()?;
    }
    Ok(())
}

#[test]
fn empty_initial_updates_do_not_allocate_storage() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let empty = Array::from_f32(&[], &[1, 1, 0, 4])?;
    for mut cache in [
        KvCache::new(1)?,
        KvCache::new_paged(2, 2)?,
        KvCache::new_paged_with_format(2, 2, KvPageFormat::Int8PerTokenHead)?,
    ] {
        assert!(cache.update(&empty, &empty, &stream).is_err());
        assert_eq!(cache.offset()?, 0);
        assert_eq!(cache.physical_page_count(), 0);
        assert!(cache.keys.is_none() && cache.values.is_none());
        assert!(cache.pages.as_ref().is_none_or(|pages| !pages.active()));
    }
    Ok(())
}

#[test]
fn invalid_layout_does_not_grow_promote_or_advance_caches() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let first = Array::from_f32(&[1.0, 0.5, -0.5, 0.25], &[1, 1, 1, 4])?;
    let next = Array::from_f32(&[0.0, 1.0, 0.0, 0.5], &[1, 1, 1, 4])?;
    let mut invalid = Vec::new();
    for shape in [[1, 2, 2, 4], [1, 1, 2, 8], [2, 1, 2, 4]] {
        let array = Array::from_native(stream.native().graph().full(
            &mirtal::Shape::new(shape)?,
            0.0,
            mirtal::DType::Float32,
        )?)?;
        invalid.push((Array::from_native(array.native().clone())?, array));
    }
    invalid.push((
        Array::from_native(first.native().clone())?,
        Array::from_f32(&[0.0; 8], &[1, 2, 1, 4])?,
    ));
    invalid.push((
        Array::from_f32(&[0.0; 4], &[1, 1, 4])?,
        Array::from_native(first.native().clone())?,
    ));
    let dtype = Array::from_native(
        stream.native().graph().astype(first.native(), mirtal::DType::Bfloat16)?,
    )?;
    invalid.push((Array::from_native(dtype.native().clone())?, dtype));
    for (mut cache, threshold) in [
        (KvCache::new(1)?, 0),
        (KvCache::new_with_window(1, Some(1))?, 0),
        (KvCache::new_paged(2, 2)?, 0),
        (KvCache::new_paged_with_format(2, 2, KvPageFormat::Int8PerTokenHead)?, 0),
        (KvCache::new_paged(2, 2)?, 3),
    ] {
        drop(cache.update_with_page_min_context(&first, &first, &stream, threshold)?);
        let capacity = cache.capacity;
        let position = cache.write_index;
        let page = cache.first_physical_page();
        for (keys, values) in &invalid {
            assert!(cache.update_with_page_min_context(keys, values, &stream, threshold).is_err());
            assert_eq!(cache.offset()?, 1);
            assert_eq!(cache.capacity, capacity);
            assert_eq!(cache.write_index, position);
            assert_eq!(cache.first_physical_page(), page);
            if let Some(keys) = &cache.keys {
                assert_eq!(keys.native().dtype()?, mirtal::DType::Float32);
                assert_eq!(&keys.to_vec_f32(&stream)?[..4], &[1.0, 0.5, -0.5, 0.25]);
            }
        }
        let context = cache.update_with_page_min_context(&next, &next, &stream, threshold)?;
        assert_eq!(cache.offset()?, 2);
        if let Some(pages) = context.paged {
            pages.page_dependency.async_eval(&stream)?;
        } else {
            let values = context.keys.to_vec_f32(&stream)?;
            assert_eq!(&values[values.len() - 4..], &[0.0, 1.0, 0.0, 0.5]);
        }
        stream.synchronize()?;
    }
    Ok(())
}
mod capacity;
