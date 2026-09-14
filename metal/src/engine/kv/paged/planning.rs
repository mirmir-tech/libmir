use std::{collections::HashMap, sync::Arc};

use super::{PagedArenaPool, PagedStore, lock, pool::ArenaKey};
use crate::engine::{Array, Error, Result};

#[derive(Default)]
pub struct DecodeCapacity {
    arenas: Vec<PlannedArena>,
}

struct PlannedArena {
    pool: Arc<PagedArenaPool>,
    key: ArenaKey,
    capacity: usize,
    used: usize,
    remaining: HashMap<u32, usize>,
}

impl PagedStore {
    pub(in crate::engine::kv) fn plan_decode(
        &self,
        keys: Option<&Array>,
        offset: usize,
        tokens: usize,
        plan: &mut DecodeCapacity,
    ) -> Result<()> {
        let end = offset.checked_add(tokens).ok_or(Error::ShapeOverflow)?;
        let needed = end.div_ceil(self.page_size);
        let planned = needed.max(self.reserve_pages);
        let mut shared = Vec::new();
        let (key, capacity, used, reservation) = if let Some(storage) = &self.storage {
            let owned = storage.page_ids.len() + storage.reserved_page_ids.len();
            let reservation = planned.saturating_sub(owned);
            let arena = lock(&storage.arena)?;
            for logical in offset / self.page_size..needed.min(storage.page_ids.len()) {
                let page = storage.page_ids[logical];
                let references = arena.references[usize::try_from(page)?];
                if references > 1 {
                    shared.push((page, references));
                }
            }
            // The usual private-tail decode needs neither allocation nor a plan entry.
            if reservation == 0 && shared.is_empty() && arena.capacity <= self.max_pages {
                return Ok(());
            }
            (
                ArenaKey::new(
                    self.layer,
                    self.page_size,
                    self.format,
                    arena.kv_heads,
                    arena.head_dim,
                    storage.input_dtype,
                )?,
                arena.capacity,
                arena.references.iter().filter(|count| **count > 0).count(),
                reservation,
            )
        } else {
            let keys = keys.ok_or(Error::NullHandle("decode K/V prefix"))?;
            let shape = keys.native().shape()?;
            let key = ArenaKey::new(
                self.layer,
                self.page_size,
                self.format,
                shape.dimensions()[1],
                shape.dimensions()[3],
                keys.native().dtype()?,
            )?;
            let (capacity, used) = match self.pool.find(&key)? {
                Some(arena) => {
                    let arena = lock(&arena)?;
                    (arena.capacity, arena.references.iter().filter(|count| **count > 0).count())
                },
                None => (0, 0),
            };
            (key, capacity, used, planned)
        };
        let index = plan
            .arenas
            .iter()
            .position(|entry| Arc::ptr_eq(&entry.pool, &self.pool) && entry.key == key);
        let entry = if let Some(index) = index {
            &mut plan.arenas[index]
        } else {
            plan.arenas.push(PlannedArena {
                pool: Arc::clone(&self.pool),
                key,
                capacity,
                used,
                remaining: HashMap::new(),
            });
            plan.arenas.last_mut().ok_or(Error::NullHandle("decode capacity plan"))?
        };
        let mut copies = 0_usize;
        for (page, references) in shared {
            let remaining = entry.remaining.entry(page).or_insert(references);
            if *remaining > 1 {
                *remaining -= 1;
                copies += 1;
            }
        }
        entry.used = entry
            .used
            .checked_add(reservation)
            .and_then(|used| used.checked_add(copies))
            .ok_or(Error::ShapeOverflow)?;
        let required = entry.used.max(entry.capacity).max(self.reserve_pages);
        if required > self.max_pages {
            return Err(Error::KvPageCapacity {
                layer: self.layer,
                required,
                maximum: self.max_pages,
            });
        }
        Ok(())
    }
}
