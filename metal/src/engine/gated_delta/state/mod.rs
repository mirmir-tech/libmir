use std::{ops::Deref, sync::Arc};

use crate::engine::{Array, Error, Result, Stream};

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub(super) struct StateArray(Storage);

#[derive(Debug)]
enum Storage {
    Owned(Array),
    Row {
        source: Arc<RowSource>,
        index: usize,
        view: Array,
    },
}

#[derive(Debug)]
struct RowSource {
    array: Array,
    rows: usize,
}

impl From<Array> for StateArray {
    fn from(array: Array) -> Self {
        Self(Storage::Owned(array))
    }
}

impl Deref for StateArray {
    type Target = Array;

    fn deref(&self) -> &Array {
        match &self.0 {
            Storage::Owned(array) => array,
            Storage::Row { view, .. } => view,
        }
    }
}

impl StateArray {
    // Keep the retained parent among explicit roots so each evaluated state
    // generation releases its producing graph.
    pub(super) fn graph_root(&self) -> &Array {
        match &self.0 {
            Storage::Owned(array) => array,
            Storage::Row { source, .. } => &source.array,
        }
    }

    pub(super) fn try_clone(&self) -> Result<Self> {
        let array = Array::from_native(self.native().clone())?;
        Ok(match &self.0 {
            Storage::Owned(_) => array.into(),
            Storage::Row { source, index, .. } => Self(Storage::Row {
                source: Arc::clone(source),
                index: *index,
                view: array,
            }),
        })
    }

    pub(super) fn join(rows: &[&Self], stream: &Stream) -> Result<Array> {
        if let Some(array) = Self::whole_batch(rows) {
            return Array::from_native(array.native().clone());
        }
        let arrays = rows.iter().map(|row| &***row).collect::<Vec<_>>();
        Array::concatenate(&arrays, 0, stream)
    }

    fn whole_batch<'a>(rows: &[&'a Self]) -> Option<&'a Array> {
        let Storage::Row { source, .. } = &rows.first()?.0 else {
            return None;
        };
        // Only an unchanged, complete cohort can consume its immutable parent.
        // Admission, removal and reordering must concatenate the actual row views.
        let complete = source.rows == rows.len()
            && rows.iter().enumerate().all(|(expected, row)| {
                matches!(&row.0, Storage::Row { source: other, index, .. }
                    if *index == expected && Arc::ptr_eq(source, other))
            });
        complete.then_some(&source.array)
    }

    pub(super) fn split(array: Array, rows: usize, stream: &Stream) -> Result<Vec<Self>> {
        let shape = array
            .shape()?
            .into_iter()
            .map(usize::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if rows == 0 || shape.first() != Some(&rows) {
            return Err(Error::InvalidModel(
                "Gated Delta packed state has incompatible rows".into(),
            ));
        }
        let source = Arc::new(RowSource { array, rows });
        (0..rows)
            .map(|index| {
                let mut start = vec![0; shape.len()];
                let mut stop = shape.clone();
                start[0] = index;
                stop[0] = index + 1;
                let view = source.array.slice(&start, &stop, stream)?;
                Ok(Self(Storage::Row { source: Arc::clone(&source), index, view }))
            })
            .collect()
    }
}
