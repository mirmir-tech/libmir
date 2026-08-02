use std::collections::{HashMap, VecDeque};

const MAX_RESIDENT_SHAPES: usize = 4;

pub(in crate::backend::model::session) struct PrefillShapeCache<T> {
    entries: HashMap<usize, T>,
    order: VecDeque<usize>,
}

impl<T> PrefillShapeCache<T> {
    pub(in crate::backend::model::session) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    pub(in crate::backend::model::session) fn take(&mut self, shape: usize) -> Option<T> {
        remove_shape(&mut self.order, shape);
        self.entries.remove(&shape)
    }

    pub(in crate::backend::model::session) fn insert(&mut self, shape: usize, value: T) {
        remove_shape(&mut self.order, shape);
        if !self.entries.contains_key(&shape)
            && self.entries.len() == MAX_RESIDENT_SHAPES
            && let Some(oldest) = self.order.pop_front()
        {
            self.entries.remove(&oldest);
        }
        self.order.push_back(shape);
        self.entries.insert(shape, value);
    }

    #[cfg(test)]
    fn contains(&self, shape: usize) -> bool {
        self.entries.contains_key(&shape)
    }
}

fn remove_shape(order: &mut VecDeque<usize>, shape: usize) {
    if let Some(index) = order.iter().position(|candidate| *candidate == shape) {
        order.remove(index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_the_least_recently_used_shape() {
        let mut cache = PrefillShapeCache::new();
        for shape in 1..=MAX_RESIDENT_SHAPES {
            cache.insert(shape, shape);
        }
        assert_eq!(cache.take(1), Some(1));
        cache.insert(1, 1);
        cache.insert(MAX_RESIDENT_SHAPES + 1, MAX_RESIDENT_SHAPES + 1);

        assert!(cache.contains(1));
        assert!(!cache.contains(2));
        assert!(cache.contains(MAX_RESIDENT_SHAPES + 1));
    }
}
