//! Sets of payload-free kinds: the vocabulary in which a lowered plan states
//! what it requires and a runtime declares what it executes.

use std::{fmt, hash::Hash, marker::PhantomData};

/// A payload-free kind with a stable position among at most 64 values.
pub trait Flag: Copy + Eq + fmt::Debug + 'static {
    /// Every value, in a stable order.
    const ALL: &'static [Self];

    /// The value's position in [`Self::ALL`].
    fn index(self) -> u32;
}

/// A set of kinds.
pub struct Set<T> {
    bits: u64,
    kind: PhantomData<T>,
}

impl<T: Flag> Set<T> {
    pub const EMPTY: Self = Self { bits: 0, kind: PhantomData };

    #[must_use]
    pub fn of(items: impl IntoIterator<Item = T>) -> Self {
        items.into_iter().fold(Self::EMPTY, Self::with)
    }

    #[must_use]
    pub fn all() -> Self {
        Self::of(T::ALL.iter().copied())
    }

    #[must_use]
    pub fn with(self, item: T) -> Self {
        Self {
            bits: self.bits | bit(item),
            kind: PhantomData,
        }
    }

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
            kind: PhantomData,
        }
    }

    /// The items of `self` that `other` lacks.
    #[must_use]
    pub const fn difference(self, other: Self) -> Self {
        Self {
            bits: self.bits & !other.bits,
            kind: PhantomData,
        }
    }

    #[must_use]
    pub fn contains(self, item: T) -> bool {
        self.bits & bit(item) != 0
    }

    pub fn insert(&mut self, item: T) {
        self.bits |= bit(item);
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }

    pub fn iter(self) -> impl Iterator<Item = T> {
        T::ALL.iter().copied().filter(move |item| self.contains(*item))
    }
}

fn bit<T: Flag>(item: T) -> u64 {
    let index = item.index();
    debug_assert!(index < u64::BITS, "{item:?} has no bit");
    1 << index
}

impl<T: Flag> FromIterator<T> for Set<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        Self::of(items)
    }
}

impl<T> Clone for Set<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Set<T> {}

impl<T> PartialEq for Set<T> {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
    }
}

impl<T> Eq for Set<T> {}

impl<T> Hash for Set<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bits.hash(state);
    }
}

impl<T> Default for Set<T> {
    fn default() -> Self {
        Self { bits: 0, kind: PhantomData }
    }
}

impl<T: Flag> fmt::Debug for Set<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_set().entries(self.iter()).finish()
    }
}

/// A fieldless enum usable as a [`Flag`], its variants in declaration order.
macro_rules! flag_enum {
    ($(#[$meta:meta])* pub enum $name:ident { $($(#[$variant_meta:meta])* $variant:ident),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        pub enum $name {
            $($(#[$variant_meta])* $variant),+
        }

        impl $crate::set::Flag for $name {
            const ALL: &'static [Self] = &[$(Self::$variant),+];

            fn index(self) -> u32 {
                self as u32
            }
        }
    };
}

pub(crate) use flag_enum;
