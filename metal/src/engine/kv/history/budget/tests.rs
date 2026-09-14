use super::*;

#[test]
fn shared_lease_accounts_once_and_rejects_overflow_growth_and_suspension()
-> crate::engine::Result<()> {
    let budget = Arc::new(Budget::new(100));
    let first = budget.reserve(60).ok_or(crate::engine::Error::HistoryBudgetUnavailable)?;
    let alias = Arc::clone(&first);
    assert_eq!(budget.snapshot().retained_bytes, 60);
    assert!(budget.reserve(41).is_none());
    assert!(budget.reserve(usize::MAX).is_none());
    let second = budget.reserve(40).ok_or(crate::engine::Error::HistoryBudgetUnavailable)?;
    assert!(budget.allows_reuse());
    budget.set_limit(99);
    assert!(!budget.allows_reuse());
    drop(second);
    assert!(budget.allows_reuse());
    budget.set_admission(Admission::Suspended);
    assert!(!budget.allows_reuse());
    assert!(budget.reserve(1).is_none());
    drop(first);
    assert_eq!(budget.snapshot().retained_bytes, 60);
    drop(alias);
    assert_eq!(budget.snapshot().retained_bytes, 0);
    budget.set_admission(Admission::Open);
    assert!(budget.reserve(99).is_some());
    assert_eq!(budget.snapshot().retained_bytes, 0);
    Ok(())
}
