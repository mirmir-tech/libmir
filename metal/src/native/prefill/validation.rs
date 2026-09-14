use std::collections::HashSet;

use runtime::backend::PrefillRequest;

use crate::native::error::{Error, Result};

pub(in crate::native) fn validate_requests<'a>(
    requests: impl IntoIterator<Item = &'a PrefillRequest>,
) -> Result<()> {
    let mut sessions = HashSet::new();
    for request in requests {
        if request.prompt_tokens.is_empty() {
            return Err(Error::EmptyPrompt);
        }
        if !sessions.insert(request.session_id) {
            return Err(Error::InvalidPrefillBatch(
                "prefill batch contains a duplicate session".into(),
            ));
        }
    }
    if sessions.is_empty() {
        return Err(Error::InvalidPrefillBatch("prefill batch cannot be empty".into()));
    }
    Ok(())
}
