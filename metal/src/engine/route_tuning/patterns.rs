use super::{Array, Error, Result, RoutingKey, elements};

pub(super) struct RoutePatterns {
    pub(super) balanced: Array,
    pub(super) hot_set: Array,
}

pub(super) fn route_patterns(key: RoutingKey, indices: &Array) -> Result<RoutePatterns> {
    let shape = indices.shape()?;
    let assignments = elements(&shape)?;
    let balanced = (0..assignments)
        .map(|assignment| u32::try_from(assignment % key.experts).map_err(Error::from))
        .collect::<Result<Vec<_>>>()?;
    let hot_set = (0..assignments)
        .map(|assignment| u32::try_from(assignment % key.top_k).map_err(Error::from))
        .collect::<Result<Vec<_>>>()?;
    Ok(RoutePatterns {
        balanced: Array::from_u32(&balanced, &shape)?,
        hot_set: Array::from_u32(&hot_set, &shape)?,
    })
}
