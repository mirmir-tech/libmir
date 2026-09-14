use super::*;

#[test]
fn bounded_padding_preserves_routes_empty_experts_and_exhausted_budget() -> Result<()> {
    let stream = mirtal::Device::gpu(0).new_stream()?;
    for budget in [PaddingBudget::Four, PaddingBudget::Eight] {
        let plan = AlignedGroup::with_budget(budget)?;
        for counts in [vec![0, 3, 0, 17, 1, 0], vec![1; 128], vec![67], vec![0, 0, 3]] {
            let mut ids = Vec::new();
            let mut starts = Vec::new();
            let routes: usize = counts.iter().sum();
            let capacity = (routes + counts.len() * budget.rows()).div_ceil(16) * 16;
            let mut offset = 0;
            let mut spent = 0;
            for (expert, &count) in counts.iter().enumerate() {
                let padding = (16 - offset % 16) % 16;
                if count > 0 && spent + padding <= capacity - routes {
                    offset += padding;
                    spent += padding;
                }
                starts.push(offset);
                offset += count;
                ids.extend(std::iter::repeat_n(u32::try_from(expert)?, count));
            }
            ids.reverse();
            let indices = mirtal::Array::from_slice(&ids, [routes])?;
            let [order, inverse, grouped] = plan.forward(&stream, &indices, counts.len())?;
            let order = stream.read::<u32>(&order)?;
            let inverse = stream.read::<u32>(&inverse)?;
            let grouped = stream.read::<u32>(&grouped)?;
            assert_eq!(order.len(), capacity);
            assert!(grouped.windows(2).all(|w| w[0] <= w[1]));
            assert!(order.iter().all(|&n| usize::try_from(n).is_ok_and(|n| n < routes)));
            let mut occupied = vec![false; capacity];
            for (route, &destination) in inverse.iter().enumerate() {
                let destination = usize::try_from(destination)?;
                assert!(!occupied[destination]);
                occupied[destination] = true;
                assert_eq!(order[destination], u32::try_from(route)?);
                assert_eq!(grouped[destination], ids[route]);
                let expert = usize::try_from(ids[route])?;
                assert!((starts[expert]..starts[expert] + counts[expert]).contains(&destination));
            }
            if counts.len() == 128 {
                assert!(
                    starts.iter().any(|start| start % 16 != 0),
                    "must exercise exhausted padding"
                );
            }
        }
    }
    Ok(())
}
