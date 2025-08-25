use crate::multilinear_product::{SchedulingParams, compute_eval_product_schedule};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_d2_n6_k2_schedule() {
        let params = SchedulingParams { d: 2, num_variables: 6, num_stages: 2 };
        let schedule = compute_eval_product_schedule(&params);
        
        println!("D=2, n=6, k=2 EvalProduct schedule: {:?}", schedule);
        
        // Expected hard stop: 6 - floor(6/2) = 6 - 3 = 3
        let expected_hard_stop = 6 - (6 / 2);
        println!("Expected hard stop: {}", expected_hard_stop);
        
        // Check if schedule respects hard stop
        let max_round = schedule.iter().max().copied().unwrap_or(0);
        println!("Max round in schedule: {}", max_round);
        
        assert!(max_round <= expected_hard_stop, "Schedule exceeds hard stop");
    }
}
