#[cfg(test)]
use crate::interpolation::multivariate::multi_product_eval_paper;
#[cfg(test)]
use crate::tests::BN254;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multivariate_d2_omega1() {
        // Test the multivariate computation for D=2, omega=1
        // This should match what happens in the first window of EvalProduct
        
        // Create simple test polynomials on {0,1}^1
        let poly1 = vec![BN254::from(1u64), BN254::from(2u64)]; // f1(0)=1, f1(1)=2
        let poly2 = vec![BN254::from(3u64), BN254::from(4u64)]; // f2(0)=3, f2(1)=4
        let polys = vec![poly1, poly2];
        
        println!("Input polynomials:");
        println!("  poly1: {:?}", polys[0]);
        println!("  poly2: {:?}", polys[1]);
        
        // Expected product: f1*f2
        // (f1*f2)(0) = 1*3 = 3
        // (f1*f2)(1) = 2*4 = 8
        println!("Expected product on {{0,1}}: [3, 8]");
        
        // Compute using multivariate function
        let result = multi_product_eval_paper::<BN254>(1, &polys, 2);
        println!("Multivariate result: {:?}", result);
        
        // The result should be evaluations on U_2^1 = {0, 1, ∞}^1 = {0, 1, ∞}
        // So result should be [f(0), f(1), f(∞)]
        // where f(∞) is the leading coefficient of the product
        
        // For degree-1 polynomials f1(x) = 1 + x, f2(x) = 3 + x
        // Product: (1+x)(3+x) = 3 + 4x + x^2
        // So f(0) = 3, f(1) = 8, f(∞) = 1 (leading coeff)
        
        assert_eq!(result.len(), 3, "Should have 3 evaluations for U_2");
        assert_eq!(result[0], BN254::from(3u64), "f(0) should be 3");
        assert_eq!(result[1], BN254::from(8u64), "f(1) should be 8"); 
        assert_eq!(result[2], BN254::from(1u64), "f(∞) should be 1 (leading coeff)");
    }
}
