use ndarray::{ArrayBase, Data, Dimension, Zip};
use num_traits::{Signed, ToPrimitive};
use std::convert::Into;
use std::ops::AddAssign;

use crate::errors::{MultiInputError, ShapeMismatch};

/// Extension trait for `ArrayBase` providing functions
/// to compute different deviation measures.
pub trait DeviationExt<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn count_eq(&self, other: &ArrayBase<S, D>) -> Result<usize, MultiInputError>
    where
        A: PartialEq;

    fn count_neq(&self, other: &ArrayBase<S, D>) -> Result<usize, MultiInputError>
    where
        A: PartialEq;

    fn sq_l2_dist(&self, other: &ArrayBase<S, D>) -> Result<A, MultiInputError>
    where
        A: AddAssign + Clone + Signed;

    fn l2_dist(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive;

    fn l1_dist(&self, other: &ArrayBase<S, D>) -> Result<A, MultiInputError>
    where
        A: AddAssign + Clone + Signed;

    fn linf_dist(&self, other: &ArrayBase<S, D>) -> Result<A, MultiInputError>
    where
        A: Clone + PartialOrd + Signed;

    fn mean_abs_dev(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive;

    fn max_abs_dev(&self, other: &ArrayBase<S, D>) -> Result<A, MultiInputError>
    where
        A: Clone + PartialOrd + Signed;

    fn mean_sq_dev(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive;

    fn root_mean_sq_dev(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive;

    fn peak_signal_to_noise_ratio(
        &self,
        other: &ArrayBase<S, D>,
        maxv: A,
    ) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive;

    private_decl! {}
}

macro_rules! return_err_if_empty {
    ($arr:expr) => {
        if $arr.len() == 0 {
            return Err(MultiInputError::EmptyInput);
        }
    };
}
macro_rules! return_err_unless_same_shape {
    ($arr_a:expr, $arr_b:expr) => {
        if $arr_a.shape() != $arr_b.shape() {
            return Err(ShapeMismatch {
                first_shape: $arr_a.shape().to_vec(),
                second_shape: $arr_b.shape().to_vec(),
            }
            .into());
        }
    };
}

impl<A, S, D> DeviationExt<A, S, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn count_eq(&self, other: &ArrayBase<S, D>) -> Result<usize, MultiInputError>
    where
        A: PartialEq,
    {
        return_err_if_empty!(self);
        return_err_unless_same_shape!(self, other);

        let mut count = 0;

        Zip::from(self).and(other).apply(|a, b| {
            if a == b {
                count += 1;
            }
        });

        Ok(count)
    }

    fn count_neq(&self, other: &ArrayBase<S, D>) -> Result<usize, MultiInputError>
    where
        A: PartialEq,
    {
        self.count_eq(other).map(|n_eq| self.len() - n_eq)
    }

    fn sq_l2_dist(&self, other: &ArrayBase<S, D>) -> Result<A, MultiInputError>
    where
        A: AddAssign + Clone + Signed,
    {
        return_err_if_empty!(self);
        return_err_unless_same_shape!(self, other);

        let mut result = A::zero();

        Zip::from(self).and(other).apply(|self_i, other_i| {
            let (a, b) = (self_i.clone(), other_i.clone());
            let abs_diff = (a - b).abs();
            result += abs_diff.clone() * abs_diff;
        });

        Ok(result)
    }

    fn l2_dist(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive,
    {
        let sq_l2_dist = self
            .sq_l2_dist(other)?
            .to_f64()
            .expect("failed cast from type A to f64");

        Ok(sq_l2_dist.sqrt())
    }

    fn l1_dist(&self, other: &ArrayBase<S, D>) -> Result<A, MultiInputError>
    where
        A: AddAssign + Clone + Signed,
    {
        return_err_if_empty!(self);
        return_err_unless_same_shape!(self, other);

        let mut result = A::zero();

        Zip::from(self).and(other).apply(|self_i, other_i| {
            let (a, b) = (self_i.clone(), other_i.clone());
            result += (a - b).abs();
        });

        Ok(result)
    }

    fn linf_dist(&self, other: &ArrayBase<S, D>) -> Result<A, MultiInputError>
    where
        A: Clone + PartialOrd + Signed,
    {
        return_err_if_empty!(self);
        return_err_unless_same_shape!(self, other);

        let mut max = A::zero();

        Zip::from(self).and(other).apply(|self_i, other_i| {
            let (a, b) = (self_i.clone(), other_i.clone());
            let diff = (a - b).abs();
            if diff > max {
                max = diff;
            }
        });

        Ok(max)
    }

    fn mean_abs_dev(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive,
    {
        let a = self
            .l1_dist(other)?
            .to_f64()
            .expect("failed cast from type A to f64");
        let b = self.len() as f64;

        Ok(a / b)
    }

    #[inline]
    fn max_abs_dev(&self, other: &ArrayBase<S, D>) -> Result<A, MultiInputError>
    where
        A: Clone + PartialOrd + Signed,
    {
        self.linf_dist(other)
    }

    fn mean_sq_dev(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive,
    {
        let a = self
            .sq_l2_dist(other)?
            .to_f64()
            .expect("failed cast from type A to f64");
        let b = self.len() as f64;

        Ok(a / b)
    }

    fn root_mean_sq_dev(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive,
    {
        let msd = self.mean_sq_dev(other)?;
        Ok(msd.sqrt())
    }

    fn peak_signal_to_noise_ratio(
        &self,
        other: &ArrayBase<S, D>,
        maxv: A,
    ) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive,
    {
        let maxv_f = maxv.to_f64().expect("failed cast from type A to f64");
        let msd = self.mean_sq_dev(&other)?;
        let psnr = 10. * f64::log10(maxv_f * maxv_f / msd);

        Ok(psnr)
    }

    private_impl! {}
}
