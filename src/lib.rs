use num_traits::{One, Zero};
use pyo3::prelude::*;
use pyo3::Python;
use qip_iterators::iterators::MatrixOp;
use qip_iterators::matrix_ops::{apply_op, apply_op_overwrite, apply_op_row};
use rayon::prelude::*;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul};

#[pyclass]
pub struct TensorMatf64 {
    mat: MatrixTree<f64>,
}

#[pymethods]
impl TensorMatf64 {
    #[new]
    fn new(indices: Vec<usize>, data: Vec<f64>) -> Self {
        Self {
            mat: MatrixTree::Leaf(MatrixOp::new_matrix(indices, data)),
        }
    }

    fn __add__(&self, other: &Self) -> Self {
        let mat = self.mat.clone().add(other.mat.clone());
        Self { mat }
    }

    fn __matmul__(&self, other: &Self) -> Self {
        let mat = self.mat.clone().mul(other.mat.clone());
        Self { mat }
    }
}

#[derive(Clone)]
pub enum MatrixTree<P> {
    Leaf(MatrixOp<P>),
    Sum(Vec<MatrixTree<P>>),
    Prod(Vec<MatrixTree<P>>),
    SumLeaf(Vec<MatrixOp<P>>),
    ProdLeaf(Vec<MatrixOp<P>>),
}

impl<P> MatrixTree<P>
where
    P: Sync + Send + Zero + One + Sum + Clone + AddAssign,
{
    pub fn apply_add(&self, n: usize, input: &[P], output: &mut [P]) {
        match self {
            MatrixTree::Leaf(op) => apply_op(n, op, input, output, 0, 0),
            MatrixTree::SumLeaf(ops) => {
                output.par_iter_mut().enumerate().for_each(|(i, x)| {
                    *x += ops
                        .iter()
                        .map(|op| apply_op_row(n, op, input, i, 0, 0))
                        .sum()
                });
            }
            MatrixTree::Sum(ops) => ops.iter().for_each(|op| op.apply_add(n, input, output)),
            MatrixTree::ProdLeaf(_) | MatrixTree::Prod(_) => {
                let out_copy = output.to_vec();
                self.apply_overwrite(n, input, output);
                output
                    .iter_mut()
                    .zip(out_copy.into_iter())
                    .for_each(|(x, a)| *x += a);
            }
        }
    }

    pub fn apply_overwrite(&self, n: usize, input: &[P], output: &mut [P]) {
        match self {
            MatrixTree::Leaf(op) => apply_op_overwrite(n, op, input, output, 0, 0),
            MatrixTree::SumLeaf(ops) => {
                output.par_iter_mut().enumerate().for_each(|(i, x)| {
                    *x = ops
                        .iter()
                        .map(|op| apply_op_row(n, op, input, i, 0, 0))
                        .sum()
                });
            }
            MatrixTree::Sum(ops) => {
                output.iter_mut().for_each(|x| *x = P::zero());
                ops.iter().for_each(|op| op.apply_add(n, input, output))
            }
            MatrixTree::ProdLeaf(ops) => match ops.as_slice() {
                [] => output
                    .iter_mut()
                    .zip(input.iter())
                    .for_each(|(a, b)| *a = b.clone()),
                [op] => {
                    apply_op_overwrite(n, op, input, output, 0, 0);
                }
                [op, ops @ ..] => {
                    let mut buffer = vec![P::zero(); output.len()];
                    let (buffa, buffb) = if ops.len() % 2 == 0 {
                        (output, buffer.as_mut_slice())
                    } else {
                        (buffer.as_mut_slice(), output)
                    };
                    apply_op_overwrite(n, op, input, buffa, 0, 0);
                    ops.iter().fold((buffa, buffb), |(buffa, buffb), op| {
                        apply_op_overwrite(n, op, buffa, buffb, 0, 0);
                        (buffb, buffa)
                    });
                }
            },
            MatrixTree::Prod(ops) => match ops.as_slice() {
                [] => output
                    .iter_mut()
                    .zip(input.iter())
                    .for_each(|(a, b)| *a = b.clone()),
                [op] => op.apply_overwrite(n, input, output),
                [op, ops @ ..] => {
                    let mut buffer = vec![P::zero(); output.len()];
                    let (buffa, buffb) = if ops.len() % 2 == 0 {
                        (output, buffer.as_mut_slice())
                    } else {
                        (buffer.as_mut_slice(), output)
                    };
                    op.apply_overwrite(n, input, buffa);
                    ops.iter().fold((buffa, buffb), |(buffa, buffb), op| {
                        op.apply_overwrite(n, buffa, buffb);
                        (buffb, buffa)
                    });
                }
            },
        }
    }
}

impl<P> Add for MatrixTree<P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (MatrixTree::Leaf(a), MatrixTree::Leaf(b)) => MatrixTree::SumLeaf(vec![a, b]),
            (MatrixTree::SumLeaf(mut a), MatrixTree::Leaf(b))
            | (MatrixTree::Leaf(b), MatrixTree::SumLeaf(mut a)) => {
                a.push(b);
                MatrixTree::SumLeaf(a)
            }
            (MatrixTree::SumLeaf(mut a), MatrixTree::SumLeaf(b)) => {
                a.extend(b);
                MatrixTree::SumLeaf(a)
            }
            (MatrixTree::Sum(mut a), MatrixTree::Sum(b)) => {
                a.extend(b);
                MatrixTree::Sum(a)
            }
            (MatrixTree::Sum(mut a), b) | (b, MatrixTree::Sum(mut a)) => {
                a.push(b);
                MatrixTree::Sum(a)
            }
            (a, b) => MatrixTree::Sum(vec![a, b]),
        }
    }
}

impl<P> Mul for MatrixTree<P> {
    type Output = MatrixTree<P>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (MatrixTree::Leaf(a), MatrixTree::Leaf(b)) => MatrixTree::ProdLeaf(vec![a, b]),
            (MatrixTree::ProdLeaf(mut a), MatrixTree::Leaf(b)) => {
                a.push(b);
                MatrixTree::ProdLeaf(a)
            }
            (MatrixTree::Leaf(a), MatrixTree::ProdLeaf(mut b)) => {
                b.insert(0, a);
                MatrixTree::ProdLeaf(b)
            }
            (MatrixTree::ProdLeaf(mut a), MatrixTree::ProdLeaf(b)) => {
                a.extend(b);
                MatrixTree::ProdLeaf(a)
            }
            (MatrixTree::Prod(mut a), MatrixTree::Prod(b)) => {
                a.extend(b);
                MatrixTree::Prod(a)
            }
            (MatrixTree::Prod(mut a), b) => {
                a.push(b);
                MatrixTree::Prod(a)
            }
            (a, MatrixTree::Prod(mut b)) => {
                b.insert(0, a);
                MatrixTree::Prod(b)
            }
            (a, b) => MatrixTree::Prod(vec![a, b]),
        }
    }
}

#[pymodule]
fn qubit_matmul(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<TensorMatf64>()?;
    Ok(())
}
