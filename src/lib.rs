use num_traits::{Num, One, Zero};
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadwriteArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::Python;
use qip_iterators::iterators::{act_on_iterator, MatrixOp};
use qip_iterators::matrix_ops::{
    apply_op, apply_op_overwrite, apply_op_row, full_to_sub, get_index, sub_to_full,
};
use rayon::prelude::*;
#[cfg(feature = "sparse")]
use sprs::*;
use std::iter::Sum;
use std::ops::{Add, AddAssign, DivAssign, Mul, MulAssign, Neg};

// Repeated code because pyo3 and macros not playing well together.
macro_rules! tensor_class {
    (nonegate, $name:ident, $t:ty) => {
        #[pyclass]
        pub struct $name {
            mat: MatrixTree<$t>,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new(indices: Vec<usize>, data: Vec<$t>) -> Self {
                Self {
                    mat: MatrixTree::Leaf(MatrixOp::new_matrix(indices, data).into()),
                }
            }

            fn apply(
                &self,
                py: Python,
                input: PyReadonlyArray1<$t>,
                output: Option<PyReadwriteArray1<$t>>,
            ) -> PyResult<Option<Py<PyArray1<$t>>>> {
                let input_slice = input
                    .as_slice()
                    .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
                let len = input_slice.len();
                if !len.is_power_of_two() {
                    return Err(PyValueError::new_err("Input array must be of length 2^n"));
                }
                if let Some(mut output) = output {
                    let output_slice = output
                        .as_slice_mut()
                        .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
                    let n = two_power(len) as usize;
                    self.mat.apply_overwrite(n, input_slice, output_slice);
                    Ok(None)
                } else {
                    let mut output = Array1::zeros((len,));
                    let output_slice = output.as_slice_mut().unwrap();
                    let n = two_power(len) as usize;
                    self.mat.apply_overwrite(n, input_slice, output_slice);
                    Ok(Some(output.to_pyarray(py).to_owned()))
                }
            }

            #[cfg(feature = "sparse")]
            fn make_sparse(
                &self,
                py: Python,
                n: usize,
            ) -> (Py<PyArray1<usize>>, Py<PyArray1<usize>>, Py<PyArray1<$t>>) {
                let sprs = self.mat.make_sparse(n);

                let nn = sprs.iter().count();
                let mut vals = Array1::zeros((nn,));
                let mut rows = Array1::zeros((nn,));
                let mut cols = Array1::zeros((nn,));
                sprs.into_iter()
                    .enumerate()
                    .for_each(|(i, (x, (row, col)))| {
                        rows[i] = row;
                        cols[i] = col;
                        vals[i] = *x;
                    });
                let vals = vals.into_pyarray(py).to_owned();
                let rows = rows.into_pyarray(py).to_owned();
                let cols = cols.into_pyarray(py).to_owned();
                (rows, cols, vals)
            }

            #[cfg(feature = "sparse")]
            fn get_sparse<'a>(&self, py: Python<'a>, n: usize) -> PyResult<&'a PyAny> {
                let sprs = self.mat.make_sparse(n);
                scipy_mat(py, &sprs)
                    .map(|e| e)
                    .map_err(|e| PyValueError::new_err(e))
            }

            fn get_dense(&self, py: Python, n: usize) -> Py<PyArray2<$t>> {
                self.mat.make_dense(n).into_pyarray(py).to_owned()
            }

            fn __add__(&self, other: &Self) -> Self {
                let mat = self.mat.clone().add(other.mat.clone());
                Self { mat }
            }

            fn __matmul__(&self, other: &Self) -> Self {
                let mat = self.mat.clone().mul(other.mat.clone());
                Self { mat }
            }

            fn __sub__(&self, _other: &Self) -> PyResult<Self> {
                Err(PyValueError::new_err("Cannot negate this array type"))
            }

            fn __neg__(&self) -> PyResult<Self> {
                Err(PyValueError::new_err("Cannot negate this array type"))
            }
        }
    };

    ($name:ident, $t:ty) => {
        #[pyclass]
        pub struct $name {
            mat: MatrixTree<$t>,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new(indices: Vec<usize>, data: Vec<$t>) -> Self {
                Self {
                    mat: MatrixTree::Leaf(MatrixOp::new_matrix(indices, data).into()),
                }
            }

            fn apply(
                &self,
                py: Python,
                input: PyReadonlyArray1<$t>,
                output: Option<PyReadwriteArray1<$t>>,
            ) -> PyResult<Option<Py<PyArray1<$t>>>> {
                let input_slice = input
                    .as_slice()
                    .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
                let len = input_slice.len();
                if !len.is_power_of_two() {
                    return Err(PyValueError::new_err("Input array must be of length 2^n"));
                }
                if let Some(mut output) = output {
                    let output_slice = output
                        .as_slice_mut()
                        .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
                    let n = two_power(len) as usize;
                    self.mat.apply_overwrite(n, input_slice, output_slice);
                    Ok(None)
                } else {
                    let mut output = Array1::zeros((len,));
                    let output_slice = output.as_slice_mut().unwrap();
                    let n = two_power(len) as usize;
                    self.mat.apply_overwrite(n, input_slice, output_slice);
                    Ok(Some(output.to_pyarray(py).to_owned()))
                }
            }

            #[cfg(feature = "sparse")]
            fn make_sparse(
                &self,
                py: Python,
                n: usize,
            ) -> (Py<PyArray1<usize>>, Py<PyArray1<usize>>, Py<PyArray1<$t>>) {
                let sprs = self.mat.make_sparse(n);

                let nn = sprs.iter().count();
                let mut vals = Array1::zeros((nn,));
                let mut rows = Array1::zeros((nn,));
                let mut cols = Array1::zeros((nn,));
                sprs.into_iter()
                    .enumerate()
                    .for_each(|(i, (x, (row, col)))| {
                        rows[i] = row;
                        cols[i] = col;
                        vals[i] = *x;
                    });
                let vals = vals.into_pyarray(py).to_owned();
                let rows = rows.into_pyarray(py).to_owned();
                let cols = cols.into_pyarray(py).to_owned();
                (rows, cols, vals)
            }

            #[cfg(feature = "sparse")]
            fn get_sparse<'a>(&self, py: Python<'a>, n: usize) -> PyResult<&'a PyAny> {
                let sprs = self.mat.make_sparse(n);
                scipy_mat(py, &sprs)
                    .map(|e| e)
                    .map_err(|e| PyValueError::new_err(e))
            }

            fn get_dense(&self, py: Python, n: usize) -> Py<PyArray2<$t>> {
                self.mat.make_dense(n).into_pyarray(py).to_owned()
            }

            fn __add__(&self, other: &Self) -> Self {
                let mat = self.mat.clone().add(other.mat.clone());
                Self { mat }
            }

            fn __matmul__(&self, other: &Self) -> Self {
                let mat = self.mat.clone().mul(other.mat.clone());
                Self { mat }
            }

            fn __sub__(&self, other: &Self) -> PyResult<Self> {
                let other = other.__neg__()?;
                let mat = self.mat.clone().add(other.mat);
                Ok(Self { mat })
            }

            fn __neg__(&self) -> PyResult<Self> {
                let mut mat = self.mat.clone();
                mat.negate().map_err(PyValueError::new_err)?;
                Ok(Self { mat })
            }

            fn __mul__(&self, other: $t) -> PyResult<Self> {
                let mat = self.mat.clone();
                Ok(Self {
                    mat: mat.scalar_mul(other),
                })
            }

            fn __div__(&self, other: $t) -> PyResult<Self> {
                let mat = self.mat.clone();
                Ok(Self {
                    mat: mat.scalar_div(other),
                })
            }
        }
    };
}

fn two_power(x: usize) -> u32 {
    let x = x.next_power_of_two();
    let leading = x.leading_zeros();
    // leading    two_power
    // BITS-1     0
    // BITS-2     1
    // BITS-3     2
    // so two_power = (BITS-1) - leading
    (usize::BITS - 1) - leading
}

#[cfg(feature = "sparse")]
fn scipy_mat<'a, P>(py: Python<'a>, mat: &CsMat<P>) -> Result<&'a PyAny, String>
where
    P: Clone + IntoPy<Py<PyAny>>,
{
    let scipy_sparse = PyModule::import(py, "scipy.sparse").map_err(|e| {
        let res = format!("Python error: {e:?}");
        e.print_and_set_sys_last_vars(py);
        res
    })?;
    let indptr = mat.indptr().to_proper().to_vec();
    scipy_sparse
        .call_method(
            "csr_matrix",
            ((mat.data().to_vec(), mat.indices().to_vec(), indptr),),
            Some([("shape", mat.shape())].into_py_dict(py)),
        )
        .map_err(|e| {
            let res = format!("Python error: {e:?}");
            e.print_and_set_sys_last_vars(py);
            res
        })
}

#[derive(Clone)]
pub enum MatrixTree<P> {
    Leaf(MatrixOp<P>),
    Sum(Vec<MatrixTree<P>>),
    Prod(Vec<MatrixTree<P>>),
    SumLeaf(Vec<MatrixOp<P>>),
    ProdLeaf(Vec<MatrixOp<P>>),
    Mul(Box<MatrixTree<P>>, P),
    Div(Box<MatrixTree<P>>, P),
}

#[cfg(feature = "sparse")]
impl<P> MatrixTree<P>
where
    P: Add + Num + Copy + Default + MulAcc + Zero + Send + Sync + One + MulAssign + DivAssign,
    for<'r> &'r P: Add<&'r P, Output = P>,
{
    fn make_sparse(&self, n: usize) -> CsMat<P> {
        match self {
            MatrixTree::Leaf(op) => make_sparse_from_op(op, n),
            MatrixTree::SumLeaf(ops) => ops
                .iter()
                .map(|op| make_sparse_from_op(op, n))
                .fold(None, |acc, x| match acc {
                    None => Some(x),
                    Some(acc) => Some(&acc + &x),
                })
                .unwrap_or_else(|| CsMat::zero((1 << n, 1 << n))),
            MatrixTree::Sum(ops) => ops
                .iter()
                .map(|op| op.make_sparse(n))
                .fold(None, |acc, x| match acc {
                    None => Some(x),
                    Some(acc) => Some(&acc + &x),
                })
                .unwrap_or_else(|| CsMat::zero((1 << n, 1 << n))),
            MatrixTree::ProdLeaf(ops) => ops
                .iter()
                .map(|op| make_sparse_from_op(op, n))
                .fold(None, |acc, x| match acc {
                    None => Some(x),
                    Some(acc) => Some(&acc * &x),
                })
                .unwrap_or_else(|| CsMat::zero((1 << n, 1 << n))),
            MatrixTree::Prod(ops) => ops
                .iter()
                .map(|op| op.make_sparse(n))
                .fold(None, |acc, x| match acc {
                    None => Some(x),
                    Some(acc) => Some(&acc * &x),
                })
                .unwrap_or_else(|| CsMat::zero((1 << n, 1 << n))),
            MatrixTree::Mul(tree, mul) => {
                let mut sparse = tree.make_sparse(n);
                sparse.data_mut().iter_mut().for_each(|x| *x *= *mul);
                sparse
            }
            MatrixTree::Div(tree, div) => {
                let mut sparse = tree.make_sparse(n);
                sparse.data_mut().iter_mut().for_each(|x| *x /= *div);
                sparse
            }
        }
    }
}

fn make_sparse_from_op<P>(op: &MatrixOp<P>, n: usize) -> CsMat<P>
where
    P: Clone + Zero + One + Num,
{
    let mut a = TriMat::new((1 << n, 1 << n));
    let nindices = op.num_indices();

    let mat_indices: Vec<usize> = (0..op.num_indices()).map(|i| get_index(op, i)).collect();
    for row in 0..a.shape().0 {
        let matrow = full_to_sub(n, &mat_indices, row);
        act_on_iterator(nindices, matrow, op, |it| {
            let f = |(i, val): (usize, P)| {
                let vecrow = sub_to_full(n, &mat_indices, i, row);
                (vecrow, val)
            };

            for (col, val) in it.map(f) {
                a.add_triplet(row, col, val)
            }
        })
    }
    a.to_csr()
}

impl<P> MatrixTree<P>
where
    P: Add
        + Num
        + Copy
        + Default
        + MulAcc
        + Zero
        + Send
        + Sync
        + One
        + MulAssign
        + DivAssign
        + 'static,
    for<'r> &'r P: Add<&'r P, Output = P>,
{
    fn make_dense(&self, n: usize) -> Array2<P> {
        match self {
            MatrixTree::Leaf(op) => make_dense_from_op(op, n),
            MatrixTree::SumLeaf(ops) => ops
                .iter()
                .map(|op| make_dense_from_op(op, n))
                .fold(None, |acc, x| match acc {
                    None => Some(x),
                    Some(acc) => Some(&acc + &x),
                })
                .unwrap_or_else(|| Array2::zeros((1 << n, 1 << n))),
            MatrixTree::Sum(ops) => ops
                .iter()
                .map(|op| op.make_dense(n))
                .fold(None, |acc, x| match acc {
                    None => Some(x),
                    Some(acc) => Some(&acc + &x),
                })
                .unwrap_or_else(|| Array2::zeros((1 << n, 1 << n))),
            MatrixTree::ProdLeaf(ops) => ops
                .iter()
                .map(|op| make_dense_from_op(op, n))
                .fold(None, |acc, x| match acc {
                    None => Some(x),
                    Some(acc) => Some(acc.dot(&x)),
                })
                .unwrap_or_else(|| Array2::zeros((1 << n, 1 << n))),
            MatrixTree::Prod(ops) => ops
                .iter()
                .map(|op| op.make_dense(n))
                .fold(None, |acc, x| match acc {
                    None => Some(x),
                    Some(acc) => Some(acc.dot(&x)),
                })
                .unwrap_or_else(|| Array2::zeros((1 << n, 1 << n))),
            MatrixTree::Mul(tree, mul) => {
                let mut dense = tree.make_dense(n);
                dense.iter_mut().for_each(|x| *x *= *mul);
                dense
            }
            MatrixTree::Div(tree, div) => {
                let mut dense = tree.make_dense(n);
                dense.iter_mut().for_each(|x| *x /= *div);
                dense
            }
        }
    }
}

fn make_dense_from_op<P>(op: &MatrixOp<P>, n: usize) -> Array2<P>
where
    P: Clone + Zero + One + Num,
{
    let mut a = Array2::zeros((1 << n, 1 << n));
    let nindices = op.num_indices();

    let mat_indices: Vec<usize> = (0..op.num_indices()).map(|i| get_index(op, i)).collect();
    for row in 0..a.shape()[0] {
        let matrow = full_to_sub(n, &mat_indices, row);
        act_on_iterator(nindices, matrow, op, |it| {
            let f = |(i, val): (usize, P)| {
                let vecrow = sub_to_full(n, &mat_indices, i, row);
                (vecrow, val)
            };

            for (col, val) in it.map(f) {
                a[(row, col)] = val;
            }
        })
    }
    a
}

impl<P> MatrixTree<P>
where
    P: Sync + Send + Zero + One + Sum + Copy + Clone + AddAssign + MulAssign + DivAssign,
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
            _ => {
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
                    .for_each(|(a, b)| *a = *b),
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
                    .for_each(|(a, b)| *a = *b),
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
            MatrixTree::Mul(tree, mul) => {
                tree.apply_overwrite(n, input, output);
                output.iter_mut().for_each(|x| *x *= *mul);
            }

            MatrixTree::Div(tree, div) => {
                tree.apply_overwrite(n, input, output);
                output.iter_mut().for_each(|x| *x /= *div);
            }
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

fn negate_op<P>(op: &mut MatrixOp<P>) -> Result<(), String>
where
    P: Neg<Output = P> + Clone,
{
    match op {
        MatrixOp::Matrix(_, d) => d.iter_mut().for_each(|x| *x = x.clone().neg()),
        MatrixOp::SparseMatrix(_, d) => d
            .iter_mut()
            .flat_map(|x| x.iter_mut())
            .for_each(|(_, x)| *x = x.clone().neg()),
        MatrixOp::Swap(_, _) => {
            return Err("Negating swap operations not yet implemented".to_string())
        }
        MatrixOp::Control(_, _, _) => {
            return Err("Negating control operations not yet implemented".to_string())
        }
    }
    Ok(())
}

impl<P> MatrixTree<P>
where
    P: Neg<Output = P> + Copy,
{
    fn negate(&mut self) -> Result<(), String> {
        match self {
            MatrixTree::Leaf(x) => negate_op(x),
            MatrixTree::SumLeaf(x) => x.iter_mut().try_for_each(|x| negate_op(x)),
            MatrixTree::ProdLeaf(x) => {
                if let Some(x) = x.first_mut() {
                    negate_op(x)
                } else {
                    Ok(())
                }
            }
            MatrixTree::Sum(x) => x.iter_mut().try_for_each(|x| x.negate()),
            MatrixTree::Prod(x) => {
                if let Some(x) = x.first_mut() {
                    x.negate()
                } else {
                    Ok(())
                }
            }
            MatrixTree::Mul(_, mul) => {
                *mul = mul.neg();
                Ok(())
            }
            MatrixTree::Div(_, div) => {
                *div = div.neg();
                Ok(())
            }
        }
    }
}

impl<P> MatrixTree<P> {
    fn scalar_mul(self, other: P) -> Self {
        MatrixTree::Mul(Box::new(self), other)
    }
    fn scalar_div(self, other: P) -> Self {
        MatrixTree::Div(Box::new(self), other)
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

// tensor_class!(TensorMatc64, Complex64);
// tensor_class!(TensorMatc32, Complex32);
tensor_class!(TensorMatf64, f64);
tensor_class!(TensorMatf32, f32);
tensor_class!(TensorMati64, i64);
tensor_class!(TensorMati32, i32);
tensor_class!(nonegate, TensorMatu64, u64);
tensor_class!(nonegate, TensorMatu32, u32);

#[pymodule]
fn qubit_matmul(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // m.add_class::<TensorMatc64>()?;
    // m.add_class::<TensorMatc32>()?;
    m.add_class::<TensorMatf64>()?;
    m.add_class::<TensorMatf32>()?;
    m.add_class::<TensorMati64>()?;
    m.add_class::<TensorMati32>()?;
    m.add_class::<TensorMatu64>()?;
    m.add_class::<TensorMatu32>()?;
    Ok(())
}
