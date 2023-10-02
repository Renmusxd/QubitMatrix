use num_traits::Zero;
use std::collections::HashMap;
use std::iter::Sum;
use std::ops::{Add, Div, DivAssign, Mul, MulAssign};

#[derive(Debug, Copy, Clone)]
enum MajorAxis {
    Column,
    Row,
}

#[derive(Debug, Clone)]
pub struct HashSparse<P> {
    major: MajorAxis,
    data: HashMap<usize, HashMap<usize, P>>,
    shape: (usize, usize),
}

impl<P> HashSparse<P>
where
    P: Zero,
{
    pub fn empty_like(&self) -> Self {
        match self.major {
            MajorAxis::Column => Self::new_col_major(self.shape),
            MajorAxis::Row => Self::new_row_major(self.shape),
        }
    }

    pub fn empty_like_shape(&self, shape: (usize, usize)) -> Self {
        match self.major {
            MajorAxis::Column => Self::new_col_major(shape),
            MajorAxis::Row => Self::new_row_major(shape),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn new_row_major(shape: (usize, usize)) -> Self {
        Self {
            shape,
            major: MajorAxis::Row,
            data: Default::default(),
        }
    }

    pub fn new_col_major(shape: (usize, usize)) -> Self {
        Self {
            shape,
            major: MajorAxis::Column,
            data: Default::default(),
        }
    }

    pub fn nonempty(&self) -> usize {
        self.data.values().map(|dat| dat.len()).sum()
    }

    pub fn into_row_major(self) -> Self {
        match self.major {
            MajorAxis::Row => self,
            MajorAxis::Column => {
                let mut a = Self::new_row_major(self.shape);
                self.data
                    .into_iter()
                    .flat_map(move |(col, data)| {
                        data.into_iter().map(move |(row, data)| (col, row, data))
                    })
                    .for_each(|(col, row, data)| {
                        a.insert(row, col, data);
                    });
                a
            }
        }
    }

    pub fn into_col_major(self) -> Self {
        match self.major {
            MajorAxis::Column => self,
            MajorAxis::Row => {
                let mut a = Self::new_col_major(self.shape);
                self.data
                    .into_iter()
                    .flat_map(move |(row, data)| {
                        data.into_iter().map(move |(col, data)| (col, row, data))
                    })
                    .for_each(|(col, row, data)| {
                        a.insert(row, col, data);
                    });
                a
            }
        }
    }

    pub fn insert(&mut self, row: usize, col: usize, val: P) {
        match self.major {
            MajorAxis::Column => {
                let entry = self.data.entry(col);
                let sub = entry.or_insert(Default::default());
                sub.insert(row, val);
            }
            MajorAxis::Row => {
                let entry = self.data.entry(row);
                let sub = entry.or_insert(Default::default());
                sub.insert(col, val);
            }
        }
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut P {
        match self.major {
            MajorAxis::Column => {
                let entry = self.data.entry(col);
                let sub = entry.or_insert(Default::default());
                sub.entry(row).or_insert(P::zero())
            }
            MajorAxis::Row => {
                let entry = self.data.entry(row);
                let sub = entry.or_insert(Default::default());
                sub.entry(col).or_insert(P::zero())
            }
        }
    }

    pub fn get_mut_option(&mut self, row: usize, col: usize) -> Option<&mut P> {
        match self.major {
            MajorAxis::Column => {
                let entry = self.data.entry(col);
                let sub = entry.or_insert(Default::default());
                sub.get_mut(&row)
            }
            MajorAxis::Row => {
                let entry = self.data.entry(row);
                let sub = entry.or_insert(Default::default());
                sub.get_mut(&col)
            }
        }
    }

    pub fn into_iter_coords(self) -> impl Iterator<Item = (usize, usize, P)> {
        let major = self.major;
        self.data.into_iter().flat_map(move |(a, data)| {
            data.into_iter().map(move |(b, data)| match major {
                MajorAxis::Row => (a, b, data),
                MajorAxis::Column => (b, a, data),
            })
        })
    }

    pub fn iter_coords(&self) -> impl Iterator<Item = (&usize, &usize, &P)> {
        let major = self.major;
        self.data.iter().flat_map(move |(a, data)| {
            data.iter().map(move |(b, data)| match major {
                MajorAxis::Row => (a, b, data),
                MajorAxis::Column => (b, a, data),
            })
        })
    }

    pub fn iter_coords_mut(&mut self) -> impl Iterator<Item = (&usize, &usize, &mut P)> {
        let major = self.major;
        self.data.iter_mut().flat_map(move |(a, data)| {
            data.iter_mut().map(move |(b, data)| match major {
                MajorAxis::Row => (a, b, data),
                MajorAxis::Column => (b, a, data),
            })
        })
    }

    pub fn filter_to_rows(self, rows: &[usize]) -> Self {
        let original_major = self.major;
        let mut new_self = self.into_row_major();

        new_self.data = rows
            .iter()
            .copied()
            .filter_map(|row| new_self.data.remove(&row).map(|arr| (row, arr)))
            .collect();

        match original_major {
            MajorAxis::Column => new_self.into_col_major(),
            MajorAxis::Row => new_self.into_row_major(),
        }
    }
}

impl<P> HashSparse<P>
where
    P: Add + Mul + Zero + Sum + Clone,
    for<'r> &'r P: Mul<&'r P, Output = P>,
{
    pub fn matmul(&self, rhs: &Self) -> Self {
        let mut out = match self.major {
            MajorAxis::Column => Self::new_col_major(self.shape),
            MajorAxis::Row => Self::new_row_major(self.shape),
        };
        let a = self.clone().into_row_major();
        let b = rhs.clone().into_col_major();

        a.data.into_iter().for_each(|(row, ldata)| {
            b.data.iter().for_each(|(col, rdata)| {
                let dot = ldata
                    .iter()
                    .filter_map(|(x, lv)| rdata.get(x).map(|rv| lv * rv))
                    .sum::<P>();
                out.insert(row, *col, dot);
            });
        });

        out
    }
}

impl<P> Add for HashSparse<P>
where
    P: Add + Clone + Zero,
    for<'r> &'r P: Add<&'r P, Output = P>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut res = self.clone();
        rhs.iter_coords().for_each(|(row, col, val)| {
            let get = res.get_mut(*row, *col);
            *get = val + get;
        });
        res
    }
}

impl<P> Mul<&P> for HashSparse<P>
where
    P: Mul + Clone + Zero,
    for<'r> &'r P: Mul<&'r P, Output = P>,
{
    type Output = Self;

    fn mul(self, rhs: &P) -> Self {
        let mut res = self.clone();
        res.iter_coords_mut().for_each(|(_, _, val)| {
            *val = rhs * val;
        });
        res
    }
}

impl<P> MulAssign<&P> for HashSparse<P>
where
    P: Mul + Clone + Zero,
    for<'r> &'r P: Mul<&'r P, Output = P>,
{
    fn mul_assign(&mut self, rhs: &P) {
        self.iter_coords_mut().for_each(|(_, _, val)| {
            *val = rhs * val;
        });
    }
}

impl<P> Div<&P> for HashSparse<P>
where
    P: Div<Output = P> + Clone + Zero,
{
    type Output = Self;

    fn div(self, rhs: &P) -> Self {
        let mut res = self.clone();
        res.iter_coords_mut().for_each(|(_, _, val)| {
            *val = val.clone() / rhs.clone();
        });
        res
    }
}

impl<P> DivAssign<&P> for HashSparse<P>
where
    P: Div<Output = P> + Clone + Zero,
{
    fn div_assign(&mut self, rhs: &P) {
        self.iter_coords_mut().for_each(|(_, _, val)| {
            *val = val.clone() / rhs.clone();
        });
    }
}
