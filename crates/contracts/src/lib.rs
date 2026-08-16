//! Portable numerical contracts owned by Solverang.
//!
//! This crate deliberately contains no algorithms and no Sinbad types. Scientific compilers
//! and simulators may implement these traits; Solverang consumes them without a reverse
//! dependency on any physics repository.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ctx { pub deterministic: bool }
impl Ctx {
    #[must_use]
    pub fn real_os_default() -> Self { Self { deterministic: true } }
}

pub trait Scalar: Copy + Send + Sync + 'static + PartialEq + core::ops::Add<Output=Self> + core::ops::Sub<Output=Self> + core::ops::Mul<Output=Self> + core::ops::Neg<Output=Self> + core::ops::AddAssign + core::ops::SubAssign {
    fn zero() -> Self;
    fn one() -> Self;
}
impl Scalar for f64 { fn zero()->Self{0.0} fn one()->Self{1.0} }

pub trait SparseIndex: Copy + Ord + core::hash::Hash + Send + Sync + 'static {
    fn from_usize(value:usize)->Self;
    fn to_usize(self)->usize;
}
impl SparseIndex for u32 { fn from_usize(value:usize)->Self{value as u32} fn to_usize(self)->usize{self as usize} }
impl SparseIndex for u64 { fn from_usize(value:usize)->Self{value as u64} fn to_usize(self)->usize{self as usize} }
impl SparseIndex for usize { fn from_usize(value:usize)->Self{value} fn to_usize(self)->usize{self} }

#[derive(Clone,Copy,Debug,PartialEq,Eq,Serialize,Deserialize)]pub enum Orientation{Csr,Csc}
#[derive(Clone,Copy,Debug,PartialEq,Eq,Serialize,Deserialize)]pub enum Symmetry{General,Symmetric}

#[derive(Clone,Debug,PartialEq,Serialize,Deserialize)]
pub struct SparsePattern<I:SparseIndex=u32>{pub orientation:Orientation,pub nrows:usize,pub ncols:usize,pub offsets:Vec<I>,pub indices:Vec<I>,pub symmetry:Symmetry}
#[derive(Clone,Debug,PartialEq,Serialize,Deserialize)]
pub struct SparseMatrix<T,I:SparseIndex=u32>{pub pattern:Arc<SparsePattern<I>>,pub values:Vec<T>}
impl<T,I:SparseIndex> SparseMatrix<T,I>{pub fn nrows(&self)->usize{self.pattern.nrows}pub fn ncols(&self)->usize{self.pattern.ncols}pub fn nnz(&self)->usize{self.values.len()}}

#[derive(Clone,Debug,Default,Serialize,Deserialize)]
pub struct CooMatrix<T,I:SparseIndex=u32>{pub nrows:usize,pub ncols:usize,pub rows:Vec<I>,pub cols:Vec<I>,pub values:Vec<T>}
impl<T:Scalar,I:SparseIndex>CooMatrix<T,I>{
    #[must_use]pub fn new(nrows:usize,ncols:usize)->Self{Self{nrows,ncols,rows:Vec::new(),cols:Vec::new(),values:Vec::new()}}
    pub fn push(&mut self,row:usize,col:usize,value:T){self.rows.push(I::from_usize(row));self.cols.push(I::from_usize(col));self.values.push(value);}
    #[must_use]pub fn finish_csr(self)->SparseMatrix<T,I>{self.compress(Orientation::Csr)}
    #[must_use]pub fn compress(self,orientation:Orientation)->SparseMatrix<T,I>{
        let major=match orientation{Orientation::Csr=>self.nrows,Orientation::Csc=>self.ncols};let mut order:Vec<usize>=(0..self.values.len()).collect();
        let key=|k:usize|match orientation{Orientation::Csr=>(self.rows[k].to_usize(),self.cols[k].to_usize()),Orientation::Csc=>(self.cols[k].to_usize(),self.rows[k].to_usize())};order.sort_by_key(|&k|key(k));
        let mut offsets=Vec::with_capacity(major+1);let mut indices=Vec::new();let mut values=Vec::new();offsets.push(I::from_usize(0));let mut cur=0usize;let mut p=0usize;
        while p<order.len(){let (maj,min)=key(order[p]);while cur<maj{offsets.push(I::from_usize(indices.len()));cur+=1;}let mut value=self.values[order[p]];let mut q=p+1;while q<order.len()&&key(order[q])==(maj,min){value+=self.values[order[q]];q+=1;}indices.push(I::from_usize(min));values.push(value);p=q;}
        while cur<major{offsets.push(I::from_usize(indices.len()));cur+=1;}
        SparseMatrix{pattern:Arc::new(SparsePattern{orientation,nrows:self.nrows,ncols:self.ncols,offsets,indices,symmetry:Symmetry::General}),values}
    }
}

#[derive(Clone,Debug,Error,PartialEq)]pub enum NumericError{
    #[error("dimension mismatch: expected {expected}, got {got}")]DimensionMismatch{expected:usize,got:usize},
    #[error("operation unsupported: {what}")]Unsupported{what:String},
    #[error("transpose action is unsupported")]UnsupportedTranspose,
    #[error("numeric failure: {0}")]Other(String),
}

pub trait Jacobian<T>{
    fn n(&self)->usize;
    fn residual(&self,ctx:&Ctx,x:&[T],out:&mut[T])->Result<(),NumericError>;
    fn assemble_into(&self,ctx:&Ctx,x:&[T],out:&mut SparseMatrix<T>)->Result<(),NumericError>;
    fn jvp(&self,ctx:&Ctx,x:&[T],v:&[T],out:&mut[T])->Result<(),NumericError>{let _=(ctx,x,v,out);Err(NumericError::Unsupported{what:"matrix-free jvp".into()})}
}

#[derive(Clone,Copy,Debug,PartialEq)]pub struct IntegratorCoeffs<S>{pub mass:S,pub damp:S,pub stiff:S}
impl<S:Scalar>IntegratorCoeffs<S>{#[must_use]pub fn bdf(mass:S)->Self{Self{mass,damp:S::zero(),stiff:S::one()}}#[must_use]pub fn generalized_alpha(mass:S,damp:S,stiff:S)->Self{Self{mass,damp,stiff}}}

#[derive(Clone,Copy,Debug,PartialEq,Eq,Hash,Serialize,Deserialize)]#[non_exhaustive]pub enum DaeIndex{Ode,Index1,Index2,Index3,HigherReduced}

pub trait DaeResidual<S:Scalar>:Jacobian<S>{
    fn residual_at(&self,ctx:&Ctx,t:f64,x:&[S],out:&mut[S])->Result<(),NumericError>;
    fn charge(&self,ctx:&Ctx,t:f64,x:&[S],out:&mut[S])->Result<(),NumericError>;
    fn mass_apply(&self,ctx:&Ctx,t:f64,x:&[S],v:&[S],out:&mut[S])->Result<(),NumericError>;
    fn iteration_matrix(&self,ctx:&Ctx,t:f64,x:&[S],coeffs:&IntegratorCoeffs<S>,out:&mut SparseMatrix<S>)->Result<(),NumericError>{let _=(ctx,t,x,coeffs,out);Err(NumericError::Unsupported{what:"assembled DAE iteration_matrix".into()})}
    fn dae_index_hint(&self)->DaeIndex;
}

/// Target-neutral executable residual seam. Anvil, handwritten kernels, interpreters and
/// future GPU backends all satisfy the same contract; Solverang does not need to know which
/// compiler produced the executable.
pub trait ExecutableResidual:Send+Sync{
    fn dimension(&self)->usize;
    fn residual(&self,x:&[f64],out:&mut[f64])->Result<(),NumericError>;
    fn jvp(&self,x:&[f64],v:&[f64],out:&mut[f64])->Result<(),NumericError>;
    fn vjp(&self,x:&[f64],v:&[f64],out:&mut[f64])->Result<(),NumericError>;
}

#[cfg(test)]mod tests{use super::*;#[test]fn coo_compression_sums_duplicates_deterministically(){let mut c=CooMatrix::<f64>::new(2,2);c.push(0,1,2.0);c.push(0,1,3.0);let m=c.finish_csr();assert_eq!(m.values,vec![5.0]);assert_eq!(m.pattern.offsets,vec![0,1,1]);}}
