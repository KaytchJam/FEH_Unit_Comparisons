use core::ops::{Add, Sub, Mul};

#[derive(Debug)]
pub struct VecF {
  data: Vec<f32>
}

// linear algebra vector of variable size
impl VecF {
  // construct a vecf constaining size 0s
  pub fn zeroes(size: usize) -> Self {
    return Self{data: vec![0f32; size]};
  }

  // construct a vecf containing size 1s
  pub fn ones(size: usize) -> Self {
    return Self{data: vec![1f32; size]};
  }

  // construct a vecf containing size Ns
  pub fn ns(n: f32, size: usize) -> Self {
    return Self{data: vec![n; size]};
  }

  // construct a vecf from a vector
  pub fn from(data: Vec<f32>) -> Self {
    return Self{data};
  }
 
  pub fn dupe(other: &VecF) -> Self {
    return VecF { data: other.data.clone() };
  }

  pub fn get(&self, index: usize) -> f32 {
    return self.data[index];
  }

  pub fn set(&mut self, index: usize, val: f32) -> () {
    self.data[index] = val;
  }

  // euclidean distance between two vecfs
  pub fn euclid_distance(&self, other: &Self) -> f32 {
    assert_eq!(self.data.len(), other.data.len());
    return (0..self.data.len())
      .into_iter()
      .map(|index: usize| (self.data[index] - other.data[index]).powi(2))
      .sum::<f32>()
      .sqrt();
  }

  // dot product of two vecfs
  pub fn dot(&self, other: &Self) -> f32 {
    assert_eq!(self.data.len(), other.data.len());
    return (0..self.data.len())
      .into_iter()
      .map(|index: usize| (self.data[index] * other.data[index]))
      .sum::<f32>();
  }

  // magnitude of a vecf
  pub fn magnitude(&self) -> f32 {
    return (self.data
      .iter()
      .map(|f| *f * *f)
      .sum::<f32>()
    ).sqrt();
  }

  // return the vecf size
  pub fn size(&self) -> usize {
    return self.data.len();
  }

   // divide the vecf by a scalar int
   pub fn divi(&self, scalar: i32) -> Self {
    return VecF{ data: self.data
      .iter()
      .map(|f: &f32| *f / scalar as f32)
      .collect::<Vec<f32>>()
    };
  }

  // divide the vecf by a scalar float
  pub fn divf(&self, scalar: f32) -> Self {
    return VecF{ data: self.data
      .iter()
      .map(|f: &f32| *f / scalar)
      .collect::<Vec<f32>>()
    };
  }

  // normalizes a vector to the 0 to 1 range
  pub fn normalize(&self) -> Self {
    let mag: f32 = self.magnitude();
    return self.divf(mag);
  }

  // Sum all elements in the VecF
  pub fn sum(&self) -> f32 {
    return self.data.iter().sum();
  }

  // Project this vector onto another vector "onto"
  pub fn project(&self, onto: &VecF) -> VecF {
    assert_eq!(self.data.len(), onto.data.len());
    return onto * (self.dot(onto) / onto.dot(onto));
  }

  // Consume the vector and return the Vec<f32> it holds
  pub fn retrieve_buffer(self) -> Vec<f32> {
    return self.data;
  }

  // 
  pub fn angle(&self, other: &VecF) -> f32 {
    return self.dot(other) / (self.magnitude() * other.magnitude()).acos();
  }
}

// VecF Trait Implementations!
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

impl Add for VecF {
  type Output = Self;
  fn add(mut self, other: Self) -> Self {
    assert_eq!(self.data.len(), other.data.len());

    let length: usize = self.data.len();
    for index in (0..length).into_iter() {
      self.data[index] += other.data[index];
    }
    return self;
  }
}

impl Add for &VecF {
  type Output = VecF;
  fn add(self, other: Self) -> VecF {
    assert_eq!(self.data.len(), other.data.len());

    return VecF::from(
      (0..self.data.len())
        .into_iter()
        .map(|i| self.data[i] + other.data[i])
        .collect::<Vec<f32>>()
    );
  }
}

impl Sub for VecF {
  type Output = Self;
  fn sub(mut self, other: Self) -> Self {
    assert_eq!(self.data.len(), other.data.len());

    let length: usize = self.data.len();
    for index in (0..length).into_iter() {
      self.data[index] -= other.data[index];
    }
    return self;
  }
}

impl Sub for &VecF {
  type Output = VecF;
  fn sub(self, other: Self) -> VecF {
    assert_eq!(self.data.len(), other.data.len());

    return VecF::from(
      (0..self.data.len())
        .into_iter()
        .map(|i| self.data[i] - other.data[i])
        .collect::<Vec<f32>>()
    );
  }
}

impl Mul<f32> for VecF {
  type Output = Self;
  fn mul(mut self, scalar: f32) -> Self {
    for value in self.data.iter_mut() {
      *value *= scalar;
    }
    return self;
  }
}

impl Mul<f32> for &VecF {
  type Output = VecF;
  fn mul(self, scalar: f32) -> VecF {
    return VecF { data: self.data.iter().map(|f| f * scalar).collect::<Vec<f32>>() };
  }
}

impl Mul<i32> for VecF {
  type Output = Self;
  fn mul(mut self, scalar: i32) -> Self {
    for value in self.data.iter_mut() {
      *value *= scalar as f32;
    }
    return self;
  }
}

impl Mul<VecF> for f32 {
  type Output = VecF;
  fn mul(self, mut vector: VecF) -> VecF {
    for value in vector.data.iter_mut() {
      *value *= self;
    }
    return vector;
  }
}

// pub struct MatF {
//   columns: Vec<VecF>
// }

// impl MatF {
//   // Returns a tuple of (rows, columns) of the MatF
//   fn dimension(&self) -> (usize, usize) {
//     return (self.columns[0].data.len(), self.columns.len());
//   }
// }

// MatF Trait Implementations
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

// impl Add for MatF {
//   type Output = Self;

//   fn add(self, other: Self) -> Self {

//   }
// }