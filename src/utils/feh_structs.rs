use super::stats_structs::vec_f::VecF;
use std::{cmp, collections::{BinaryHeap, HashMap}};


// Wrapper function for comparing distances between units
struct UnitDistance<'a>(&'a String, f32);

impl<'a> std::cmp::Eq for UnitDistance<'a> {}

const EPSILON: f32 = 0.0001f32;
impl<'a> PartialEq for UnitDistance<'a> {
    fn eq(&self, other: &Self) -> bool {
        return self.1 - other.1 <= EPSILON
    }
}

impl<'a> PartialOrd for UnitDistance<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        return self.1.partial_cmp(&other.1);
    }
}

impl<'a> Ord for UnitDistance<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        return self.1.total_cmp(&other.1).reverse();
    }
}

// Boils a feh unit down to their name and stats
#[derive(Debug)]
pub struct FehUnit {
  pub name: String,
  pub stats: VecF // size 5
}

pub struct FehCurrent<'a> {
  pub current_unit: Option<&'a FehUnit>,
  pub current_stats: Option<VecF>,
  pub dragonflowers: usize,
  pub merges: usize,
  pub stat_priority_list: Option<[u8; 5]>,
  pub merge_idx: (usize, usize),
  pub df_idx: usize
}

// Manager struct for the current FehUnit 
impl<'a> FehCurrent<'a> {
  // Return a new FehCurrent struct
  pub fn new() -> Self {
    return FehCurrent{ 
      current_unit: None, 
      current_stats: None, 
      dragonflowers: 0, 
      merges: 0, 
      stat_priority_list: None,
      merge_idx: (0,1),
      df_idx: 0
    };
  }

  // Add num_merges # of merges to FehCurrent, updates self.current_stats
  pub fn add_merges(&mut self, num_merges: usize) -> &mut Self {
    let stat_vec: &mut VecF = self.current_stats.as_mut().unwrap();
    let mut merges_at: usize = 0;
    
    while merges_at < num_merges {
      stat_vec.set(self.merge_idx.0, stat_vec.get(self.merge_idx.0) + 1f32);
      stat_vec.set(self.merge_idx.1, stat_vec.get(self.merge_idx.1) + 1f32);
      self.merge_idx.0 = (self.merge_idx.0 + 2) % 5;
      self.merge_idx.1 = (self.merge_idx.1 + 2) % 5;
      merges_at += 1;
    }

    self.merges += num_merges;
    return self;
  }

  // Add num_dragonflowers # of dragonflowers to FehCurrent, updates self.current_stats
  pub fn add_dragonflowers(&mut self, num_dragonflowers: usize) -> &mut Self {
    let stat_vec: &mut VecF = self.current_stats.as_mut().unwrap();
    let mut dfs_at: usize = 0;

    while dfs_at < num_dragonflowers {
      stat_vec.set(self.df_idx, stat_vec.get(self.df_idx) + 1f32);
      self.df_idx = (self.df_idx + 1) % 5;
      dfs_at += 1;
    }

    self.dragonflowers += num_dragonflowers;
    return self;
  }

  // Bind a unit to the FehCurrent struct
  pub fn set_unit(&mut self, hero: &'a FehUnit) -> &mut Self {
    let stat_vec: &VecF = &hero.stats;
    self.current_unit = Some(hero);
    self.current_stats = Some(VecF::dupe(stat_vec));
    self.stat_priority_list = Some(create_priority_stat_list(stat_vec));
    self.reset_unit();

    return self;
  }

  // Reset the dragon flowers & merges on the unit
  pub fn reset_unit(&mut self) -> &mut Self {
    self.current_stats = Some(VecF::dupe(&self.current_unit.unwrap().stats));
    self.dragonflowers = 0;
    self.merges = 0;
    self.merge_idx = (0,1);
    self.df_idx = 0;

    return self;
  }
}

// Enum for tracking different distance metrics
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
  EUCLIDEAN = 0,
  COSINE = 1
}

impl DistanceMetric {
  const METRICS: [DistanceMetric; 2] = [Self::EUCLIDEAN, Self::COSINE];

  // Return the cardinality of the set of all distance metric enumerations
  pub fn cardinality() -> usize {
    return Self::METRICS.len();
  }

  // Return a distance metric by index
  pub fn from_index(index: usize) -> DistanceMetric {
    return Self::METRICS[index];
  }

  // Iterator for every element in METRICS
  pub fn iter() -> core::slice::Iter<'static, DistanceMetric> {
    return Self::METRICS.iter();
  }

  pub fn into_iter() -> core::array::IntoIter<DistanceMetric, 2> {
    return Self::METRICS.into_iter();
  }

  // Returns the name of the distance metric being used
  pub fn to_string(&self) -> String {
    match self {
      DistanceMetric::EUCLIDEAN => String::from("Euclidean"),
      DistanceMetric::COSINE => String::from("Cosine")
    }
  }

  // Return a distance function from a distance metric
  fn get_distance_func(&self) -> impl Fn(&VecF, &VecF, Option<f32>) -> f32 {
    match self {
      DistanceMetric::EUCLIDEAN => |v1: &VecF, v2: &VecF, _f: Option<f32>| v1.euclid_distance(v2),
      DistanceMetric::COSINE =>  |v1: &VecF, v2: &VecF, f: Option<f32>| (v1.dot(v2) / (f.unwrap() * v2.magnitude())).acos()
    }
  }

  // Returns pre_computation parameters to be passed into the distance function
  fn pre_computations(&self, v: &VecF) -> Option<f32> {
    match self {
      DistanceMetric::EUCLIDEAN => None,
      DistanceMetric::COSINE => Some(v.magnitude())
    }
  }
}

// Construct a list of unit proximity based on the passed in DistanceMetric
pub fn k_nearest_units<'a>(all_units: &'a HashMap<String,FehUnit>, cur_unit_name: &String, cur_unit_stats: &VecF, metric: DistanceMetric) -> Vec<&'a String> {
  let mut unit_heap: BinaryHeap<UnitDistance> = BinaryHeap::new();
  let pre_compute: Option<f32> =  metric.pre_computations(cur_unit_stats);
  let distance = metric.get_distance_func();

  for (unit_name, unit_struct) in all_units.iter() {
    if unit_name != cur_unit_name {
      let dist: f32 = distance(cur_unit_stats, &unit_struct.stats, pre_compute);
      unit_heap.push(UnitDistance(unit_name, dist));
    }
  }

  return unit_heap
    .into_sorted_vec()
    .into_iter()
    .map(|hero| hero.0)
    .collect::<Vec<&String>>();
}

// Print the k closest vectors from the sorted list
pub fn print_k_closest(all_units: &HashMap<String, FehUnit>, unit: &&str, unit_stats: &VecF, list: &Vec<&String>, k: usize, metric: DistanceMetric) -> () {
  println!("------------------------------------------\nThe top {} [{}] closest units to {} were...\n------------------------------------------", k, metric.to_string(), unit);
  for index in 0..k {
      let close_hero = list[index];
      println!("{}) {}, Diffs = {}", index + 1, close_hero, vector_to_string_diffs(&(&all_units.get(close_hero).unwrap().stats - unit_stats)));
  }
  println!("------------------------------------------\n")
}

// Print the k farthest vectors from the sorted list
pub fn print_k_farthest(all_units: &HashMap<String, FehUnit>, unit: &&str, unit_stats: &VecF, list: &Vec<&String>, k: usize, metric: DistanceMetric) -> () {
  println!("------------------------------------------\nThe top {} [{}] farthest units to {} were...\n------------------------------------------", k, metric.to_string(), unit);
  for index in ((list.len() - k)..(list.len())).rev() {
      let close_hero = list[index];
      println!("{}) {}, Diffs = {}", list.len() - index, close_hero, vector_to_string_diffs(&(&all_units.get(close_hero).unwrap().stats - unit_stats)));
  }
  println!("------------------------------------------\n")
}
  
// Return the center of mass of the Hashmap
pub fn get_center_of_mass(all_units: &HashMap<String, FehUnit>) -> VecF {
  let mut com = VecF::zeroes(5);
  for (_ , unit) in all_units.iter() {
      com = &com + &unit.stats;
  }

  return (1f32 / (all_units.len() as f32)) * (com);
}


// Return a list of indices that map to the VecF values in sorted decreasing order
fn create_priority_stat_list(stats_in: &VecF) -> [u8; 5] {
  let mut ordered_list: [(usize, u8); 5] = [
      (0, stats_in.get(0) as u8), 
      (1, stats_in.get(1) as u8), 
      (2, stats_in.get(2) as u8), 
      (3, stats_in.get(3) as u8), 
      (4, stats_in.get(4) as u8)
      ];
      
      // Insertion Sort
      let mut i: usize = 1;
      while i < 5 {
          let mut j: usize = i;
          while j > 0 && ordered_list[j - 1].1 < ordered_list[j].1 {
    ordered_list.swap(j-1, j);
    j -= 1;
  }
  i+= 1;
}

return ordered_list.map(|(idx, _)| idx as u8);
}

// Return the sign of the integer as a character
fn sign_to_char(i: i32) -> char {
  return if i < 0 { '-' }
    else if i > 0 { '+' }
    else { '\0' }
}

// Turn the integer to a string, sign included
fn float_stat_delta_string(i: i32) -> String {
  return String::from(sign_to_char(i)) + i.abs().to_string().as_str();
}

// Print the vector with float_stat_delta_string() signs
fn vector_to_string_diffs(stat_vec: &VecF) -> String {
  return format!("[Hp: {} Atk: {} Spd: {} Def: {}, Res: {}]", 
    float_stat_delta_string(stat_vec.get(0) as i32), 
    float_stat_delta_string(stat_vec.get(1) as i32), 
    float_stat_delta_string(stat_vec.get(2) as i32),
    float_stat_delta_string(stat_vec.get(3) as i32),
    float_stat_delta_string(stat_vec.get(4) as i32)
  );
}

