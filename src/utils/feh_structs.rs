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

// Return a sorted vector of the nearest heroes to the current unit
pub fn k_nearest_units<'a>(all_units: &'a HashMap<String, FehUnit>, cur_unit_name: &String, cur_unit_stats: &VecF) -> Vec<&'a String> {
  let mut unit_heap: BinaryHeap<UnitDistance> = BinaryHeap::new();

  for (unit_name, unit_stats_vec) in all_units.iter() {
    if unit_name != cur_unit_name {
      let dist: f32 = cur_unit_stats.euclid_distance(&unit_stats_vec.stats);
      unit_heap.push(UnitDistance(unit_name, dist));
    }
  }

  return unit_heap
    .into_sorted_vec()
    .into_iter()
    .map(|hero| hero.0)
  . collect::<Vec<&String>>();
}
  
// Return the center of mass of the Hashmap
pub fn get_center_of_mass(all_units: &HashMap<String, FehUnit>) -> VecF {
    let mut com = VecF::zeroes(5);
    for (_ , unit) in all_units.iter() {
        com = &com + &unit.stats;
    }
  
    return (1f32 / (all_units.len() as f32)) * (com);
  }

// Manager struct for the current FehUnit 

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
