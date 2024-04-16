use std::ops::BitAnd;
use std::process::Output;
use std::vec;
use rand::seq::index;
use rand::{seq::SliceRandom, Rng};
use std::collections::HashMap;
use std::hash::Hash;
use std::fmt;

struct SampleStream {
  buffer: Vec<i32>,
  first: usize,
  last: usize,
  full: bool,
  mean: f32,
}

impl SampleStream {
  // create a new SampleStream
  fn new(capacity: usize) -> Self {
    return SampleStream {
        buffer: vec![0; capacity],
        first: 0,
        last: 0,
        full: false,
        mean: 0.0f32,
    };
  }

  // insert an item into the SampleStream
  fn insert(&mut self, val: i32) -> &mut Self {
    let prev_cap = self.size() - self.capacity();
    let prev_val = self.buffer[self.last];
  
    self.buffer[self.last] = val;
    self.last += 1;
    self.last = self.last * !(self.last >= self.buffer.len()) as usize;
    self.first += self.full as usize;
    self.first = self.first * !(self.first >= self.buffer.len()) as usize;
    self.full = self.full || self.last == self.first;
    self.mean = (self.mean * (prev_cap as f32) + (val - prev_val) as f32) / (self.size() - self.capacity()) as f32;
    return self;
  }

  // get the size of the SampleStream
  fn size(&self) -> usize {
    return self.buffer.len();
  }

  // get the capacity of the SampleStream
  fn capacity(&self) -> usize {
    return (!self.full as usize) * (self.buffer.len() - self.last);
  }

  // print the entire SampleStream container
  fn print_buffer(&self) -> &Self {
    print!("{:?}\n", self.buffer);
    return self;
  }

  // return the last element in the SampleStream
  fn peek(&self) -> i32 {
    let is_first: bool = self.last == 0;
    let index: usize = (((self.last as isize - 1) * !is_first as isize)
        + is_first as isize * (self.buffer.len() as isize - 1))
        as usize;
    return self.buffer[index];
  }

  // return the first element in the SampleStream
  fn front(&self) -> i32 {
    return self.buffer[self.first];
  }

  // returns whether the SampleStream has non-destructive inserts left
  // and the number available.
  fn is_full(&self) -> bool {
    return self.full;
  }

  // returns the mean
  fn mean(&self) -> f32 {
    return self.mean;
  }

  // Iterator for the SampleStream type
  fn iter(&self) -> Iter<'_> {
    return Iter { stream: self, offset: 0 };
  }
}
 
struct Iter<'a> {
    stream: &'a SampleStream,
    offset: usize
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a i32;
    fn next(&mut self) -> Option<&'a i32> {
        let rval: usize = (self.stream.first + self.offset) % self.stream.size();
        let cond: usize = usize::from(self.offset < self.stream.size() - self.stream.capacity());
        let choices: [Option<&i32>; 2] = [None, Some(&self.stream.buffer[rval])];

        self.offset += cond;
        return choices[cond];
    }
}

fn print_and_get_samples<T: std::fmt::Debug>(data: &Vec<T>, num_samples: usize) -> Vec<&T> {
    let sample: Vec<_> = data
        .choose_multiple(&mut rand::thread_rng(), num_samples)
        .collect();
    
    std::println!("{:?}", sample);
    return sample;
}

fn example_one() {
    let list: Vec<(&str, bool)> = vec![("John", true), ("Susan", true), ("Yemi", false), ("Chaka", false), ("Yu", false)];
    let samples: usize = 3;
    print_and_get_samples(&list, samples);
}

fn example_two() {
    let list = vec![("MANUFACTURING", 1000), ("NEWTORKS",250), ("END USER",550), ("DATA CENTRE",200)];
    let mut sum: i32 = 0;
    for val in list.iter() {
        sum += val.1;
    }

    for val in list.iter() {
        print!("Field: '{}' : Proportion: {}\n", val.0, val.1 as f32 / sum as f32);
    }
}

fn normalize_to_char(num: i32, local_max: i32, local_min: i32) -> char {
    let num: f64 = num.abs() as f64;
    const UPPER_CHAR: f64 = 'z' as i64 as f64;
    const LOWER_CHAR: f64 = 'a' as i64 as f64;
    let local_max: f64 = local_max as f64;
    let local_min: f64 = local_min as f64;


    let calc = (num - local_min) / (local_max - local_min) * (UPPER_CHAR - LOWER_CHAR) + LOWER_CHAR;

    print!("Calc: {}\n", calc);

    return char::from(calc as u8);
}

fn example_sample_stream_and_frequency_dist(stream_capacity : usize) {
    let vals = [1500, 244, 400, 64, 1008, 5000, 701];
    let mut stream = SampleStream::new(stream_capacity);
    for v in vals.into_iter() {
        print!("Inserted: {}, Mean: {}\n", stream.insert(v).peek(), stream.mean());
    }

    let (mut stream_min, mut stream_max) : (i32, i32) = (i32::MAX, i32::MIN);
    for value in stream.iter() {
      stream_min = if *value < stream_min { *value } else { stream_min };
      stream_max = if *value > stream_max { *value } else { stream_max };
    }

    //let stream_max = *stream.iter().max().unwrap();
    //let stream_min = *stream.iter().min().unwrap();

    for v in stream.iter().enumerate() {
        print!("index: {} - value: {}\n", v.0, v.1);
    }

    print!("Collection: {:?}, Full: {}, First: {}, Last: {}, Front: {}, Peek: {}, Final mean: {}\n", 
        stream.iter().map(|v: &i32| -> char { return normalize_to_char(*v, stream_max, stream_min); }).collect::<Vec<char>>(),
        stream.print_buffer().is_full(),
        stream.first,
        stream.last,
        stream.front(), 
        stream.peek(), 
        stream.mean()
    );
}


// fn example_residual_plot() {
//   let x_data = vec![58f32, 59f32, 60f32, 61f32, 62f32, 63f32, 64f32, 65f32];
//   let y_data = vec![113f32, 115f32, 118f32, 121f32, 124f32, 128f32, 131f32, 134f32];
// }

// PROBABILITY
// THE SET OF POSSIBLE OUTCOMES OF A CHANCE EXPERIMENT IS KNOWN AS THE SAMPLE SPACE

#[derive(Debug)]
enum Gender {
  MALE,
  FEMALE,
  NB
}

#[derive(Debug)]
enum Genres {
  CLASSICAL,
  ROCK,
  COUNTRY,
  OTHER
}

const GENDER_LIST: &[Gender; 3] = &[Gender::MALE, Gender::FEMALE, Gender::NB];
const GENRE_LIST: &[Genres; 4] = &[Genres::CLASSICAL, Genres::ROCK, Genres::COUNTRY, Genres::OTHER];

// AN EVENT - ANY COLLECTION OF OUTCOMES FROM THE SAMPLE SPACE OF A CHANCE EXPERIMENT
// SIMPLE EVENT - EVENT CONSISTING OF EXACTLY 1 OUTCOME

pub mod stats_functions {
  use crate::stats_problems::{Hash, HashMap};
  use rand::Rng;

  pub fn get_simple_random_sample(pop_list: &[i32; 12], observations: usize, rng : &mut rand::rngs::ThreadRng) -> Vec<i32> {
    let mut sample: Vec<i32> = vec![0; 12];
  
    let mut i = 0;
    while i < observations {
      let index: usize = rng.gen_range(0..pop_list.len());
      let valid: i32 = i32::from(sample[index] < pop_list[index]);
      sample[index] += 1 & (valid as i32);
      i += 1 & (valid as usize);
    }
  
    return sample;
  }
  
  pub fn vec_expectation(dist: &Vec<f32>) -> f32 {
      return dist.iter().enumerate().map(|(index, prob) : (usize, &f32)| (index + 1) as f32 * (*prob)).sum();
  }

  pub fn create_range_distribution_exclusive(start: usize, bin_range: usize, num_bins: usize) -> Vec<(std::ops::Range<i32>,i32)> {
    let mut storage: Vec<(std::ops::Range<i32>, i32)> = vec![(0..0, 0); num_bins];
  
    for (index, tup) in storage.iter_mut().enumerate() {
      let begin: i32 = start as i32 + index as i32 * bin_range as i32;
      let end: i32 = begin + bin_range as i32;
      tup.0 = begin..end;
    }
  
    return storage;
  }

  // TRIMMED MEAN
  pub fn trimmed_mean(data: &[i32], deletions: usize) -> Option<f32> {
    let mut ret: Option<f32> = None;
    let mut start: usize = deletions;
    let end: usize = data.len() - deletions;
  
    while start < end {
      ret = Some(ret.unwrap_or(0f32) + data[start] as f32);
      start += 1;
    }
    
    return if ret.is_some() { Some(ret.unwrap() / ((end - deletions) as f32)) } else { None };
  }
  
  // TRIMMED MEAN WITH A GIVEN PERENTAGE
  pub fn trimmed_mean_percent(data: &[i32], trimmed_percent: f32) -> Option<f32> {
    let deletions = (trimmed_percent * data.len() as f32) as usize;
    return trimmed_mean(data, deletions);
  }
  
  pub fn variance(data: &[i32], mean: f32) -> f32 {
    let mut deviation_sums: f32 = 0f32;
    for item in data.iter() {
      deviation_sums += f32::powi(*item as f32 - mean, 2);
    }
    return deviation_sums / (data.len() - 1) as f32;
  }

  const EPSILON: f32 = 0.01;
  pub fn chebyshev_rule(k: f32) -> f32 {
    return if k.abs() - 1f32 >= EPSILON {100f32 * (1f32 - 1f32 / k.powi(2))} else {0f32};
  }

  pub fn z_score(val: f32, mean: f32, stdev: f32) -> f32 {
    return (val - mean) / stdev;
  }

  pub fn calc_pcc_int(sample_x: &Vec<i32>, sample_y: &Vec<i32>) -> Option<f32> {
    if sample_x.len() != sample_y.len() { return None }
    let length = sample_x.len();
    let (mut x_mean, mut y_mean): (f32, f32) = (0f32, 0f32);
    let (mut x_stdev, mut y_stdev): (f32, f32) = (0f32, 0f32);
  
    for i in 0..length {
      x_mean += sample_x[i] as f32;
      y_mean += sample_y[i] as f32;
    }
  
    x_mean /= length as f32;
    y_mean /= length as f32;
  
    for i in 0..length {
      x_stdev += (sample_x[i] as f32 - x_mean).powi(2);
      y_stdev += (sample_y[i] as f32 - y_mean).powi(2);
    }
  
    x_stdev = (x_stdev / (length - 1) as f32).sqrt();
    y_stdev = (y_stdev / (length - 1) as f32).sqrt();
  
    let mut pcc: f32 = 0f32;
    for i in 0..length {
      pcc += z_score(sample_x[i] as f32, x_mean, x_stdev) * z_score(sample_y[i] as f32, y_mean, y_stdev);
    }
  
    return Some(pcc / (length - 1) as f32);
  }
  
  pub fn calc_pcc_f32(sample_x: &Vec<f32>, sample_y: &Vec<f32>) -> Option<f32> {
    if sample_x.len() != sample_y.len() { return None }
    let length = sample_x.len();
    let (mut x_mean, mut y_mean): (f32, f32) = (0f32, 0f32);
    let (mut x_stdev, mut y_stdev): (f32, f32) = (0f32, 0f32);
  
    for i in 0..length {
      x_mean += sample_x[i];
      y_mean += sample_y[i];
    }
  
    x_mean /= length as f32;
    y_mean /= length as f32;
  
    for i in 0..length {
      x_stdev += (sample_x[i] - x_mean).powi(2);
      y_stdev += (sample_y[i] - y_mean).powi(2);
    }
    
    x_stdev = (x_stdev / (length - 1) as f32).sqrt();
    y_stdev = (y_stdev / (length - 1) as f32).sqrt();
  
    let mut pcc: f32 = 0f32;
    for i in 0..length {
      pcc += z_score(sample_x[i], x_mean, x_stdev) * z_score(sample_y[i], y_mean, y_stdev);
    }
  
    return Some(pcc / (length - 1) as f32);
  }
  
  pub fn calc_pearson_correlation_coef_int(bivariate_sample: &Vec<(i32, i32)>) -> f32 {
    let (mut x_mean, mut y_mean): (f32, f32) = (0f32, 0f32);
    let (mut x_stdev, mut y_stdev): (f32, f32) = (0f32, 0f32);
  
    // sum for means
    for (x, y) in bivariate_sample.iter() {
      x_mean += *x as f32;
      y_mean += *y as f32;
    }
  
    x_mean /= bivariate_sample.len() as f32;
    y_mean /= bivariate_sample.len() as f32;
  
    // sum difference with mean
    for (x, y) in bivariate_sample.iter() {
      x_stdev += (*x as f32 - x_mean).powi(2);
      y_stdev += (*y as f32 - y_mean).powi(2);
    }
    
    x_stdev = (x_stdev / (bivariate_sample.len() - 1) as f32).sqrt();
    y_stdev = (y_stdev / (bivariate_sample.len() - 1) as f32).sqrt();
  
    let mut pcc = 0f32;
    for (x, y) in bivariate_sample.iter() { pcc += z_score(*x as f32, x_mean, x_stdev) * z_score(*y as f32, y_mean, y_stdev); }
    pcc /= (bivariate_sample.len() - 1) as f32;
    
    print!("Pearson Corellation Coefficient: {}\n", pcc);
    return pcc;
  }

  pub fn calc_pearson_correlation_coef_float(bivariate_sample: &Vec<(f32, f32)>) -> f32 {
    let (mut x_mean, mut y_mean): (f32, f32) = (0f32, 0f32);
    let (mut x_stdev, mut y_stdev): (f32, f32) = (0f32, 0f32);
  
    // sum for means
    for (x, y) in bivariate_sample.iter() {
      x_mean += *x;
      y_mean += *y;
    }
  
    x_mean /= bivariate_sample.len() as f32;
    y_mean /= bivariate_sample.len() as f32;
    
    // sum difference with mean
    for (x, y) in bivariate_sample.iter() {
      x_stdev += (*x as f32 - x_mean).powi(2);
      y_stdev += (*y as f32 - y_mean).powi(2);
    }
    
    x_stdev = (x_stdev / (bivariate_sample.len() - 1) as f32).sqrt();
    y_stdev = (y_stdev / (bivariate_sample.len() - 1) as f32).sqrt();
  
    let pcc: f32 = bivariate_sample.iter().map(|(x,y)| -> f32 {
      z_score(*x, x_mean, x_stdev) * z_score(*y, y_mean, y_stdev)
    }).sum::<f32>() / (bivariate_sample.len() - 1) as f32;
  
    print!("Pearson Corellation Coefficient: {}\n", pcc);
    return pcc;
  }
  
  pub fn least_squares(x_sample: &Vec<f32>, y_sample: &Vec<f32>) -> Option<(f32, f32)> {
    if x_sample.len() != y_sample.len() { return None }
    let length = x_sample.len();
    let (mut x_mean, mut y_mean): (f32, f32) = (0f32, 0f32);
  
    for i in 0..length {
      x_mean += x_sample[i];
      y_mean += y_sample[i];
    }
  
    x_mean /= length as f32;
    y_mean /= length as f32;
  
    let mut numerator_sum = 0f32;
    let mut denominator_sum = 0f32;
    for i in 0..length {
      let x_diff = x_sample[i] - x_mean;
      numerator_sum += x_diff * (y_sample[i] - y_mean);
      denominator_sum += x_diff.powi(2);
    }
  
    print!("x mean: {}, y mean: {}\n", x_mean, y_mean);
  
    let slope: f32 = numerator_sum / denominator_sum;
    return Some((slope, y_mean - slope * x_mean));
  }
  
  pub fn calc_mean_simple(sample: &Vec<f32>) -> f32 {
    return sample.iter().sum::<f32>() / sample.len() as f32;
  }
  
  pub fn calc_residuals(x_sample: &Vec<f32>, y_sample: &Vec<f32>, (slope, y_intercept): (f32, f32)) -> Option<Vec<f32>> {
    if x_sample.len() != y_sample.len() { return None; }
    let length = x_sample.len();
    return Some((0..length).into_iter().map(|i| y_sample[i] - (slope * x_sample[i] + y_intercept)).collect::<Vec<f32>>());
  }
  
  pub fn coef_of_determination(ss_resid: f32, ss_total: f32) -> f32 {
    return 1f32 - ss_resid / ss_total;
  }
  
  // NOT
  pub fn not<T: Copy + Eq + PartialEq + Hash>(sample_space: &Vec<T>, event1: &Vec<T>) -> Vec<T> {
    let mut seen: HashMap<T, bool> = HashMap::new();
    for e in event1.into_iter() { if !seen.contains_key(e) { seen.insert(*e, true); }}
    return sample_space.into_iter().filter(|t: &&T| !seen.contains_key(t)).map(|t: &T| *t).collect::<Vec<T>>();
  }
  
  // UNION OF TWO EVENTS
  pub fn union<T: Copy + Eq + PartialEq + Hash>(event1: &Vec<T>, event2: &Vec<T>) -> Vec<T> {
    let mut seen: HashMap<T, bool> = HashMap::new();
    for i in event1.into_iter() { if !seen.contains_key(i) { seen.insert(*i, true); }}
    for e in event2.into_iter() { if !seen.contains_key(e) { seen.insert(*e, true); }}
    return seen.keys().map(|t: &T| *t).collect::<Vec<T>>()
  }
  
  pub fn collect_new<T: Copy + Eq + PartialEq + Hash>(seen: &mut HashMap<T, bool>, event: &Vec<T>) -> Vec<T> {
    for e in event.into_iter() { if !seen.contains_key(e) { seen.insert(*e, true); }}
    return seen.keys().map(|t: &T| *t).collect::<Vec<T>>()
  }
  
  // UNION OF N EVENTS
  pub fn arbitrary_union<T: Copy + Eq + PartialEq + Hash>(event_set: &[&Vec<T>]) -> Vec<T>{
    let mut seen: HashMap<T, bool> = HashMap::new();
    let mut out: Vec<T> = Vec::new();
  
    for index in 0..event_set.len() {
      out = collect_new(&mut seen, &event_set[index]);
    }
  
    return out;
  }
  
  // INTERSECTION OF TWO EVENTS
  pub fn intersect<T: Copy + Eq + PartialEq + Hash>(event1: &Vec<T>, event2: &Vec<T>) -> Vec<T> {
    let mut seen: HashMap<T, u32> = HashMap::new();
    for i in event1.into_iter().chain(event2.into_iter()) { 
      if !seen.contains_key(i) { seen.insert(*i,1u32); } 
      else { *seen.get_mut(i).unwrap() += 1u32; }
    }
    return seen.keys().filter(|k| seen[k] > 1).map(|k| *k).collect::<Vec<T>>();
  }
  
  // CHECKS IF TWO VECTORS ARE MUTUALLY EXCLUSIVE OR NOT
  pub fn disjoint<T: Copy + Eq + PartialEq + Hash>(event1: &Vec<T>, event2: &Vec<T>) -> bool {
    return intersect(event1, event2).len() == 0usize;
  }
  
  pub fn make_frequencies<T: Copy + Eq + PartialEq + Hash>(arr: &Vec<T>) -> HashMap<T,i32> {
    let mut freqs: HashMap<T, i32> = HashMap::new();
    for item in arr.iter() {
      let pt: Option<&mut i32> = freqs.get_mut(item);
      if pt.is_some() { (*pt.unwrap()) += 1; } else { freqs.insert(*item, 1); };
    }
    
    return freqs;
  }
  
  pub fn make_joint_frequencies<T: Copy + Eq + PartialEq + Hash>(arr1: &Vec<T>, arr2: &Vec<T>) -> HashMap<T,i32> {
    let mut freqs: HashMap<T, i32> = HashMap::new();
    for item in arr1.iter().chain(arr2.iter()) {
        let pt: Option<&mut i32> = freqs.get_mut(item);
        if pt.is_some() { (*pt.unwrap()) += 1; } else { freqs.insert(*item, 1); };
      }
      
    return freqs;
  }

  pub fn to_probability<T: Copy + Eq + PartialEq + Hash>(freqs: HashMap<T,i32>) -> Vec<f32> {
    let count: usize = freqs.iter().map(|i| *i.1).sum::<i32>() as usize;
    return freqs.iter().map(|i| *i.1 as f32 / count as f32).collect::<Vec<f32>>();
  }
  
  pub fn gen_event_prob_map<T: Copy + Eq + PartialEq + Hash>(sample_space: &Vec<T>) -> HashMap<T, f32> {
    let mut dist: HashMap<T, f32> = HashMap::new();
    let lengthf: f32 = sample_space.len() as f32;
    
    for t in sample_space.into_iter() { 
      if !dist.contains_key(t) { 
        dist.insert(*t, 1f32 / lengthf); 
      } else  {
        let t_ref: &mut f32 = dist.get_mut(t).unwrap();
        *t_ref = *t_ref + 1f32 / lengthf;
      }
    }
    return dist;
  }
  
  pub fn relative_frequency<T: Eq>(sample_space: &Vec<T>, item: &T) -> f32 {
    return sample_space.into_iter().filter(|t| **t == *item).count() as f32 / sample_space.len() as f32;
  }
  
  pub fn calc_entropy(set: &Vec<i32>) -> f32 {
    let freqs: HashMap<i32, i32> = make_frequencies(set);
    let count: f32 = set.len() as i32 as f32;
    let mut entropy: f32 = 0f32;
    
    for u in set.iter() {
      let prob: f32 = *freqs.get(u).unwrap() as f32 / count;
      entropy += prob * prob.log2();
    }
      
    return -1f32 * entropy;
  }

  pub fn calc_joint_entropy(set1: &Vec<i32>, set2: &Vec<i32>) -> f32 {
    let freqs: HashMap<i32, i32> = make_joint_frequencies(&set1, &set2);
      let count: f32 = (set1.len() + set2.len()) as i32 as f32;
      let mut entropy: f32 = 0f32;
      
      for u in freqs.iter() {
        let prob: f32 = *freqs.get(u.0).unwrap() as f32 / count;
        entropy += prob * prob.log2();
      }
      
      return -1f32 * entropy;
  }

  pub fn mutual_information(set1: &Vec<i32>, set2: &Vec<i32>) -> f32 {
      let e1 = calc_entropy(set1);
      let e2 = calc_entropy(set2);
      let je = calc_joint_entropy(set1, set2);
      return e1 + e2 - je;
    }
  }
  pub fn example_sampling_and_relative_freqs() {
    let pop_frequencies = [117, 157, 158, 115, 78, 44, 21, 7, 6, 1, 3, 1];
    let total = pop_frequencies.iter().sum::<i32>();
    print!("Total: {}\n", total);
    print!("Frequencies: {:?}\n", pop_frequencies);
  
    const SAMPLE_SIZE: usize = 200;
    let div_list_with_int = |i: i32, divisor: usize| return i as f32 / divisor as f32 * 100f32;
    let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
    let mut sample_list: [Option<Vec<f32>>; 4] = [None, None, None, None];
  
    for i in 0..4 {
      sample_list[i] = Some(stats_functions::get_simple_random_sample(&pop_frequencies, SAMPLE_SIZE, &mut rng)
        .into_iter()
        .map(|int: i32| div_list_with_int(int, SAMPLE_SIZE))
        .collect::<Vec<f32>>()
      );
    }
  
    let pop_rel_freqs = pop_frequencies.iter().map(|i: &i32| div_list_with_int(*i, total as usize)).collect::<Vec<f32>>();
    print!("Population: {:?}, expected value: {}\n", pop_rel_freqs, stats_functions::vec_expectation(&pop_rel_freqs) / 100f32);
    for (index, sample) in sample_list.iter().enumerate() {
      let sample_ref: &Vec<f32> = sample.as_ref().unwrap();
      print!("Sample {}: {:?}, expected value: {}\n", index + 1, sample_ref, stats_functions::vec_expectation(sample_ref) / 100f32);
    }
  }
  
fn example_five() {
  const T: &str = "Tracy";
  const R: &str = "Rishi";
  const J: &str = "Jacob";
  const K: &str= "Kaylan";
  const A: &str = "Alec";

  let mut author_last_names = vec![
  ("CS", T), ("JR", T), 
  ("W", R), ("E", R), 
  ("LC", J), ("J", J), 
  ("K", K), ("HR", K), 
  ("RC", A), ("A", A)];
  author_last_names.sort_by(|a: &(&str, &str), b: &(&str, &str)| a.0.cmp(b.0));
  print!("Author - Obtainer Orderings : {:?}\n", author_last_names);
}

fn example_bin_distributions_at_random() {
  let mut range_list: Vec<(std::ops::Range<i32>, i32)> = stats_functions::create_range_distribution_exclusive(0, 10, 3);
  let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
  const ITERS: usize = 20;

  for _i in 0..ITERS {
    let num: i32 = rng.gen_range(0..30);
    for r in range_list.iter_mut() {
      let in_range: bool = num >= r.0.start && num < r.0.end;
      r.1 += in_range as i32;
    }
  }

  print!("Final bins: {:?}\n", range_list);
}


fn example_bin_distributions_with_data() {
  let mut r_list: Vec<(std::ops::Range<i32>, i32)> = stats_functions::create_range_distribution_exclusive(0, 6, 6);
  let data = [24, 20, 16, 32, 14, 22, 2 ,12, 24, 6, 10, 20,
    8, 16, 12, 24, 14, 20, 18, 14, 16, 18, 20, 22,
    24, 26, 28, 18 ,14, 10, 12, 24, 6 ,12, 18, 16,
    34, 18, 20, 22 ,24 ,26 ,18 ,2 ,18 ,12 ,12, 8,
    24, 10, 14, 16 ,22 ,24 ,22 ,20, 24 ,28 ,20, 22,
    26, 20, 6, 14 ,16 ,18 ,24 ,18 ,16 ,6 , 16, 10,
    14, 18, 24, 22 ,28 ,24 ,30, 34 ,26, 24, 22, 28,
    30, 22, 24 ,22, 32];

  for i in data.iter() {
    let num: i32 = *i;
    for r in r_list.iter_mut() {
      r.1 += r.0.contains(&num) as i32;
    }
  }

  print!("Final bins: {:?}\n", r_list);
}

fn example_trimmed_mean() {
  let mut data = [12000000, 6246950, 5400000, 4917000, 3804360, 3727000, 3710000, 3080000, 2578000, 2098600, 1600000, 1070000, 840360, 813679, 366931];
  let total: i32 = data.iter().sum();
  let mean = total as f32 / data.len() as f32;

  data.sort();
  const END_DELETIONS: usize = 2;
  let t_mean = stats_functions::trimmed_mean(&data, END_DELETIONS).unwrap();
  print!("mean: {}, t mean: {}, var: {}\n", mean, t_mean, stats_functions::variance(&data,mean));
}

fn example_mean_stdev_zscore() {
  let mut dist: Vec<(std::ops::Range<i32>, i32)> = stats_functions::create_range_distribution_exclusive(0, 5, 9);
  dist.push((45..60, 0));
  dist.push((60..90, 0));
  dist.push((90..200, 0));
  
  const MOCK_SAMPLE_SIZE: usize = 100;
  let rel_freqs = &mut [0.04f32, 0.13f32, 0.16f32, 0.17f32, 0.14f32, 0.05f32, 0.12f32, 0.03f32, 0.03f32, 0.06f32, 0.05f32, 0.02f32];
  for i in 0..rel_freqs.len() {
    dist[i].1 += (rel_freqs[i] * MOCK_SAMPLE_SIZE as f32) as i32;
  }
  
  print!("{:?}\n", dist);
  
  let mut total_approximation: f32 = 0f32;
  let mut total_values: i32 = 0;
  for tup in dist.iter() {
    let multiplier: f32 = (tup.0.end - tup.0.start) as f32 / 2f32 + tup.0.start as f32;
    total_approximation += multiplier * tup.1 as f32;
    total_values += tup.1;
  }
  
  print!("Total approximation: {}, total values: {}\n", total_approximation, total_values);
  let mean = total_approximation / MOCK_SAMPLE_SIZE as f32;
  print!("mean: {}\n", mean);
  let mut variance: f32 = 0f32;
  
  // calc std.dev
  for tup in dist.iter() {
    let diff: f32 = (tup.0.end - tup.0.start) as f32 / 2f32 - mean;
    variance += diff.powi(2) * tup.1 as f32;
  }
  
  variance /= (MOCK_SAMPLE_SIZE - 1) as f32;
  let stdev: f32 = variance.sqrt();
  print!("Variance: {}\n", variance);
  print!("Standard Deviation: {}\n", stdev);
  
  const LOWER: f32 = 0f32;
  const UPPER: f32 = 75f32;
  const OTHER_UPPER: f32 = 47f32;
  let z1 = stats_functions::z_score(LOWER, mean, stdev);
  let z2 = stats_functions::z_score(UPPER, mean, stdev);
  let z3 = stats_functions::z_score(OTHER_UPPER, mean, stdev);
  print!("z({}) = {}, z({}) = {}, z({}) = {}\n", LOWER, z1, UPPER, z2, OTHER_UPPER, z3);
  print!("Cheby {}: {} Cheby {}: {} Cheby {}: {}\n", 
  LOWER, stats_functions::chebyshev_rule(z1) / 100f32, 
  UPPER, stats_functions::chebyshev_rule(z2) / 100f32,
  OTHER_UPPER, stats_functions::chebyshev_rule(z3) / 100f32
  );
  
  let mut local_sum = 0;
  for i in dist.iter().enumerate() {
    if i.0 > 10 { break }
    local_sum += i.1.1;
  }
  
  print!("{}\n", local_sum);
  
}

fn example_pcc() {
  let data = vec![(64.6f32, 8011f32), (53f32, 7323f32), (46.3f32, 8735f32), (42.5f32, 7548f32), (38.5f32, 7071f32), (33.9f32, 8248f32)];
  stats_functions::calc_pearson_correlation_coef_float(&data);
}

fn example_pcc_two() {
  let data1 = vec![80, 76, 75, 62, 100, 100, 88, 64, 50, 54, 83];
  let data2 = vec![62, 66, 63, 51, 54, 75, 65, 56, 45, 48, 71];
  print!("{}\n",stats_functions::calc_pcc_int(&data1, &data2).unwrap());
}

fn example_pcc_three() {
  let data1 = vec![150f32, 300f32, 350f32, 3703f32, 390f32, 480f32];
  let data2 = vec![2.3f32, 3.0f32, 4.4f32, 5.0f32, 5.2f32, 5.7f32];
  print!("{}\n",stats_functions::calc_pcc_f32(&data1, &data2).unwrap());
}

fn example_lsr() {
  let x_data: Vec<f32> = vec![2f32, 6f32, 7f32, 9f32, 12f32];
  let y_data: Vec<f32> = vec![90f32, 45f32, 30f32, 5f32, 2f32];
  
  let (m, b) = stats_functions::least_squares(&x_data, &y_data).unwrap_or((0f32, 0f32));
  print!("slope: {}, y_intercept: {}\n", m, b);
  (2..=12).into_iter().map(|x| print!("y({}) = {}\n",x, m * (x as f32) + b)).count();

  let residuals = stats_functions::calc_residuals(&x_data, &y_data, (m, b)).unwrap();
  let ss_resid = residuals.iter().map(|f| f.powi(2)).sum::<f32>();

  let y_mean = stats_functions::calc_mean_simple(&y_data);
  let ss_total = y_data.iter().map(|y| (y - y_mean).powi(2)).sum::<f32>();

  print!("Total sum of squares: {},\nSum of Residual Squares: {},\nResiduals: {:?}\n", ss_total, ss_resid, residuals);
  let coef_of_det = stats_functions::coef_of_determination(ss_resid, ss_total);
  print!("Coef of Determination: {}\n", coef_of_det);
}
  
fn example_sample_space() {
  let mut ss: Vec<(&Gender, &Genres)> = Vec::new();
  for gend in GENDER_LIST.iter() {
    for genre in GENRE_LIST.iter() {
      ss.push((gend, genre));
    }
  }
  
  // PRINTING THE GENERATED SAMPLE SPACE
  print!("{:?}\n", ss);
}

fn simple_conditional_probability<T: Copy + Eq + PartialEq + Hash>(event: &Vec<T>, conditional_event: &Vec<T>) -> f32 {
  return stats_functions::intersect(event, conditional_event).len() as f32 / conditional_event.len() as f32;
}

fn example_ss_2() {
  let SAMPLE_SPACE: Vec<&str> = vec!["LLL", "RLL", "LRL", "LLR", "RRL", "RLR", "LRR", "RRR"];
  let A = vec!["RLL", "LRL", "LLR"];
  let B = vec!["LLL", "RLL", "LRL", "LLR"];
  let C = vec!["LLL", "RRR"];

  print!("A: {:?}, B: {:?}, A UNION B: {:?}\n", A, B, stats_functions::union(&A, &B));
  print!("B: {:?}, C: {:?}, B INTERSECT C: {:?}\n", B, C, stats_functions::intersect(&B, &C));
  print!("C: {:?}, NOT C: {:?}\n", C, stats_functions::not(&SAMPLE_SPACE, &C));
  print!("are A and C disjoint? {}\n", stats_functions::disjoint(&A, &C));

  let abc = stats_functions::arbitrary_union(&[&A, &B, &C]);
  print!("not(A u B u C) = not({:?}) {:?}\n", abc, stats_functions::not(&SAMPLE_SPACE, &abc));
  print!("P(C|B) = {}\n", simple_conditional_probability(&C, &B));
  print!("P(B) = {}\n", simple_conditional_probability(&B, &SAMPLE_SPACE));
}

fn example_simple_probability() -> f32 {
  let events: Vec<f32> = vec![0.30f32, 0.25f32, 0.18f32, 0.15f32, 0.12f32];
  let indexes: Vec<usize> = (0..events.len()).into_iter().collect::<Vec<usize>>();
  let uni: Vec<usize> = stats_functions::union(&stats_functions::union(&indexes[2..3].to_vec(), &indexes[3..4].to_vec()), &indexes[4..5].to_vec());
  let uni2: Vec<usize> = stats_functions::arbitrary_union(&[&indexes[2..3].to_vec(), &indexes[3..4].to_vec(), &indexes[4..5].to_vec()]);
  
  print!("{:?}\n", uni);
  print!("{:?}\n", uni2);
  return uni.iter().map(|index: &usize| events[*index as usize]).sum::<f32>();
}

fn eec_dynamic_energy(voltage: f32, frequency: f32, time: f32, leakage_percent: f32) -> (f32, f32, f32) {
  let power: f32 = voltage.powi(2) * frequency; // assume alpha & C to be constant
  let dt: f32 = 0.001f32;
  let mut t: f32 = 0f32;
  let mut energy: f32 = 0.0f32;

  // simple numerical method of integration
  while t < time {
    energy += power * dt;
    t += dt;
  }

  return (power, energy, energy * leakage_percent);
}

fn eec_hw2_3() {
  let data = [
    ("Overclocked", 1.5f32, 1.5f32, 0.8f32, 1.5f32),
    ("Normal", 1f32, 1f32, 1f32, 1.4f32),
    ("Reduced", 0.8f32, 0.8f32, 1.1f32, 1.6f32),
    ("Green", 0.6f32, 0.6f32, 1.25f32, 1.7f32)
  ];

  for row in data.iter() {
    let (power, energy, energy_leaked) = eec_dynamic_energy(row.1, row.2, row.3, row.4);
    print!("{} = Power: {} _watts, Energy: {} _watts * T, Leakage: {} _watts * T\n", row.0, power, energy, energy_leaked);
  }
}

// DIFFERENT DATA STRUCTURES
// =========================
// SAMPLE SPACE
// PROBABILITY DISTRIBUTIONS
// EVENT SPACE

// COULD PROBABLY MAKE A "PROBABALISTIC" TRAIT
// LETS YOU CALL "PROBABILITY()" ON SOME VARIABLE

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
enum BathTub {
  GAS,
  ELECTRIC
}

impl std::fmt::Display for BathTub {
  // This trait requires `fmt` with this exact signature.
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    match self {
      BathTub::GAS => { write!(f, "{}", "GAS") }
      BathTub::ELECTRIC => { write!(f, "{}", "ELECTRIC")}
    }
  }
}


fn combination_probability<T: Copy + Eq + PartialEq + Hash>(combo: &Vec<T>, p_dist: &HashMap<T, f32>) -> f32 {
  let mut set_probability: f32 = 1f32;
  for event in combo.iter() {
    set_probability *= p_dist.get(event).unwrap();
  }
  return set_probability;
}

fn rec_combo_gen<T: Clone + Copy>(combo_list: &mut Vec<Vec<T>>, buffer: &mut Vec<Option<T>>, events: &Vec<T>, cur_slot: usize, slots: usize) {
  if cur_slot == slots { 
    combo_list.push(buffer.into_iter().flatten().map(|t| *t).collect::<Vec<T>>());
    return ();
  }

  for index in 0..events.len() {
    buffer[cur_slot] = Some(events[index]);
    rec_combo_gen(combo_list, buffer, events, cur_slot + 1, slots);
  }
}

fn generate_combinations<T: Clone + Copy>(events: &Vec<T>, slots: usize) -> Vec<Vec<T>> {
  let mut combos: Vec<Vec<T>> = Vec::new();
  let mut buffer: Vec<Option<T>> = vec![None; slots];
  rec_combo_gen(&mut combos, &mut buffer, events, 0, slots);
  return combos;
}

fn example_discrete_dist() {
  let INITIAL_SAMPLE_SPACE: Vec<BathTub> =  vec![BathTub::GAS, BathTub::GAS, BathTub::GAS, BathTub::ELECTRIC, BathTub::ELECTRIC];
  let event_probs: HashMap<BathTub, f32> = stats_functions::gen_event_prob_map(&INITIAL_SAMPLE_SPACE);

  print!("Event Probabilities:\n");
  for event in event_probs.keys() {
    print!("P({}) = {}\n", event, event_probs.get(event).unwrap());
  }

  // GENERATING COMBINATIONS, CAN PROBABLY BE DONE RECURSIVELY
  const SLOTS: usize = 4;
  let combos: Vec<Vec<BathTub>> = generate_combinations(&event_probs.keys().map(|k| *k).collect::<Vec<BathTub>>(), SLOTS);
  //print!("COMBOES: {:?}\n", combos);

  // LET OUR RANDOM VARIABLE X BE THE NUMBER OF ELECTRIC BATHTUBS
  const SEARCH: BathTub = BathTub::ELECTRIC;
  let mut outcome_probs: HashMap<usize, (Vec<&Vec<BathTub>>, f32)> = HashMap::new();
  for permutation in combos.iter() {
    let X = permutation.iter().filter(|tub_type: &&BathTub| **tub_type == SEARCH).count();
    if !outcome_probs.contains_key(&X) {
      outcome_probs.insert(X, (vec![permutation], combination_probability(permutation, &event_probs)));
    } else {
      let mut_val: &mut (Vec<&Vec<BathTub>>, f32) = outcome_probs.get_mut(&X).unwrap();
      mut_val.0.push(permutation);
      mut_val.1 += combination_probability(permutation, &event_probs);
    }
  }

  print!("Probability of the # of electric stoves being some number X\n");
  let rv_mean: f32 = outcome_probs.keys().map(|k| (*k as f32) * (outcome_probs[k].1)).sum::<f32>();
  let rv_variance_2: f32 = outcome_probs.keys().map(|k| (*k as f32).powi(2) * (outcome_probs[k].1)).sum::<f32>() - rv_mean.powi(2);
  let rv_variance: f32 = outcome_probs.keys().map(|k| (*k as f32 - rv_mean).powi(2) * outcome_probs[k].1).sum::<f32>();

  print!("Expected value: {}, Variance: {}, Variance_2: {}, Standard Deviation: {}\n", rv_mean, rv_variance, rv_variance_2, rv_variance.sqrt());
  for outcome in outcome_probs.keys() {
    print!("P(X = {}) = {}\n", outcome, outcome_probs.get(outcome).unwrap().1);
  }

  print!("P(2 <= X <= 5\n")
}

fn example_monty_hall() {
    let initial_items = vec!['A', 'B', 'C'];
    let combos: Vec<Vec<char>> = generate_combinations(&initial_items, 3);
    print!("{:?}\n", combos);
}

pub fn example_entropy() {
    let v1 = vec![1, 2, 4, 5, 6, 7, 8, 1, 1, 4, 4, 5, 7, 8, 2, 4, 5, 6, 5, 5, 3, 9, 9, 2];
    let entropy: f32 = stats_functions::calc_entropy(&v1);
    print!("The entropy of {:?} is {}\n", v1, entropy);

    let v2 = vec![1,1,1,1,1,1,1,1,1,1,1,2];
    let entropy: f32 = stats_functions::calc_entropy(&v2);
    print!("The entropy of {:?} is {}\n", v2, entropy);

    print!("Joint entropy of {:?} = {}\n", v1.iter().chain(v2.iter()).collect::<Vec<&i32>>(), stats_functions::calc_joint_entropy(&v1, &v2));
    print!("Mutual information: {}\n", stats_functions::mutual_information(&v1,&v2));

    let v3: Vec<i32> = vec![1,2,3,4,5];
    let entropy: f32 = stats_functions::calc_entropy(&v3);
    print!("The entropy of {:?} is {}\n", v3, entropy);

    let v4: Vec<i32> = vec![1,2,3,4,5,6,7,8,9,10,11];
    let entropy: f32 = stats_functions::calc_entropy(&v4);
    print!("The entropy of {:?} is {}\n", v4, entropy);

    let v4: Vec<i32> = vec![1,2,3,4,5,6,7,8,9,10];
    let entropy: f32 = stats_functions::calc_entropy(&v4);
    print!("The entropy of {:?} is {}\n", v4, entropy);
}