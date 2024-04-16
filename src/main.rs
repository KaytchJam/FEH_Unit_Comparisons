
mod stats_structs;

use reqwest::{RequestBuilder, Response};
use serde::{Serialize, Deserialize};
use std::{cmp, collections::{BinaryHeap, HashMap}};
use crate::stats_structs::vec_f::VecF;

// BASIC STRUCT FOR DESCRIBING A FEH UNIT
#[derive(Debug)]
struct FehUnit {
  name: String,
  stats: VecF // size 5
}

// FOR JSON SERIALIZATION
#[derive(Serialize, Deserialize, Debug)]
struct FehJson {
  atk: String,
  attr: String,
  def: String,
  hp: String,
  icon: String,
  illustrator: String,
  movement: String,
  rating: String,
  rating_old: String,
  rating_value: String,
  res: String,
  spd: String,
  stamp: String,
  stars: String,
  tier_icon: String,
  tier_icon_old: String,
  title: String,
  title_1: String,
  total: String,
  va_en: String,
  va_jp: String
}

// ATTEMPT TO SEND A REQUEST 'max_attempts' TIMES
fn attempt_request(url: String, max_attempts: usize) -> String {
  assert!(max_attempts > 0);

  let mut req_result: Result<String, Box<dyn std::error::Error>>;
  let mut attempt: usize = 0;

  // Potential failure block
  while attempt < max_attempts {
    req_result = send_request(&url);
    if req_result.is_ok() { return req_result.unwrap(); }
    attempt += 1;
    print!("Requests failures = {}", attempt);
  }

  panic!();
}

// SEND THE REQUEST TO GAMEPRESS
#[tokio::main]
async fn send_request(req: &String) -> Result<String, Box<dyn std::error::Error>> {
    let client = reqwest::Client::builder()
        .build()?;

    let request: RequestBuilder = client.request(reqwest::Method::GET, req);
    let response: Response = request.send().await?;
    let body: String = response.text().await?;
    Ok(body)
}

fn str_to_f32(vof64: Vec<String>) -> Vec<f32> {
  return vof64.iter()
    .map(|s| s.parse::<i32>()
    .unwrap() as f32)
    .collect::<Vec<f32>>();
}

fn get_stats(fj: FehJson) -> VecF {
  return VecF::from(str_to_f32(
    vec![
      fj.hp, 
      fj.atk, 
      fj.spd,
      fj.def,
      fj.res,
    ])
  );
}

// extract the json data
fn extract_data(data: &str) -> serde_json::Result<HashMap<String,FehUnit>> {
  let mut unit_map: HashMap<String,FehUnit> = HashMap::new();
  let mut as_serde: serde_json::Value = serde_json::from_str(data)?;

  for unit in as_serde.as_array_mut().unwrap().into_iter() {
    // convert the json string into a fehjson struct
    match serde_json::from_value::<FehJson>(unit.take()) {
      // Check for error in FehJson creation
      Err(error) => {
        print!("Failure. {}", error.to_string());
        panic!();
      },

      // Turn FehJson into proper Feh Struct
      Ok(unit_json) => {
        let name = unit_json.title_1.as_str();
        unit_map.insert(String::from(name), FehUnit { name: String::from(name), stats: get_stats(unit_json) });
      }
    }
  }

  return Ok(unit_map);
}

struct UnitDistance<'a>(&'a String, f32);

const EPSILON: f32 = 0.0001f32;

impl<'a> std::cmp::Eq for UnitDistance<'a> {
}

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

/*
fn retrive_k_units(unit_heap: BinaryHeap<UnitDistance>, k: usize) -> Vec<String> {
  return vec![];
}
*/

fn k_nearest_units<'a>(all_units: &'a HashMap<String, FehUnit>, unit: &'a&'a str) -> (&'a &'a str, Vec<&'a String>) {
  let ref_unit: &FehUnit = all_units.get(&String::from(*unit)).unwrap();
  let mut unit_heap: BinaryHeap<UnitDistance> = BinaryHeap::new();
 
  for (unit_name, unit_stats_vec) in all_units.iter() {
    if *unit_name != ref_unit.name {
      let dist = VecF::euclid_distance(&ref_unit.stats, &unit_stats_vec.stats);
      unit_heap.push(UnitDistance(unit_name, dist));
    }
  }

  return (unit, unit_heap
    .into_sorted_vec()
    .into_iter()
    .map(|hero| hero.0)
    .collect::<Vec<&String>>());
}

fn distance_from_vec<'a>(all_units: &'a HashMap<String, FehUnit>, v: &VecF) -> Vec<&'a String> {
  let mut unit_heap: BinaryHeap<UnitDistance> = BinaryHeap::new();
 
  for (unit_name, unit_stats_vec) in all_units.iter() {
    let dist = VecF::euclid_distance(&v, &unit_stats_vec.stats);
    unit_heap.push(UnitDistance(unit_name, dist));
  }

  return unit_heap
    .into_sorted_vec()
    .into_iter()
    .map(|hero| hero.0)
    .collect::<Vec<&String>>();
}

fn hero_stat_diffs(all_units: &HashMap<String, FehUnit>, from_hero: &String, to_hero: &String) -> VecF {
  return &all_units.get(to_hero).unwrap().stats - &all_units.get(from_hero).unwrap().stats;
}

fn sign_to_char(i: i32) -> char {
  return if i < 0 { '-' }
    else if i > 0 { '+' }
    else { '\0' }
}

fn float_stat_delta_string(i: i32) -> String {
  return String::from(sign_to_char(i)) + i.abs().to_string().as_str();
}

fn vector_to_string_diffs(stat_vec: &VecF) -> String {
  return format!("[Hp: {} Atk: {} Spd: {} Def: {}, Res: {}]", 
    float_stat_delta_string(stat_vec.get(0) as i32), 
    float_stat_delta_string(stat_vec.get(1) as i32), 
    float_stat_delta_string(stat_vec.get(2) as i32),
    float_stat_delta_string(stat_vec.get(3) as i32),
    float_stat_delta_string(stat_vec.get(4) as i32)
  );
}

fn print_k_closest(all_units: &HashMap<String, FehUnit>, unit: &&str, list: &Vec<&String>, k: usize) -> () {
  println!("------------------------------------------\nThe top {} [statwise] closest units to {} were...\n------------------------------------------", k, unit);
  let unit_string = &String::from(*unit);
  for index in 0..k {
    let close_hero = list[index];
    println!("{}) {}, Diffs = {}", index + 1, close_hero, vector_to_string_diffs(&hero_stat_diffs(all_units, unit_string, close_hero)));
  }
  println!("------------------------------------------\n")
}

fn print_k_closest_vec(all_units: &HashMap<String, FehUnit>, v: &VecF, list: &Vec<&String>, k: usize) -> () {
  println!("------------------------------------------\nThe top {} [statwise] closest units to vector {:?} were...\n------------------------------------------", k, v);
  for index in 0..k {
    let close_hero = list[index];
    println!("{}) {}, Diffs = {}", index + 1, close_hero, vector_to_string_diffs(&(&all_units.get(close_hero).unwrap().stats - v)));
  }
  println!("------------------------------------------\n")
}

fn print_k_farthest(all_units: &HashMap<String, FehUnit>, unit: &&str, list: &Vec<&String>, k: usize) -> () {
  println!("------------------------------------------\nThe top {} [statwise] farthest units to {} were...\n------------------------------------------", k, unit);
  let unit_string = &String::from(*unit);
  for index in ((list.len() - k)..(list.len())).rev() {
    let far_hero = list[index];
    println!("{}) {}, Diffs = {}", list.len() - index, far_hero, vector_to_string_diffs(&hero_stat_diffs(all_units, unit_string, far_hero)));
  }
  println!("------------------------------------------\n")
}

fn print_k_farthest_vec(all_units: &HashMap<String, FehUnit>, v: &VecF, list: &Vec<&String>, k: usize) -> () {
  println!("------------------------------------------\nThe top {} [statwise] farthest units to vector {:?} were...\n------------------------------------------", k, v);
  for index in ((list.len() - k)..(list.len())).rev() {
    let close_hero = list[index];
    println!("{}) {}, Diffs = {}", list.len() - index, close_hero, vector_to_string_diffs(&(&all_units.get(close_hero).unwrap().stats - v)));
  }
  println!("------------------------------------------\n")
}

fn get_center_of_mass(all_units: &HashMap<String, FehUnit>) -> VecF {
  let mut com = VecF::zeroes(5);
  for (_ , unit) in all_units.iter() {
    com = &com + &unit.stats;
  }

  return (1f32 / (all_units.len() as f32)) * (com);
}

fn main() {
  const GAMEPRESS_JSON_URL: &str = "https://gamepress.gg/sites/default/files/aggregatedjson/hero-list-FEH.json?2040027994887217598";

  // retrive json
  let data: String = attempt_request(String::from(GAMEPRESS_JSON_URL), 3);
  println!("HTTP Request to Gamepress successful.");

  // fill in our "all_units" vector
  let all_units: HashMap<String, FehUnit> = match extract_data(data.as_str()) {
    Err(e) => panic!("{}", e.to_string()),
    Ok(data) => data
  };
  
  // do interesting stuff w/ data
  const UNIT_NAME: &str = "Arden";
  println!("Finding nearest neighbors to {}...", UNIT_NAME);
  let (unit, list): (&&str, Vec<&String>) = k_nearest_units(&all_units, &UNIT_NAME);

  print_k_closest(&all_units, unit, &list, 10);
  print_k_farthest(&all_units, unit, &list, 10);

  let com = get_center_of_mass(&all_units);
  println!("Stat center of mass: {:?}", com);
  //println!("The farthest unit from the center of mass is {:?}", farthest_from(&all_units, com));

  let dists = distance_from_vec(&all_units, &com);
  print_k_closest_vec(&all_units, &com, &dists, 10);
  print_k_farthest_vec(&all_units, &com, &dists, 10);
}
