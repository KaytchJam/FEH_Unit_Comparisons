mod utils;

use reqwest::{RequestBuilder, Response};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::utils::{feh_structs::{FehCurrent, FehUnit}, stats_structs::vec_f::VecF};

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
        unit_map.insert(String::from(name), FehUnit { name: String::from(name), stats: get_stats(unit_json)});
      }
    }
  }

  return Ok(unit_map);
}

// Return the sign of the integer as a character
fn sign_to_char(i: i32) -> char {
  return if i < 0 { '-' }
    else if i > 0 { '+' }
    else { '\0' }
}

// Print the k closest vectors from the sorted list
pub fn print_k_closest(all_units: &HashMap<String, FehUnit>, unit: &&str, unit_stats: &VecF, list: &Vec<&String>, k: usize) -> () {
  println!("------------------------------------------\nThe top {} [statwise] closest units to {} were...\n------------------------------------------", k, unit);
  for index in 0..k {
      let close_hero = list[index];
      println!("{}) {}, Diffs = {}", index + 1, close_hero, vector_to_string_diffs(&(&all_units.get(close_hero).unwrap().stats - unit_stats)));
  }
  println!("------------------------------------------\n")
}

// Print the k farthest vectors from the sorted list
pub fn print_k_farthest(all_units: &HashMap<String, FehUnit>, unit: &&str, unit_stats: &VecF, list: &Vec<&String>, k: usize) -> () {
  println!("------------------------------------------\nThe top {} [statwise] closest units to {} were...\n------------------------------------------", k, unit);
  for index in ((list.len() - k)..(list.len())).rev() {
      let close_hero = list[index];
      println!("{}) {}, Diffs = {}", list.len() - index, close_hero, vector_to_string_diffs(&(&all_units.get(close_hero).unwrap().stats - unit_stats)));
  }
  println!("------------------------------------------\n")
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

fn main() {
  let all_units: HashMap<String, FehUnit> = create_unit_dataset();
  let mut cur: FehCurrent = FehCurrent::new();

  // Get user input
  let mut user_in: String = String::new();
  println!("Enter a unit name: ");
  if let Err(e) = std::io::stdin().read_line(&mut user_in) {
    println!("Error encountered: {}", e.to_string());
    panic!();
  }

  // Stat add-ons
  const MERGES: usize = 0;
  const DRAGONFLOWERS: usize = 0;

  // Unit Retrieval
  user_in = String::from(&user_in[0..(user_in.len()-2)]);
  let unit: &FehUnit = all_units.get(&user_in).unwrap();
  cur.set_unit(unit);
  println!("{} : stats = {:?}", unit.name, unit.stats);

  // Add modifiers
  cur.add_merges(MERGES).add_dragonflowers(DRAGONFLOWERS);
  println!("{} : stats + {} merges + {} DFs = {:?}", unit.name, MERGES, DRAGONFLOWERS, cur.current_stats);

  // Compare the unit
  let nearest = utils::feh_structs::k_nearest_units(&all_units, &user_in, cur.current_stats.as_ref().unwrap());
  print_k_closest(&all_units, &user_in.as_str(), cur.current_stats.as_ref().unwrap(), &nearest, 10);
  print_k_farthest(&all_units, &user_in.as_str(), cur.current_stats.as_ref().unwrap(), &nearest, 10);
}

fn execute_feh_cli() {
  print!("Initializing data...");
  let hero_cur = FehCurrent::new();
  println!("Welcome to the FEH Command Line Executable.\n");
}

// Generate a hashmap from the JSON FEH Unit Data
fn create_unit_dataset() -> HashMap<String, FehUnit> {
  const GAMEPRESS_JSON_URL: &str = "https://gamepress.gg/sites/default/files/aggregatedjson/hero-list-FEH.json?2040027994887217598";

  // retrive json
  let data: String = attempt_request(String::from(GAMEPRESS_JSON_URL), 3);
  println!("HTTP Request to Gamepress successful.");

  // fill in our "all_units" vector
  let all_units: HashMap<String, FehUnit> = match extract_data(data.as_str()) {
    Err(e) => panic!("{}", e.to_string()),
    Ok(data) => data
  };

  return all_units;
}

fn peuso_main() {
  let all_units: HashMap<String, FehUnit> = create_unit_dataset();
  
  // do interesting stuff w/ data
  const UNIT_NAME: &str = "Brave Tiki (Adult)";
  println!("Finding nearest neighbors to {}...", UNIT_NAME);
  //let (unit, list): (&&str, Vec<&String>) = k_nearest_units(&all_units, &UNIT_NAME);

  //print_k_closest(&all_units, unit, &list, 10);
  //print_k_farthest(&all_units, unit, &list, 10);

  let com = utils::feh_structs::get_center_of_mass(&all_units);
  println!("Stat center of mass: {:?}", com);
  //println!("The farthest unit from the center of mass is {:?}", farthest_from(&all_units, com));

  // let dists = distance_from_vec(&all_units, &com);
  // print_k_closest_vec(&all_units, &com, &dists, 10);
  // print_k_farthest_vec(&all_units, &com, &dists, 10);
}

// General Loop 
/*
{
  unit: Option<&FehUnit> = None;
  while true {
    request_input();
    read_input();

    match in {
      out_1 => {},
      out_2 => {},
      out_3 => {},
      _ => reread_input()
    };
  }


}
*/