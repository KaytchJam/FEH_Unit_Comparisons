use reqwest::{RequestBuilder, Response};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use super::{utils::stats_structs::vec_f::VecF, FehUnit, DistanceMetric};

use std::fs::File;
use std::io::Write;
use std::path::Path;

// FOR JSON SERIALIZATION
#[derive(Serialize, Deserialize, Debug)]
pub struct FehJson {
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

// Takes a FehJson struct and turns it into a VecF vector
pub fn get_stats(fj: FehJson) -> VecF {
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

// Converts a vector of strings into a vector of floats
fn str_to_f32(vof64: Vec<String>) -> Vec<f32> {
    return vof64.iter()
      .map(|s| s.parse::<i32>()
      .unwrap() as f32)
      .collect::<Vec<f32>>();
}

// extract the json data
fn extract_data(data: &str) -> serde_json::Result<HashMap<String,FehUnit>> {
    let mut unit_map: HashMap<String,FehUnit> = HashMap::new();
    let mut as_serde: serde_json::Value = serde_json::from_str(data)?;

    const APSOTROPHE: &str = "&#039;";
  
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
          let mut out_str = String::from(unit_json.title_1.as_str());
          if unit_json.title_1.contains(APSOTROPHE) { out_str = out_str.replace(APSOTROPHE, "'"); }
          let name = out_str.as_str();
          unit_map.insert(String::from(name).to_ascii_uppercase(), FehUnit { name: String::from(name), stats: get_stats(unit_json)});
        }
      }
    }
  
    return Ok(unit_map);
}

// Generate a hashmap from the JSON FEH Unit Data
pub fn create_unit_dataset() -> HashMap<String, FehUnit> {
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

// Writes each FehUnit to the csv file
fn write_unit(f: &mut File, unit: &FehUnit) -> Result<(), std::io::Error> {
  f.write((unit.name.clone() + ",").as_bytes())?; // write name with comma
  for idx in 0..=3 { f.write(((unit.stats.get(idx) as u32).to_string() + ",").as_bytes())?; } // write HP ~ DEF with commas
  f.write(((unit.stats.get(4) as u32).to_string() + "\n").as_bytes())?; // write res with newline
  return Ok(());
}

// Converts the Hashmap dataset into a CSV file
pub fn dataset_to_csv(all_units: &HashMap<String, FehUnit>, target_file: &str) -> bool {
  if Path::new(target_file).exists() { 
    println!("File already exists.");
    return true; 
  }
  // INITS
  const COLUMNS: [&str; 6] = ["Title,", "HP,", "ATK,", "SPD,", "DEF,", "RES\n"];
  let mut f: File = File::create(target_file).expect("Wasn't able to create file");

  // WRITES
  for col in COLUMNS { f.write(col.as_bytes()).expect("couldn't write col in COLUMNS to the target_file");} // WRITE COLUMNS
  for (_, unit) in all_units.iter() { write_unit(&mut f, unit).expect("Failed to write the unit to file");} // WRITE UNIT DATA
  return false; // file closes on drop
}

fn write_list(f_name: &str, all_units: &HashMap<String, FehUnit>, columns: &[&str], unit: &FehUnit, nearest: core::slice::Iter<&String>) {
  let mut f: File = File::create(f_name).expect("couldn't create the metric file");
  for col in columns { f.write(col.as_bytes()).expect("couldn't write col in COLUMNS to the target_file");} // WRITE COLUMNS
  f.write("0,".as_bytes()).expect("Failed to write leading 0");
  write_unit(&mut f, unit).expect("Failed to write unit");

  for (idx, unit_name) in nearest.enumerate() {
    f.write(((idx+1).to_string() + ",").as_bytes()).expect("Failed to convert the \"rank\" to bytes");
    let temp_unit = all_units.get(*unit_name).unwrap();
    write_unit(&mut f, temp_unit).expect("Failed to write unit");
  }
}

pub fn save_nearest_to_csv(all_units: &HashMap<String, FehUnit>, nearest: &Vec<&String>, unit: &FehUnit, metric: DistanceMetric, directory: &str) {
  // Path to create:
  let character_dir = unit.name.clone().replace("'","").replace(" ","_").to_ascii_lowercase();

  if !Path::new(&(directory.to_owned() + character_dir.as_str())).exists() {
    std::fs::create_dir(directory.to_owned() + "/" + character_dir.as_str()).expect("Couldn't create the directory");
  }
  
  let mut metric_file_name = "/".to_owned() + character_dir.as_str() +  "_" + metric.to_string().replace(" (1.5)", "").as_str();
  metric_file_name = directory.to_owned() + character_dir.as_str() + metric_file_name.to_ascii_lowercase().as_str();
  print!("{}", metric_file_name);

  // INITS
  const COLUMNS: [&str; 7] = ["Rank,", "Title,", "HP,", "ATK,", "SPD,", "DEF,", "RES\n"];
  write_list((metric_file_name.clone() + "_nearest").as_str(), all_units, &COLUMNS, unit, nearest.iter());
  write_list((metric_file_name.clone() + "_farthest").as_str(), all_units, &COLUMNS, unit, nearest.iter());
}