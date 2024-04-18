use reqwest::{RequestBuilder, Response};
use serde::{Serialize, Deserialize};
use super::utils::stats_structs::vec_f::VecF;
use std::collections::HashMap;
use super::FehUnit;

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

