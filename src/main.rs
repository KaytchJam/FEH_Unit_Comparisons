mod utils;
mod feh_json;

use std::collections::HashMap;
use crate::utils::feh_structs::{DistanceMetric, FehCurrent, FehUnit};

fn clear_screen() {
  print!("{}[2J", 27 as char);
}

fn main() {
  let all_units: HashMap<String, FehUnit> = feh_json::create_unit_dataset();
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
  clear_screen();
  user_in = String::from(&user_in[0..(user_in.len()-2)]);
  let unit: &FehUnit = all_units.get(&user_in).unwrap();
  cur.set_unit(unit);
  println!("{} : stats = {:?}", unit.name, unit.stats);

  // Add modifiers
  cur.add_merges(MERGES).add_dragonflowers(DRAGONFLOWERS);
  println!("{} : stats + {} merges + {} DFs = {:?}", unit.name, MERGES, DRAGONFLOWERS, cur.current_stats);

  // Pre-loop stuffs
  let cur_stats = cur.current_stats.as_ref().unwrap();
  let user_str = &user_in.as_str();
  let mut buffer: String = String::new();

  // Iterate through every metric -> Make Unit Comparisons
  for metric in DistanceMetric::into_iter() {
    let nearest: Vec<&String> = utils::feh_structs::k_nearest_units(&all_units, &user_in, cur_stats, metric);
    utils::feh_structs::print_k_closest(&all_units, user_str, cur_stats, &nearest, 10, metric);
    utils::feh_structs::print_k_farthest(&all_units, user_str, cur_stats, &nearest, 10, metric);

    println!("Enter anything to continue...");
    _ = std::io::stdin().read_line(&mut buffer);
    clear_screen();
  }
}

// fn execute_feh_cli() {
//   print!("Initializing data...");
//   let hero_cur = FehCurrent::new();
//   println!("Welcome to the FEH Command Line Executable.\n");
// }

// fn peuso_main() {
//   let all_units: HashMap<String, FehUnit> = feh_json::create_unit_dataset();
  
//   // do interesting stuff w/ data
//   const UNIT_NAME: &str = "Brave Tiki (Adult)";
//   println!("Finding nearest neighbors to {}...", UNIT_NAME);
//   //let (unit, list): (&&str, Vec<&String>) = k_nearest_units(&all_units, &UNIT_NAME);

//   //print_k_closest(&all_units, unit, &list, 10);
//   //print_k_farthest(&all_units, unit, &list, 10);

//   let com = utils::feh_structs::get_center_of_mass(&all_units);
//   println!("Stat center of mass: {:?}", com);
//   //println!("The farthest unit from the center of mass is {:?}", farthest_from(&all_units, com));

//   // let dists = distance_from_vec(&all_units, &com);
//   // print_k_closest_vec(&all_units, &com, &dists, 10);
//   // print_k_farthest_vec(&all_units, &com, &dists, 10);
// }

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