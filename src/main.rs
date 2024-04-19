mod utils;
mod feh_json;

use std::collections::HashMap;
use crate::utils::feh_structs::{DistanceMetric, FehCurrent, FehUnit};

fn clear_screen() {
  print!("{}[2J{}[1;1H", 27 as char, 27 as char);
}

fn main() {
  const RETREIVE_ON_ERROR: usize = 8;

  let all_units: HashMap<String, FehUnit> = feh_json::create_unit_dataset();
  let mut cur: FehCurrent = FehCurrent::new();

  // Get user input
  let mut user_in: String = String::new();
  println!("Enter a unit name: ");
  if let Err(e) = std::io::stdin().read_line(&mut user_in) {
    println!("Error encountered: {}", e.to_string());
    panic!();
  }
  
  // Unit Retrieval
  user_in = String::from(&user_in[0..(user_in.len()-2)]);
  let unit: &FehUnit = match all_units.get(&user_in.to_ascii_uppercase()) {
    Some(unit) => unit,

    None => {
      clear_screen();
      println!("{} could not be found. Did you mean...?", user_in);
      let set = utils::feh_structs::get_n_closest(&all_units, &user_in, RETREIVE_ON_ERROR);
      println!("Enter the index of the character you meant to type in:");

      let mut index_in: String = String::new();
      if let Err(e) = std::io::stdin().read_line(&mut index_in) {
        println!("Error encountered: {}", e.to_string());
      }

      let n = index_in.trim().parse::<usize>().unwrap();
      user_in = String::from(set[n-1].as_str());
      all_units.get(&set[n-1].to_ascii_uppercase()).as_ref().unwrap()
    }
  };
  
  clear_screen();
  cur.set_unit(unit);
  println!("{} : stats = {:?}", unit.name, unit.stats);
  
  // Stat add-ons
  const MERGES: usize = 0;
  const DRAGONFLOWERS: usize = 0;

  // Add modifiers
  cur.add_merges(MERGES).add_dragonflowers(DRAGONFLOWERS);
  println!("{} : stats + {} merges + {} DFs = {:?}", unit.name, MERGES, DRAGONFLOWERS, cur.current_stats);

  // Pre-loop stuffs
  let cur_stats = cur.current_stats.as_ref().unwrap();
  let user_str = &user_in.as_str();
  let mut buffer: String = String::new();

  println!("Press ENTER to show top 10 closest & farthest units...");
  _ = std::io::stdin().read_line(&mut buffer);
    clear_screen();

  // Iterate through every metric -> Make Unit Comparisons
  for (i, metric) in DistanceMetric::into_iter().enumerate() {
    let nearest: Vec<&String> = utils::feh_structs::k_nearest_units(&all_units, &user_in, cur_stats, metric);
    utils::feh_structs::print_k_closest(&all_units, user_str, cur_stats, &nearest, 10, metric);
    utils::feh_structs::print_k_farthest(&all_units, user_str, cur_stats, &nearest, 10, metric);

    if i != DistanceMetric::cardinality() - 1 { println!("Press ENTER to move on..."); } else { println!("Press ENTER to exit..."); }
    _ = std::io::stdin().read_line(&mut buffer);
    clear_screen();
  }

  println!("Finished!");
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