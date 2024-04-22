### FEH Unit Comparisons

#### What is this project about?

This project is a way for me to practice making API requests w/ Rust. I'm working with "units" from the game "Fire Emblem Heroes," a mobile ~~gacha~~ game and comparing unit stat similarity based on various distance metrics (Euclidean, Minkowski, Cosine, ...).

#### About FEH Units.

Units have various attributes that affect gameplay: weapon, movement skill, special skill, general skills, seals, as well as augments in the form of dragon flowers and merges. Beyond the design and presentation of the character themselves, the 4 immutable aspects of a fire emblem heroes unit are: their stats, their weapon type, the unit's color, and their movement type. In this case we're looking exclusively at the unit's stat distribution `(HP, ATTACK, SPEED, DEFENSE, RES)` and seeing how we can relate units to each other based off said distribution.

The data for the project was retreived from the Gamepress Fire Emblem Heroes Database. The database isn't entirely up to date (As of writing this the last update to the Gamepress FEH database was October 10th, 2023). JSON data for each unit is fetched and each unit's data is parsed into a FehUnit struct defined below:

```rust
struct FehUnit {
    name: String, // Store the unit's name
    stats: VecF // Store the unit's stat
}
```

The struct itself is pretty straightforward. Every unit has their name stored as a String owned by the struct. Stats are represented by a VecF which is a `std::vec::Vec` wrapper for linear algebra n-sized vector operations.

#### The General Loop.

On execution of `main` the user inputs the name of the unit they'd like to form comparisons in reference to. Using different "distance metrics" a list of the k-nearest and k-farthest units to the chosen unit is made and then printed to terminal. An option also exists to export all distances from a selected unit to a `.csv` file; this option is given as a prompt to the user during runtime. 

```rust
/// For managing the current unit being used as reference
struct FehCurrent<'a> {
    current_unit: Option<&'a FehCurrent>,
    current_stats: Option<VecF>,
    dragonflowers: usize,
    merges: usize,
    stat_priority_list: Option<[u8; 5]>,
    merge_idx: (usize, usize),
    df_idx: usize
}

/// Easy way of enumerating the different distance metrics used in comparisons
enum DistanceMetric {
    EUCLIDEAN = 0,
    COSINE = 1,
    MANHATTAN = 2,
    MINKOWSKI = 3 // p-norm with p = 1.5
}
```

The general loop looks something like below:

```rust
const OUTPUT_PATH: &str = "output/metrics/";
const NUM_TO_PRINT: usize = 10;

// largely pseudocode by the actual file looks something like this
fn main() {
    // INITIALIZATION & INPUTS
    let all_units: &HashMap<String, FehUnit> = gen_unit_map();
    let user_in: String = get_unit_input();
    let mut cur: FehCurrent = FehCurrent::new();
    
    // GET THE UNIT, SET CURRENT UNIT TO SELECTED UNIT
    cur.set_unit(all_units[user_in].unwrap());
    let export_to_csv = get_export_input();

    // ITERATE THROUGH EVERY DISTANCE METRIC, SHOW RESULTS
    for metric in DistanceMetric::into_iter() {
        let distances: Vec<&String> = all_unit_distances(&cur, &all_units, metric);
        print_top_k(&cur, &all_units, &distances, metric);

        if export_to_csv { 
            dist_to_csv(&cur, &all_units, &distances, metric, OUTPUT_PATH); 
        }
    }

    return ();
}
```

The actual file has a bit more. For example, if a input `user_in` doesn't match any unit name in the `all_units` HashMap we then do a edit_distance comparison with all units in `all_units` and give the user a subset of possible approximations to choose from. Various print statements and variable storages are ommitted as well.

#### Future Plans

- [ ] Export graphs top K nearest & farthest
- [ ] Prompt to let user select a different character
- [ ] Prompt to left user input modifiers
    - [ ] Dragonflower prompt
    - [ ] Merge Prompt
- [ ] Remove Dragonflowers function
- [ ] Remove Merges function



