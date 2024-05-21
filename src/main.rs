use pbqff::config::Config;
use pbqff::coord_type::{Cart, CoordType};
use psqs::program::molpro::Molpro;
use psqs::queue::pbs::Pbs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_file = if args.len() < 2 {
        "pbqff.toml"
    } else {
        args[1].as_str()
    };
    let config = Config::load(config_file);
    let no_del = false;
    let (spectro, output) = <Cart as CoordType<_, _, Molpro>>::run(
        Cart,
        ".",
        &mut std::io::stdout(),
        &Pbs::new(
            config.chunk_size,
            config.job_limit,
            config.sleep_int,
            "pts",
            no_del,
            config.queue_template.clone(),
        ),
        &config,
    );
    spectro
        .write_output(&mut std::io::stdout(), &output)
        .unwrap();
}
