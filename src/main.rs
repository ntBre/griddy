use pbqff::cleanup;
use pbqff::config::Config;
use pbqff::coord_type::{Cart, CoordType};
use psqs::max_threads;
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
    let work_dir = ".";
    let pts_dir = "pts";
    max_threads(8);
    cleanup(work_dir);
    let _ = std::fs::create_dir(pts_dir);
    let (spectro, output) = <Cart as CoordType<_, _, Molpro>>::run(
        Cart,
        work_dir,
        &mut std::io::stdout(),
        &Pbs::new(
            config.chunk_size,
            config.job_limit,
            config.sleep_int,
            pts_dir,
            no_del,
            config.queue_template.clone(),
        ),
        &config,
    );
    spectro
        .write_output(&mut std::io::stdout(), &output)
        .unwrap();
}
