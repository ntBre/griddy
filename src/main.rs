use std::io::stdout;
use std::path::Path;

use pbqff::config::Config;
use pbqff::coord_type::cart::{freqs, FirstOutput};
use pbqff::coord_type::findiff::FiniteDifference;
use pbqff::coord_type::{Cart, Derivative, FirstPart, Nderiv};
use pbqff::{cleanup, Output, Spectro};
use psqs::max_threads;
use psqs::program::molpro::Molpro;
use psqs::queue::pbs::Pbs;

fn run(
    dir: impl AsRef<Path>,
    queue: &Pbs,
    config: &Config,
) -> (Spectro, Output) {
    let FirstOutput {
        n,
        nfc2,
        nfc3,
        mut fcs,
        mut mol,
        energies,
        targets,
        ..
    } = Cart
        .first_part::<_, _, Molpro>(
            &mut stdout(),
            &FirstPart::from(config.clone()),
            queue,
            Nderiv::Four,
            &dir,
            dir.as_ref().join("pts"),
        )
        .unwrap();

    let freq_dir = &dir.as_ref().join("freqs");
    let (fc2, f3, f4) = Cart.make_fcs(
        targets,
        &energies,
        &mut fcs,
        n,
        Derivative::Quartic(nfc2, nfc3, 0),
        Some(freq_dir),
    );

    if let Some(d) = &config.dummy_atoms {
        mol.atoms.truncate(mol.atoms.len() - d);
    }

    freqs(Some(freq_dir), &mol, fc2, f3, f4)
}

/// At a high level, all I have to do is wrap this `run` call in two loops over
/// He positions. That will bring me back to the shell script version I was
/// running before. At that point, I need to start diving into the pbqff code
/// and duplicating it in many cases, removing anything that changes the
/// positions of the atoms. In particular, I will need to remove any calls to
/// `normalize`. I should also check if Molpro has an option to prevent geometry
/// reorientation because I need to ensure that the molecule is aligned in the
/// same way on the axis for all of the He positions (not flipping sign, which
/// would flip the relative He position)
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
    let (spectro, output) = run(
        work_dir,
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
