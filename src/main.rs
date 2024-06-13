use std::fs::read_to_string;
use std::io::stderr;
use std::ops::{Range, RangeInclusive};
use std::path::Path;

use clap::Parser;
use log::info;
use pbqff::cleanup;
use pbqff::coord_type::cart::freqs;
use pbqff::coord_type::findiff::bighash::{BigHash, Target};
use pbqff::coord_type::findiff::FiniteDifference;
use pbqff::coord_type::{Cart, Derivative, FirstPart};
use psqs::geom::Geom;
use psqs::max_threads;
use psqs::program::molpro::Molpro;
use psqs::program::{Job, Program, Template};
use psqs::queue::pbs::Pbs;
use psqs::queue::{Check, Queue};
use serde::Deserialize;
use symm::{Atom, Molecule};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_config() {
        let got = Config::load("testfiles/pbqff.toml");
        assert_eq!(got.yrange, 1..=2);
        assert_eq!(got.zrange, -3..=3);
    }
}

fn optimize(
    opt_dir: impl AsRef<Path>,
    queue: &Pbs,
    geoms: Vec<OptInput>,
    template: Template,
    charge: isize,
) -> Vec<OptOutput> {
    let opt_dir = opt_dir.as_ref();
    let mut jobs = Vec::new();
    let mut ret = Vec::new();
    for (i, geom) in geoms.into_iter().enumerate() {
        let opt_file = opt_dir.join("opt").to_str().unwrap().to_owned();
        jobs.push(Job::new(
            Molpro::new(
                opt_file + &i.to_string(),
                template.clone(),
                charge,
                geom.geometry,
            ),
            i,
        ));
        ret.push(OptOutput {
            y: geom.y,
            z: geom.z,
            ref_energy: None,
            geom: None,
        });
    }
    let mut res = vec![Default::default(); jobs.len()];
    let res = match queue.energize(opt_dir.to_str().unwrap(), jobs, &mut res) {
        Ok(time) => {
            info!("total optimize time: {time:.2} s");
            res
        }
        Err(failed_indices) => {
            info!("filtering out {} failed indices", failed_indices.len());
            assert_eq!(res.len(), ret.len());
            let res = filter_failed(res, &failed_indices);
            ret = filter_failed(ret, &failed_indices);
            assert_eq!(res.len(), ret.len());
            res
        }
    };

    for (i, r) in res.into_iter().enumerate() {
        ret[i].ref_energy = Some(r.energy);
        ret[i].geom = Some(r.cart_geom.unwrap());
    }

    ret
}

fn filter_failed<T>(res: Vec<T>, failed_indices: &[usize]) -> Vec<T> {
    res.into_iter()
        .enumerate()
        .filter(|(i, _)| !failed_indices.contains(i))
        .map(|(_, res)| res)
        .collect()
}

fn first_part(
    config: &FirstPart,
    pts_dir: impl AsRef<Path>,
    OptOutput { y, z, ref_energy, geom }: OptOutput,
    start_index: usize,
) -> BuiltJobs {
    let ref_energy = ref_energy.unwrap();
    let geom = geom.unwrap();
    let template = Template::from(&config.template);
    let ndummies = config.dummy_atoms.unwrap_or(0);
    // 3 * (#atoms - #dummy_atoms)
    let n = 3 * (geom.len() - ndummies);
    let nfc2 = n * n;
    let nfc3 = n * (n + 1) * (n + 2) / 6;
    let nfc4 = n * (n + 1) * (n + 2) * (n + 3) / 24;
    let mut fcs = vec![0.0; nfc2 + nfc3 + nfc4];
    let mut mol = Molecule::new(geom);
    if let Some(ws) = &config.weights {
        for (i, w) in ws.iter().enumerate() {
            mol.atoms[i].weight = Some(*w);
        }
    }
    let pg = mol.point_group();
    eprintln!("geometry {y:.2} {z:.2}:\n{mol}");
    let mut target_map = BigHash::new(mol.clone(), pg);
    let geoms = Cart.build_points(
        Geom::Xyz(mol.atoms.clone()),
        config.step_size,
        ref_energy,
        Derivative::Quartic(nfc2, nfc3, nfc4),
        &mut fcs,
        &mut target_map,
        n,
    );
    let targets = target_map.values();
    let jobs: Vec<_> = geoms
        .into_iter()
        .enumerate()
        .map(|(job_num, mol)| {
            let filename = format!("job.{:08}", job_num + start_index);
            let filename = pts_dir
                .as_ref()
                .join(filename)
                .to_string_lossy()
                .to_string();
            Job::new(
                Molpro::new(
                    filename,
                    template.clone(),
                    config.charge,
                    mol.geom,
                ),
                mol.index + start_index,
            )
        })
        .collect();

    BuiltJobs { n, nfc2, nfc3, fcs, mol, targets, jobs }
}

struct OptInput {
    y: f64,
    z: f64,
    geometry: Geom,
}

fn build_opt_inputs(
    geom_template: &str,
    yrange: RangeInclusive<isize>,
    zrange: RangeInclusive<isize>,
) -> Vec<OptInput> {
    let mut opt_inputs = Vec::new();
    // molpro orients a diatomic molecule along the z-axis, so we need to step
    // He in the yz- (or xz-) plane, with the wider range along z
    for z in zrange.step_by(2).map(|z| z as f64 / 10.0) {
        for y in yrange.clone().step_by(2).map(|y| y as f64 / 10.0) {
            // require {{y}} and {{z}} placeholders in Z-matrix geometry for
            // positioning the He atom for each calculation
            let geometry = Geom::Zmat(
                geom_template
                    .replace("{{y}}", &y.to_string())
                    .replace("{{z}}", &z.to_string()),
            );
            opt_inputs.push(OptInput { y, z, geometry });
        }
    }
    opt_inputs
}

#[derive(serde::Serialize, serde::Deserialize)]
struct OptOutput {
    y: f64,
    z: f64,
    ref_energy: Option<f64>,
    geom: Option<Vec<Atom>>,
}

struct BuiltJobs {
    n: usize,
    nfc2: usize,
    nfc3: usize,
    fcs: Vec<f64>,
    mol: Molecule,
    targets: Vec<Target>,
    jobs: Vec<Job<Molpro>>,
}

struct RunJobs {
    y: f64,
    z: f64,
    n: usize,
    nfc2: usize,
    nfc3: usize,
    fcs: Vec<f64>,
    mol: Molecule,
    targets: Vec<Target>,
    jobs: Range<usize>,
}

/// Serialize `opts` to JSON and save to `path`. Logs any errors, but should
/// never panic.
fn write_opt_checkpoint(opts: &Vec<OptOutput>, path: impl AsRef<Path>) {
    match serde_json::to_string_pretty(opts) {
        Ok(s) => {
            if let Err(e) = std::fs::write(path, s) {
                eprintln!("error writing opt checkpoint: {e:?}");
            }
        }
        Err(e) => {
            eprintln!("error converting opt output to json: {e:?}");
        }
    }
}

/// Load a sequence of [OptOutput]s from the JSON file at `path`.
fn load_opt_checkpoint(path: impl AsRef<Path>) -> Vec<OptOutput> {
    let s = read_to_string(path).unwrap();
    serde_json::from_str(&s).unwrap()
}

#[derive(Parser)]
#[command(author, about, long_about = None)]
struct Args {
    #[arg(value_parser, default_value = "pbqff.toml")]
    config_file: String,

    /// Resume from the opt checkpoint file in the current directory.
    #[arg(short, long, default_value_t = false)]
    checkpoint: bool,

    /// Set the maximum number of threads to use. Defaults to 0, which means to
    /// use as many threads as there are CPUS.
    #[arg(short, long, default_value_t = 0)]
    threads: usize,
}

#[derive(Deserialize)]
struct Config {
    pbqff: pbqff::config::Config,
    yrange: RangeInclusive<isize>,
    zrange: RangeInclusive<isize>,
}

impl Config {
    fn load(path: impl AsRef<Path>) -> Self {
        let s = read_to_string(path).unwrap();
        toml::from_str(&s).unwrap()
    }
}

/// TODO ensure that the molecule is aligned in the same way on the axis for all
/// of the He positions (not flipping sign, which would flip the relative He
/// position). from what I can tell, Molpro is handling this, just verify
fn main() {
    env_logger::init();

    let args = Args::parse();

    let config = Config::load(args.config_file);
    let no_del = false;
    let work_dir = ".";
    let opt_dir = "opt";
    let pts_dir = "pts";
    info!("initializing thread pool with {} threads", args.threads);
    max_threads(args.threads);

    let queue = Pbs::new(
        config.pbqff.chunk_size,
        config.pbqff.job_limit,
        config.pbqff.sleep_int,
        pts_dir,
        no_del,
        config.pbqff.queue_template.clone(),
    );

    info!("cleaning up directories from a previous run");
    cleanup(work_dir);

    info!("building new directories");
    std::fs::create_dir(pts_dir).unwrap();
    std::fs::create_dir(opt_dir).unwrap();

    const OPT_CHK: &str = "opts.json";

    let opts = if args.checkpoint {
        info!("loading optimizations from checkpoint");
        load_opt_checkpoint(OPT_CHK)
    } else {
        let geom_template = config
            .pbqff
            .geometry
            .zmat()
            .expect("griddy requires Z-matrix input");
        let opt_inputs =
            build_opt_inputs(geom_template, config.yrange, config.zrange);

        let template = Template::from(&config.pbqff.template);
        let opts = optimize(
            opt_dir,
            &queue,
            opt_inputs,
            template,
            config.pbqff.charge,
        );

        write_opt_checkpoint(&opts, OPT_CHK);
        opts
    };

    info!("building jobs from opt output");
    let mut run_jobs = Vec::new();
    let mut all_jobs = Vec::new();
    let mut start_index = 0;
    for o @ OptOutput { y, z, .. } in opts {
        let BuiltJobs { n, nfc2, nfc3, fcs, mol, targets, jobs } = first_part(
            &FirstPart::from(config.pbqff.clone()),
            pts_dir,
            o,
            start_index,
        );
        start_index += jobs.len();
        let start = all_jobs.len();
        all_jobs.extend(jobs);
        let end = all_jobs.len();
        run_jobs.push(RunJobs {
            y,
            z,
            n,
            nfc2,
            nfc3,
            fcs,
            mol,
            targets,
            jobs: start..end,
        });
    }

    info!("running jobs");

    // drain into energies
    let mut energies = vec![0.0; all_jobs.len()];
    queue
        .drain(pts_dir, all_jobs, &mut energies, Check::None)
        .unwrap();

    info!("finished running jobs");

    println!("{:>5} {:>5} {:>8} {:>8}", "y", "z", "harm", "corr");

    for RunJobs { y, z, n, nfc2, nfc3, mut fcs, mut mol, targets, jobs } in
        run_jobs
    {
        let (fc2, f3, f4) = Cart.make_fcs(
            targets,
            &energies[jobs],
            &mut fcs,
            n,
            Derivative::Quartic(nfc2, nfc3, 0),
            None::<&str>,
        );

        if let Some(d) = &config.pbqff.dummy_atoms {
            mol.atoms.truncate(mol.atoms.len() - d);
        }

        let (spectro, output) = freqs(None::<&str>, &mol, fc2, f3, f4);
        spectro.write_output(&mut stderr(), &output).unwrap();

        println!(
            "{y:5.2} {z:5.2} {:8.2} {:8.2}",
            output.harms[0], output.corrs[0]
        );
    }
}
