use std::io::stderr;
use std::ops::Range;
use std::path::Path;

use pbqff::cleanup;
use pbqff::config::Config;
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
use symm::{Atom, Molecule};

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
    queue
        .energize(opt_dir.to_str().unwrap(), jobs, &mut res)
        .unwrap();

    for (i, r) in res.into_iter().enumerate() {
        ret[i].ref_energy = Some(r.energy);
        ret[i].geom = Some(r.cart_geom.unwrap());
    }

    ret
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

#[derive(serde::Serialize)]
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

/// TODO ensure that the molecule is aligned in the same way on the axis for all
/// of the He positions (not flipping sign, which would flip the relative He
/// position). from what I can tell, Molpro is handling this, just verify
///
/// TODO combine optimizations and then combine all points. this will better
/// avoid the 5 minute issue on maple and also speed things up overall, I think.
/// just like in semp, I first need to run all of the optimizations together
/// (just pass in each geometry to the very top of first_part), gather all of
/// the results, call the middle part of first_part to build all of the jobs
/// (plus some metadata describing where each chunk is), pass this huge list of
/// jobs to drain, and then divide up the results to pass to the freqs part of
/// run. it looks like FirstOutput is basically this metadata, except that I
/// also need to track the indices into energies to split at
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
    let opt_dir = "opt";
    let pts_dir = "pts";
    max_threads(8);

    println!("{:>5} {:>5} {:>8} {:>8}", "y", "z", "harm", "corr");

    let geom_template = config
        .geometry
        .zmat()
        .cloned()
        .expect("griddy requires Z-matrix input");

    let mut opt_inputs = Vec::new();
    // molpro orients a diatomic molecule along the z-axis, so we need to step
    // He in the yz- (or xz-) plane, with the wider range along z
    for z in (-60..60).step_by(2).map(|z| z as f64 / 10.0) {
        for y in (10..60).step_by(2).map(|y| y as f64 / 10.0) {
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

    let queue = Pbs::new(
        config.chunk_size,
        config.job_limit,
        config.sleep_int,
        pts_dir,
        no_del,
        config.queue_template.clone(),
    );

    cleanup(work_dir);
    std::fs::create_dir(pts_dir).unwrap();
    std::fs::create_dir(opt_dir).unwrap();

    let template = Template::from(&config.template);
    let opts = optimize(opt_dir, &queue, opt_inputs, template, config.charge);

    write_opt_checkpoint(&opts, "opts.json");

    let mut run_jobs = Vec::new();
    let mut all_jobs = Vec::new();
    let mut start_index = 0;
    for o @ OptOutput { y, z, .. } in opts {
        let BuiltJobs { n, nfc2, nfc3, fcs, mol, targets, jobs } = first_part(
            &FirstPart::from(config.clone()),
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

    // drain into energies
    let mut energies = vec![0.0; all_jobs.len()];
    queue
        .drain(pts_dir, all_jobs, &mut energies, Check::None)
        .unwrap();

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

        if let Some(d) = &config.dummy_atoms {
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
