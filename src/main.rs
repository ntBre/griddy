use std::error::Error;
use std::io::stdout;
use std::path::Path;

use pbqff::config::Config;
use pbqff::coord_type::cart::{freqs, FirstOutput};
use pbqff::coord_type::findiff::bighash::BigHash;
use pbqff::coord_type::findiff::FiniteDifference;
use pbqff::coord_type::{Cart, Derivative, FirstPart, Nderiv};
use pbqff::{cleanup, optimize, ref_energy, Output, Spectro};
use psqs::geom::Geom;
use psqs::max_threads;
use psqs::program::molpro::Molpro;
use psqs::program::{Job, Program, Template};
use psqs::queue::pbs::Pbs;
use psqs::queue::{Check, Queue};
use symm::Molecule;

fn first_part(
    config: &FirstPart,
    queue: &Pbs,
    nderiv: Nderiv,
    root_dir: impl AsRef<Path>,
    pts_dir: impl AsRef<Path>,
) -> Result<FirstOutput, Box<dyn Error>> {
    let template = Template::from(&config.template);
    let (geom, ref_energy) = if config.optimize {
        let res = optimize::<_, Molpro>(
            &root_dir,
            queue,
            config.geometry.clone(),
            template.clone(),
            config.charge,
        )
        .expect("optimization failed");
        let Some(cart) = res.cart_geom else {
            panic!("failed to extract cart geom from {res:?}");
        };
        (Geom::Xyz(cart), res.energy)
    } else {
        let ref_energy = ref_energy::<_, Molpro>(
            queue,
            config.geometry.clone(),
            template.clone(),
            config.charge,
        );
        (config.geometry.clone(), ref_energy)
    };
    let geom = geom.xyz().expect("expected an XYZ geometry, not Zmat");
    let ndummies = config.dummy_atoms.unwrap_or(0);
    // 3 * (#atoms - #dummy_atoms)
    let n = 3 * (geom.len() - ndummies);
    let nfc2 = n * n;
    let nfc3 = n * (n + 1) * (n + 2) / 6;
    let nfc4 = n * (n + 1) * (n + 2) * (n + 3) / 24;
    let deriv = match nderiv {
        Nderiv::Two => Derivative::Harmonic(nfc2),
        Nderiv::Four => Derivative::Quartic(nfc2, nfc3, nfc4),
    };
    let mut fcs = vec![0.0; nfc2 + nfc3 + nfc4];
    let mut mol = Molecule::new(geom.to_vec());
    if let Some(ws) = &config.weights {
        for (i, w) in ws.iter().enumerate() {
            mol.atoms[i].weight = Some(*w);
        }
    }
    if std::env::var("GRIDDY_NORMALIZE").is_ok() {
        println!("normalizing geometry");
        mol.normalize();
    } else {
        println!("NOT normalizing geometry");
    }
    let pg = mol.point_group();
    println!("geometry:\n{mol}");
    println!("point group:{pg}");
    let mut target_map = BigHash::new(mol.clone(), pg);
    let geoms = Cart.build_points(
        Geom::Xyz(mol.atoms.clone()),
        config.step_size,
        ref_energy,
        deriv,
        &mut fcs,
        &mut target_map,
        n,
    );
    let targets = target_map.values();
    let jobs: Vec<_> = geoms
        .into_iter()
        .enumerate()
        .map(|(job_num, mol)| {
            let filename = format!("job.{job_num:08}");
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
                mol.index,
            )
        })
        .collect();
    let njobs = jobs.len();
    println!("{n} Cartesian coordinates requires {njobs} points");

    // drain into energies
    let mut energies = vec![0.0; njobs];
    let time = queue.drain(
        pts_dir.as_ref().to_str().unwrap(),
        jobs,
        &mut energies,
        Check::None,
    )?;

    eprintln!("total job time: {time:.1} sec");

    Ok(FirstOutput {
        n,
        nfc2,
        nfc3,
        fcs,
        mol,
        energies,
        targets,
        ref_energy,
        pg,
    })
}

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
    } = first_part(
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
    spectro.write_output(&mut stdout(), &output).unwrap();
    std::fs::write("spectro.json", output.to_json_pretty().unwrap()).unwrap();
}
