use std::error::Error;
use std::io::{stdin, Cursor, Read};

fn main() -> Result<(), Box<dyn Error>> {
    let stdin = full_stdin()?;
    let mut reader = Cursor::new(stdin);
    let mesh = stl_io::read_stl(&mut reader)?;
    mesh.validate()?;

    Ok(())
}

fn full_stdin() -> Result<Vec<u8>, Box<dyn Error>> {
    let mut vec = Vec::new();
    stdin().lock().read_to_end(&mut vec)?;
    Ok(vec)
}
