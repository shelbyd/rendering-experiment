use image::{ImageBuffer, Rgb};
use std::error::Error;
use std::io::{stdin, Cursor, Read};
use stl_io::IndexedMesh;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

type Result<T> = core::result::Result<T, Box<dyn Error>>;

fn main() -> Result<()> {
    let stdin = full_stdin()?;
    let mesh = parse_mesh(&stdin)?;
    let image = render(&mesh)?;
    image.save_with_format("/dev/stdout", image::ImageFormat::PNG)?;
    Ok(())
}

fn full_stdin() -> Result<Vec<u8>> {
    let mut vec = Vec::new();
    stdin().lock().read_to_end(&mut vec)?;
    Ok(vec)
}

fn parse_mesh(bytes: &[u8]) -> Result<IndexedMesh> {
    let mut reader = Cursor::new(bytes);
    let mesh = stl_io::read_stl(&mut reader)?;
    mesh.validate()?;
    Ok(mesh)
}

fn render(_mesh: &IndexedMesh) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    Ok(ImageBuffer::from_fn(WIDTH, HEIGHT, |x, y| {
        let sum = (x + y) as u8;
        Rgb([sum, sum, sum])
    }))
}
