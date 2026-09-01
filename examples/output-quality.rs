#![allow(clippy::print_stdout)]

#[cfg(feature = "cuda")]
mod output_quality;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    output_quality::run()
}

#[cfg(not(feature = "cuda"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("the output-quality example requires --features cuda".into())
}
