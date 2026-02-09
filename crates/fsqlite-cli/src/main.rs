fn main() {}

#[cfg(test)]
mod tests {
    #[test]
    fn test_cli_process_exposes_argv0() {
        let argv0 = std::env::args_os()
            .next()
            .expect("test process should expose argv[0]");
        assert!(!argv0.is_empty(), "argv[0] should not be empty");
    }
}
