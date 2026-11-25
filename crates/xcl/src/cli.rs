//! Argument handling shared by the subcommands. An option that needs a value is an error when
//! the value is missing or is not the kind it expects, and an argument a subcommand does not
//! know is reported: both used to be dropped silently, so a typo changed what the command did
//! and it still exited 0 (#2266). `xcl disasm` has done this all along.

/// `name value` of `args` as text: `None` when absent, an error when the value is missing or is
/// another option.
pub fn text<'a>(args: &'a [String], name: &str) -> Result<Option<&'a str>, String> {
    let Some(index) = args.iter().position(|arg| arg == name) else { return Ok(None) };
    match args.get(index + 1) {
        Some(value) if !value.starts_with("--") => Ok(Some(value)),
        _ => Err(format!("{name} needs a value")),
    }
}

/// `name value` as a number: decimal, or hexadecimal with `0x`.
pub fn number(args: &[String], name: &str) -> Result<Option<usize>, String> {
    let Some(value) = text(args, name)? else { return Ok(None) };
    let (digits, radix) = match value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
        Some(hex) => (hex, 16),
        None => (value, 10),
    };
    usize::from_str_radix(digits, radix).map(Some).map_err(|_| format!("invalid value for {name}: {value}"))
}

/// `name value` as a 64-bit hexadecimal number, `0x` allowed (`--hash`).
pub fn hex_u64(args: &[String], name: &str) -> Result<Option<u64>, String> {
    let Some(value) = text(args, name)? else { return Ok(None) };
    let digits = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")).unwrap_or(value);
    u64::from_str_radix(digits, 16).map(Some).map_err(|_| format!("invalid value for {name}: {value}"))
}

/// The first argument that looks like an option and is not one the subcommand knows.
pub fn unknown<'a>(args: &'a [String], known: &[&str]) -> Option<&'a str> {
    args.iter().map(String::as_str).find(|arg| arg.starts_with('-') && !known.contains(arg))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_value_that_is_missing_or_another_option_is_an_error() {
        assert_eq!(text(&args(&["--find", "x"]), "--find").unwrap(), Some("x"));
        assert_eq!(text(&args(&[]), "--find").unwrap(), None);
        assert!(text(&args(&["--find"]), "--find").unwrap_err().contains("needs a value"));
        assert!(text(&args(&["--find", "--full"]), "--find").unwrap_err().contains("needs a value"));
    }

    #[test]
    fn numbers_are_checked_decimal_or_hexadecimal_not_defaulted() {
        assert_eq!(number(&args(&["--top", "42"]), "--top").unwrap(), Some(42));
        assert_eq!(number(&args(&["--top", "0x10"]), "--top").unwrap(), Some(16));
        assert_eq!(number(&args(&[]), "--top").unwrap(), None);
        let e = number(&args(&["--top", "abc"]), "--top").unwrap_err();
        assert!(e.contains("invalid value for --top: abc"), "{e}");
        assert!(number(&args(&["--top", "-1"]), "--top").is_err());
        assert_eq!(hex_u64(&args(&["--hash", "0x1a2b"]), "--hash").unwrap(), Some(0x1a2b));
        assert!(hex_u64(&args(&["--hash", "0xZZ"]), "--hash").unwrap_err().contains("invalid value"));
    }

    #[test]
    fn an_option_no_one_knows_is_named() {
        let known = ["--top", "--full"];
        assert_eq!(unknown(&args(&["--top", "3", "--full"]), &known), None);
        assert_eq!(unknown(&args(&["--top", "3", "--fnid", "foo"]), &known), Some("--fnid"));
        // (values are not looked at: only arguments that look like options are)
        assert_eq!(unknown(&args(&["--top", "x"]), &known), None);
    }
}
