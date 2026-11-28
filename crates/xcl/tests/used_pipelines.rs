//! The list of the pipelines a title used (`used-pipelines.txt`, beside the DXIL cache) belongs
//! to title runs. A test binary links the crate without `cfg(test)`, and it must neither build
//! pipelines ahead from the list nor add its own pipelines to it: the retail pipeline tests'
//! keys made a title build those pipelines ahead in its next run.

use std::time::Duration;
use xcl::hle::d3d12x::pso_async;

#[test]
fn a_test_binary_neither_reads_nor_writes_the_used_pipelines_list() {
    let dir = std::env::temp_dir().join(format!("xcl-used-pipelines-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let list = dir.join("used-pipelines.txt");
    // a list an earlier run wrote (this binary's first pipeline call reads it, if anything does)
    let earlier = "000000005eed0003\n";
    std::fs::write(&list, earlier).unwrap();
    std::env::set_var("XCL_DXIL_CACHE", &dir);

    // a pipeline of the shaders in the list: not built ahead, only when used
    let (known, unknown) = (0x7E57_8001usize, 0x7E57_8002usize);
    pso_async::submit_lazy(known, false, Box::new(|| 0x4323), 0x5EED_0003);
    std::thread::sleep(Duration::from_millis(500));
    let built_ahead = !pso_async::is_pending(known);
    // a pipeline used for the first time: not added to the list
    pso_async::submit_lazy(unknown, false, Box::new(|| 0x4324), 0x5EED_0004);
    assert_eq!(pso_async::resolve_wait(unknown), 0x4324);
    assert_eq!(pso_async::resolve_wait(known), 0x4323);
    pso_async::released(known);
    pso_async::released(unknown);
    let after = std::fs::read_to_string(&list).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);

    assert!(!built_ahead, "a test binary built a pipeline ahead from the user's list");
    assert_eq!(after, earlier, "a test binary added its pipelines to the user's list");
}
