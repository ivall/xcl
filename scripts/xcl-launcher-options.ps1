function Get-XclOptions {
    param([string] $SourceRoot)

    # Flag means presence: even "0" enables these flags. Boolean accepts an explicit 0 or 1.
    $definitions = @'
Name|Area|Kind|Description
ACTIVITY|Diagnostics|Flag|Log GPU work, storage activity and memory use every two seconds.
THREAD_DUMP_AT|Diagnostics|Value|Capture thread stacks at elapsed seconds, for example 30,120,180,205,210.
TRACE_FAULTS|Diagnostics|Flag|Log the first 32 access violations before the game handles them.
RUN_SECONDS|Launch|Value|Finish the run and write its report after this many seconds.
NETWORK|Launch|Boolean|Set to 0 for XCL's offline socket behavior.
DATA_ROOT|Launch|Value|Root folder for title settings, cache, saves, overlay and logs.
WINDOW_POS|Display|Value|Window position: auto, center, or left,top in screen pixels.
ADAPTER|Graphics|Value|Select a GPU by adapter index or part of its name.
RAYTRACING|Graphics|Value|Raytracing tier to report: auto, off, tier1_0 or tier1_1.
D3D12_DIR|Graphics|Value|Folder containing the D3D12 Agility SDK DLLs.
D3D12_DEBUG|Graphics|Flag|Enable the D3D12 debug layer.
DRED|Graphics|Flag|Enable D3D12 device removal diagnostics.
D3D12_TRACE|Graphics|Flag|Log additional D3D12 operations.
PSO_WAIT_MS|Pipelines|Value|Maximum wait in milliseconds for a pending pipeline, e.g. 50. Late dispatches may be skipped.
ASYNC_PSO|Pipelines|Boolean|Set to 0 to build pipelines synchronously.
LAZY_PSO|Pipelines|Boolean|Set to 0 to disable lazy pipeline creation.
PSO_PRELOAD|Pipelines|Boolean|Set to 0 to disable background pipeline preloading.
PSO_DISK_CACHE|Pipelines|Boolean|Set to 0 to disable caching compute pipeline driver blobs on disk.
TRANSLATED_CACHE|Shaders|Boolean|Set to 0 to disable the disk cache of translated shaders.
DXC_DIR|Shaders|Value|Folder containing the x64 DXC compiler and validator DLLs.
DXC_THREADS|Shaders|Value|Concurrent shader compiler jobs (1 to 64).
DXIL_CACHE|Shaders|Value|Folder for compiled shader cache files.
DUMP_SHADERS|Shaders|Boolean|Set to 1 to save captured shaders.
RECOMP_DXC|Shaders|Boolean|Set to 1 to compile translated HLSL while dumping shaders.
DUMP_GS|Shaders|Value|Folder for geometry shader dumps.
SPLIT_WAVE64|Shaders|Boolean|Set to 1 to force testing the split wave64 shader path.
DEBUG_PS_EXPR|Shaders|Value|Override a pixel shader output using hexhash:HLSL_expression.
DEBUG_PS_SUB|Shaders|Value|Override pixel shader HLSL using hexhash:from=>to.
DEBUG_NO_STENCIL|Shaders|Value|Disable stencil for comma-separated pixel shader hashes in hex.
DEBUG_NO_CULL|Shaders|Value|Disable culling for comma-separated pixel shader hashes in hex.
DEBUG_LOD0|Shaders|Value|Force level zero sampling for comma-separated pixel shader hashes in hex.
TRACE_SHADERS|Shaders|Value|Comma-separated vertex or pixel shader hashes in hex to trace.
TRACE_MANY|Shaders|Value|Maximum number of matching draws to trace.
TRACE_UI_ANY|Shaders|Flag|Allow draw tracing on any render target.
TRACE_ADDRESSES|Diagnostics|Flag|Trace console root descriptor addresses.
TRACE_FRAME|Diagnostics|Flag|Trace frame presentation.
TRACE_FRAME_EVENTS|Diagnostics|Flag|Trace the first 40 frame event observations.
TRACE_APU|Audio|Boolean|Set to 1 to enable audio processor tracing.
OPUS_DIR|Audio|Value|Directory containing the optional opus.dll decoder.
FRAME_DUMP|Diagnostics|Value|Dump a frame every N presentations (N greater than 1; 1 selects 120).
FRAME_DUMP_FULL|Diagnostics|Flag|Sample every pixel in frame dumps.
TRACE_IMPORTS|Diagnostics|Value|Trace hle, all, or comma-separated dll!Function prefixes. Prefix exclusions with a minus sign.
TRACE_MEM|Memory|Value|Enable memory tracing. An integer above 1 sets the kernel allocation log limit.
UMA_MB|Memory|Value|Size in MiB of the title unified memory region.
COHERENT_MB|Memory|Value|Size in MiB of the coherent memory region.
UMA_CHECK|Memory|Boolean|Set to 0 to skip the coherent memory startup check.
COHERENT_HEAPS|Memory|Boolean|Set to 0 to disable placement of eligible allocations in coherent heaps.
TEXTURE_SKIP_UNCHANGED|Graphics|Boolean|Set to 0 to disable skipping unchanged title texture uploads.
SYNC_SMALL_TARGETS|Graphics|Boolean|Set to 0 to disable synchronization of small render targets.
WAIT_TIMEOUT_MS|Graphics|Value|Timeout in milliseconds for console GPU value waits.
SPLIT_MIDLIST_WAIT|Graphics|Boolean|Set to 0 to disable splitting a command list at a wait.
LATE_RELEASE|Graphics|Boolean|Set to 0 to disable deferred resource release.
CONSOLE_LISTS|Graphics|Flag|Enable console command lists.
CONSOLE_LISTS_LOG|Diagnostics|Flag|Enable console command list logging.
LIST_TRACE|Diagnostics|Flag|Enable title command list tracing.
WATCH|Diagnostics|Value|Hexadecimal game-relative address to watch for writes.
PROBE|Diagnostics|Value|Up to three comma-separated hexadecimal game-relative probe addresses.
WATCH_FENCE|Diagnostics|Value|Watch for writes over the first eight bytes of the Nth fence.
WATCH_DMA|Diagnostics|Flag|Watch writes to the DMA command list object.
WATCH_DEVICE|Diagnostics|Flag|Watch writes to the D3D12 device object.
SKIP_CPU_CHECK|Launch|Flag|Skip the launcher's CPU compatibility check.
DEBUG_MAGENTA|Shaders|Value|Replace pixel shader outputs with colors: hexhash[:rrggbb] entries separated by commas.
VRAM_ARENAS|Memory|Flag|Use VRAM arenas for texture placement instead of the title's mapped pages.
RECOMP_NO_STRUCTURED|Shaders|Flag|Disable structured control flow when recompiling shaders.
X64_ALL_MASKS|Shaders|Flag|Disable the optimization that skips exchanges for lane-local wave64 mask reads.
RECOMP_WHY|Shaders|Flag|Log extra shader recompiler diagnostics.
WAVE64_SPIN|Shaders|Boolean|Control spin synchronization for eligible split-wave64 programs.
SAFE_BARRIER|Shaders|Boolean|Set to 0 to disable the recompiler's safe barrier mode.
DSTORAGE_BCPACK_FILL|Storage|Boolean|Enable the optional DirectStorage BCPack fill path.
DUMP_EVERY|Shader CLI|Value|Save every Nth complete shader translation when dumping HLSL.
DUMP_HLSL|Shader CLI|Value|Folder where the shader command saves translated HLSL.
EXAMPLE_HLSL|Shader CLI|Value|Write the example shader's HLSL to this file.
PIPELINE_THREADS|Shader CLI|Value|Maximum concurrent driver pipeline compilations in the shader command.
RECOMP_ONLY|Shader CLI|Value|Restrict recompilation to the specified hexadecimal shader name hash.
RECOMP_WINDOWS|Shader CLI|Value|Set the number of memory windows for shader command recompilation.
'@ | ConvertFrom-Csv -Delimiter '|'

    $metadata = @{}
    foreach ($entry in $definitions) {
        $entry.Name = 'XCL_' + $entry.Name
        $metadata[$entry.Name] = $entry
    }
    $src = if ($SourceRoot) { Join-Path $SourceRoot 'crates\xcl\src' }
    if (-not $src -or -not [IO.Directory]::Exists($src)) {
        foreach ($name in 'XCL_ACTIVITY','XCL_TRACE_FAULTS','XCL_THREAD_DUMP_AT','XCL_RUN_SECONDS','XCL_DATA_ROOT','XCL_NETWORK','XCL_ADAPTER') {
            $metadata[$name]
        }
        return
    }

    $found = @{}
    # Also cover module environment-key constants and per-shader debug switches.
    $read = '(?:\benv::var(?:_os)?\s*\(\s*|\bdebug_pso_switch\s*\(\s*|\bconst\s+ENV\s*:\s*&str\s*=\s*)"(XCL_[A-Z0-9_]+)"'
    foreach ($file in Get-ChildItem -LiteralPath $src -Recurse -Filter '*.rs' -File) {
        $relative = $file.FullName.Substring($src.Length + 1)
        if ($relative -match '(^|[\\/])(tests?([\\/]|\.rs$)|bin[\\/])' -or $file.Name -like '*_spec.rs') { continue }
        $code = [IO.File]::ReadAllText($file.FullName)
        if (-not $code.Contains('XCL_')) { continue }
        $code = $code -creplace '(?ms)^#\[cfg\(test\)\]\s*(?:pub(?:\(crate\))?\s+)?mod\s+\w+\s*\{.*?^\}', ''
        $code = $code -creplace '(?m)^\s*//[^\r\n]*', ''
        foreach ($hit in [regex]::Matches($code, $read)) {
            $name = $hit.Groups[1].Value
            if ($found.ContainsKey($name) -or $name -match '^XCL_(HOST$|REQUIRE_GPU$|TEST_|SAMPLE_)') { continue }
            if ($metadata.ContainsKey($name)) {
                $found[$name] = $metadata[$name]
            } else {
                $area = if ($relative -eq 'shaders.rs') { 'Shader CLI' } else { 'Advanced' }
                $found[$name] = [pscustomobject]@{
                    Name = $name; Area = $area; Kind = 'Value'
                    Description = "Source option in $relative. Enter the value accepted by this build."
                }
            }
        }
    }
    $found.Values | Sort-Object Area,Name
}
