//! Device-only shader preparation; native context use belongs to the UI adapter.
use std::sync::{
    Arc, Mutex, OnceLock, Weak,
    atomic::{AtomicU64, Ordering},
};
use windows::{Win32::Graphics::Direct3D11::*, core::Interface};
static CREATED: AtomicU64 = AtomicU64::new(0);
static CACHE_HITS: AtomicU64 = AtomicU64::new(0);
const MAX_DEVICE_CACHES: usize = 8;
const MAX_PROGRAMS_PER_DEVICE: usize = 64;
struct ProgramCache {
    device: ID3D11Device,
    programs: Mutex<Vec<Weak<Program>>>,
}
struct Program {
    vertex: ID3D11VertexShader,
    fragment: ID3D11PixelShader,
    bytecode: Arc<[u8]>,
    // A live program retains its device cache; weak entries release dead programs.
    _cache: Arc<ProgramCache>,
}
fn device_cache(device: &ID3D11Device) -> anyhow::Result<Arc<ProgramCache>> {
    static CACHES: OnceLock<Mutex<Vec<Weak<ProgramCache>>>> = OnceLock::new();
    let mut caches = CACHES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .map_err(|_| anyhow::anyhow!("device program cache poisoned"))?;
    caches.retain(|cache| cache.strong_count() != 0);
    for cache in caches.iter().filter_map(Weak::upgrade) {
        if cache.device.as_raw() == device.as_raw() {
            return Ok(cache);
        }
    }
    anyhow::ensure!(caches.len() < MAX_DEVICE_CACHES, "background device-cache admission exceeded");
    let cache = Arc::new(ProgramCache { device: device.clone(), programs: Mutex::new(Vec::new()) });
    caches.push(Arc::downgrade(&cache));
    Ok(cache)
}
impl ProgramCache {
    fn prepare(self: &Arc<Self>, bytecode: Arc<[u8]>) -> anyhow::Result<Arc<Program>> {
        let mut programs = self
            .programs
            .lock()
            .map_err(|_| anyhow::anyhow!("background program cache poisoned"))?;
        programs.retain(|program| program.strong_count() != 0);
        for program in programs.iter().filter_map(Weak::upgrade) {
            if program.bytecode.as_ref() == bytecode.as_ref() {
                CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                return Ok(program);
            }
        }
        anyhow::ensure!(
            programs.len() < MAX_PROGRAMS_PER_DEVICE,
            "background program-cache admission exceeded"
        );
        let mut vertex = None;
        let mut fragment = None;
        // Only background factories enter this mutex; driver work cannot block paint.
        unsafe {
            self.device.CreateVertexShader(
                include_bytes!("background_triangle.dxbc"),
                None,
                Some(&mut vertex),
            )?;
            self.device.CreatePixelShader(&bytecode, None, Some(&mut fragment))?;
        }
        let program = Arc::new(Program {
            vertex: vertex.ok_or_else(|| anyhow::anyhow!("missing background vertex program"))?,
            fragment: fragment
                .ok_or_else(|| anyhow::anyhow!("missing background fragment program"))?,
            bytecode,
            _cache: self.clone(),
        });
        programs.push(Arc::downgrade(&program));
        CREATED.fetch_add(1, Ordering::Relaxed);
        Ok(program)
    }
}
pub(super) struct NativeBackgroundShader {
    pub vertex: ID3D11VertexShader,
    pub fragment: ID3D11PixelShader,
    pub target: ID3D11RenderTargetView,
    pub bytecode: Arc<[u8]>,
    pub uniforms: ID3D11Buffer,
    _program: Arc<Program>,
}
impl NativeBackgroundShader {
    pub fn new(
        device: &ID3D11Device,
        texture: &ID3D11Texture2D,
        fragment: &[u8],
    ) -> anyhow::Result<Self> {
        let program = device_cache(device)?.prepare(Arc::from(fragment))?;
        let mut target = None;
        unsafe { device.CreateRenderTargetView(texture, None, Some(&mut target)) }?;
        let mut uniforms = None;
        // 每个纹理所有者只保留一个常量缓冲；时间更新不新建纹理或重新编译。
        unsafe {
            device.CreateBuffer(
                &D3D11_BUFFER_DESC {
                    ByteWidth: 16,
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                    ..Default::default()
                },
                None,
                Some(&mut uniforms),
            )?;
        }
        Ok(Self {
            vertex: program.vertex.clone(),
            fragment: program.fragment.clone(),
            target: target.ok_or_else(|| anyhow::anyhow!("missing background render target"))?,
            bytecode: program.bytecode.clone(),
            uniforms: uniforms.ok_or_else(|| anyhow::anyhow!("missing background uniforms"))?,
            _program: program,
        })
    }
}
pub(super) fn statistics() -> (u64, u64) {
    (CREATED.load(Ordering::Acquire), CACHE_HITS.load(Ordering::Acquire))
}
